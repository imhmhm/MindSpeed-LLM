# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Isolate the scheme-E dscale failure: feed lite_pre_backward the exact
normalized logits l = L*rstd the op used and compare its dlogits/dscale/
dbase against the same math in torch on identical values."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault(
    'ASCEND_CUSTOM_OPP_PATH',
    str(Path.home() / 'work/dataset/huashan_zhh_guiyang_turbo/github/cann-ops/build_out'))

import torch  # noqa: E402
import torch_npu  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

from mindspeed_llm.ops.triton.mhc_lite_heads import lite_pre_backward  # noqa: E402

torch.manual_seed(3)
torch_npu.npu.set_device(0)
DEV = torch.device('npu:0')

SB, E, H, NP = 64, 4, 1024, 24
N32 = 8 + NP
EPS = 1e-5

TORCH_NPU = Path(torch_npu.__file__).parent
ext = load(
    name='ascendc_lite_pre_ext',
    sources=[str(Path(__file__).parent / 'ascendc_lite_pre/extension.cpp')],
    extra_include_paths=[
        str(TORCH_NPU / 'include'),
        str(TORCH_NPU / 'include/third_party/acl/inc'),
        '/usr/local/Ascend/cann-9.1.1/python/site-packages/cann_ops_transformer/common/inc',
    ],
    extra_ldflags=[f'-L{TORCH_NPU}/lib', '-ltorch_npu'],
    verbose=False,
)

x = (torch.randn(SB, E * H, device=DEV) * 1.5).to(torch.bfloat16)
w = torch.randn(N32, E * H, device=DEV) * 0.02
gamma = 1.0 + torch.randn(E * H, device=DEV) * 0.05
scale = torch.tensor([0.011, 0.013, 0.017], device=DEV)
base = (torch.randn(N32, device=DEV) * 0.5).contiguous()

perm_flat = torch.eye(E)[torch.tensor(
    list(__import__('itertools').permutations(range(E))))].flatten(1).to(DEV)
perm_t = perm_flat.t().contiguous()

xf = x
wp = (w.to(torch.bfloat16) * gamma.to(torch.bfloat16).view(1, -1))
L = torch.matmul(xf, wp.t())
lane = torch.tensor([0] * 4 + [1] * 4 + [2] * 24, device=DEV)
scale32 = scale[lane]

y, hpre8, hpost8, coeff, rstd = ext.lite_pre_heads(xf, L, scale32, base)
l = L.float() * rstd.unsqueeze(-1)

g = torch.randn(SB, H, device=DEV).to(torch.bfloat16)
ghp = torch.randn(SB, 4, device=DEV)
ghr = torch.randn(SB, 16, device=DEV)

dl, ds3, db32 = lite_pre_backward(g, x.view(SB, E, H), ghp, ghr, perm_t, l,
                                  scale32[[0, 4, 8]], base)

# ---- torch math on the identical l / tensors, mirroring the kernel formulas
lf = l
gf = g.float().view(SB, 1, H).expand(SB, E, H)
x3 = x.view(SB, E, H).float()
dw = (gf * x3).sum(-1)  # [SB,4] dots == the kernel's dw0..dw3

s0, s1, s2 = scale32[[0, 4, 8]]
z0 = lf[:, :4] * s0 + base[:4]
sig0 = torch.sigmoid(z0)
dz0 = dw * sig0 * (1 - sig0)  # [SB,4]
z1 = lf[:, 4:8] * s1 + base[4:8]
sig1 = torch.sigmoid(z1)
dz1 = ghp * 2 * sig1 * (1 - sig1)
z2 = lf[:, 8:] * s2 + base[8:]
coeff_t = torch.softmax(z2, -1)
dc = ghr @ perm_t  # [SB,24]
dzc = coeff_t * (dc - (dc * coeff_t).sum(-1, keepdim=True))

dl_ref = torch.cat([dz0 * s0, dz1 * s1, dzc * s2], -1)
ds_ref = torch.stack([(dz0 * lf[:, :4]).sum(), (dz1 * lf[:, 4:8]).sum(),
                      (dzc * lf[:, 8:]).sum()])
db_ref = torch.cat([dz0.sum(0), dz1.sum(0), dzc.sum(0)])

for name, mine, ref in (('dlogits', dl, dl_ref), ('dscale', ds3, ds_ref), ('dbase', db32, db_ref)):
    md = (mine - ref).abs().max().item()
    print(f'{name:8s} md {md:.3e}   mine { [f"{v:+.3e}" for v in mine.flatten()[:4].tolist()] }'
          f'  ref { [f"{v:+.3e}" for v in ref.flatten()[:4].tolist()] }')

# dw dots themselves (the y-path reduction feeding the pre head)
dw_ref = dw
print('dw magnitude (pre-head input):', dw.abs().max().item())

# ---- full wrapper path: leaf grads vs fp32 autograd, per-lane dscale
from mindspeed_llm.ops.ascendc.mhc_lite_ac import lite_pre_ascendc  # noqa: E402

SB2 = 1024
x2 = (torch.randn(SB2, E, H, device=DEV) * 1.5).to(torch.bfloat16)
xg = x2.detach().view(SB2, 1, E, H).requires_grad_()
wg = w.detach().requires_grad_()
gg = gamma.detach().requires_grad_()
sg = scale.detach().requires_grad_()
bg = base.detach().requires_grad_()

y, hp, hr = lite_pre_ascendc(
    xg, wg, gg, sg, bg, perm_flat, perm_t, EPS)
gy = torch.randn_like(y)
ghp2 = torch.randn_like(hp)
ghr2 = torch.randn_like(hr)
grads = torch.autograd.grad((y, hp, hr), (xg, wg, gg, sg, bg), (gy, ghp2, ghr2))

xr = x2.detach().view(SB2, 1, E, H).float().requires_grad_()
wr2 = w.detach().requires_grad_()
gr2 = gamma.detach().requires_grad_()
sr2 = scale.detach().requires_grad_()
br2 = base.detach().requires_grad_()


def ref_chain(xr, wr2, gr2, sr2, br2):
    xf32 = xr.view(SB2, E * H)
    rstd_r = torch.rsqrt(xf32.pow(2).mean(-1, keepdim=True) + EPS)
    logits_r = (xf32 * rstd_r * gr2) @ wr2.t()
    pre_l, post_l, res_l = torch.split(logits_r, [E, E, NP], -1)
    h_pre = torch.sigmoid(pre_l * sr2[0] + br2[:E])
    h_post = 2 * torch.sigmoid(post_l * sr2[1] + br2[E:2 * E])
    coeff = torch.softmax(res_l * sr2[2] + br2[2 * E:], -1)
    h_res = coeff @ perm_flat
    y_r = (h_pre.unsqueeze(-1) * xr.view(SB2, E, H)).sum(1)
    return y_r, h_post, h_res


yr, hpr, hrr = ref_chain(xr, wr2, gr2, sr2, br2)
rgrads = torch.autograd.grad((yr, hpr, hrr), (xr, wr2, gr2, sr2, br2),
                             (gy.view(SB2, H).float(),
                              ghp2.reshape(SB2, E), ghr2.reshape(SB2, E * E)))

names = ('dx', 'dW', 'dgamma', 'dscale', 'dbase')
for name, mine, ref in zip(names, grads, rgrads):
    md = (mine.float() - ref).abs().max().item()
    extra = ''
    if name == 'dscale':
        extra = f'  mine {[f"{v:+.3e}" for v in mine.tolist()]} ref {[f"{v:+.3e}" for v in ref.tolist()]}'
    print(f'{name:8s} md {md:.3e} / max|ref| {ref.abs().max().item():.3e}{extra}')

# ---- mechanism: is the residual dscale error the bf16 GEMM rounding?
# rebuild ds with the identical torch math, feeding l from a bf16 GEMM vs
# an fp32 GEMM; only the logits precision changes
gf2 = gy.view(SB2, 1, H).float().expand(SB2, E, H)
dw2 = (gf2 * x2.view(SB2, E, H).float()).sum(-1)

ref_ds = rgrads[3]
L_bf = torch.matmul(x2.view(SB2, E * H), wp.t()).contiguous()
L_fp = torch.matmul(x2.view(SB2, E * H).float(), (w * gamma.view(1, -1)).t()).contiguous()
rstd_free = torch.rsqrt(x2.view(SB2, E * H).float().pow(2).mean(-1) + EPS)


def ds_rel(lt, ref_ds_in, base_in, dw_in):
    z0 = lt[:, :4] * scale[0] + base_in[:4]
    s0p = torch.sigmoid(z0)
    dz0 = dw_in * s0p * (1 - s0p)
    z1 = lt[:, 4:8] * scale[1] + base_in[4:8]
    s1p = torch.sigmoid(z1)
    dz1 = ghp2.view(SB2, E) * 2 * s1p * (1 - s1p)
    z2 = lt[:, 8:] * scale[2] + base_in[8:]
    cf = torch.softmax(z2, -1)
    dc = ghr2.view(SB2, E * E) @ perm_t
    dzc = cf * (dc - (dc * cf).sum(-1, keepdim=True))
    d = torch.stack([(dz0 * lt[:, :4]).sum(), (dz1 * lt[:, 4:8]).sum(),
                     (dzc * lt[:, 8:]).sum()])
    return (d - ref_ds_in).abs().max() / ref_ds_in.abs().max()


for tag, lt in (('bf16-GEMM logits', L_bf.float() * rstd_free.unsqueeze(-1)),
                ('fp32-GEMM logits', L_fp * rstd_free.unsqueeze(-1))):
    print(f'dscale from {tag:16s} (random base): rel err vs fp32 autograd '
          f'{ds_rel(lt, ref_ds, base, dw2).item():.3e}')

# structured base as build_lite initializes it: saturated sigmoid heads and
# a near-one-hot res softmax -- the regime the e2e harness actually gates
base_h = torch.zeros(N32, device=DEV)
base_h[:E] = -8.0
base_h[1] = 8.0
base_h[2 * E:] = -8.0
base_h[2 * E] = 0.0
base_h = (base_h + torch.randn(N32, device=DEV) * 0.5).contiguous()

sgh = sg
bh = base_h.clone().requires_grad_()
y3o, hp3o, hr3o = lite_pre_ascendc(xg, wg, gg, sgh, bh, perm_flat, perm_t, EPS)
g3 = torch.autograd.grad((y3o, hp3o, hr3o), (xg, wg, gg, sgh, bh), (gy, ghp2, ghr2))

shr = sgh.detach().float().requires_grad_()
bhr = bh.detach().float().requires_grad_()
wr3 = wg.detach().float().requires_grad_()
gr3 = gg.detach().float().requires_grad_()
xr3 = xg.detach().float().requires_grad_()
rstd3 = torch.rsqrt(xr3.view(SB2, E * H).pow(2).mean(-1, keepdim=True) + EPS)
lr3 = (xr3.view(SB2, E * H) * rstd3 * gr3) @ wr3.t()
pre3, post3, res3 = torch.split(lr3, [E, E, NP], -1)
hpr3 = 2 * torch.sigmoid(post3 * shr[1] + bhr[E:2 * E])
cfr = torch.softmax(res3 * shr[2] + bhr[2 * E:], -1)
hrr3 = cfr @ perm_flat
hpre3 = torch.sigmoid(pre3 * shr[0] + bhr[:E])
yr3 = (hpre3.unsqueeze(-1) * xr3.view(SB2, E, H)).sum(1)
rg3 = torch.autograd.grad((yr3, hpr3, hrr3), (xr3, wr3, gr3, shr, bhr),
                          (gy.view(SB2, H), ghp2.view(SB2, E), ghr2.view(SB2, E * E)))

for name, mine, ref in zip(names, g3, rg3):
    md = (mine.float() - ref).abs().max().item()
    extra = ''
    if name == 'dscale':
        extra = f'  mine {[f"{v:+.3e}" for v in mine.tolist()]} ref {[f"{v:+.3e}" for v in ref.tolist()]}'
    print(f'{name:8s} (structured base) md {md:.3e} / max|ref| {ref.abs().max().item():.3e}{extra}')

L_bf_h = torch.matmul(x2.view(SB2, E * H), (wg.detach().type_as(x2) * gg.detach().type_as(x2).view(1, -1)).t()).contiguous()
for tag, lt in (('bf16-GEMM logits', L_bf_h.float() * rstd_free.unsqueeze(-1)),
                ('fp32-GEMM logits', L_fp * rstd_free.unsqueeze(-1))):
    print(f'dscale from {tag:16s} (structured base): rel err '
          f'{ds_rel(lt, rg3[3], base_h, dw2).item():.3e}')

# ---- tier1 mechanism: the two bf16 chains produce different l at bf16
# rounding level (t0 rounds xn before the GEMM, ac multiplies by exact fp32
# rstd after); measure the dscale gap between the chains themselves
xn_bf, _ = torch_npu.npu_rms_norm(x2.view(SB2, E * H), gg.detach().type_as(x2), epsilon=EPS)
l_t0 = torch.matmul(xn_bf, wg.detach().type_as(x2).t()).float()
ds_ac = ds_rel(L_bf_h.float() * rstd_free.unsqueeze(-1), rg3[3], base_h, dw2)
ds_t0 = ds_rel(l_t0, rg3[3], base_h, dw2)
print(f'dscale chain gap (ac-form vs t0-form l, structured base): '
      f'{(ds_ac - ds_t0).abs().item():.3e}  (ac {ds_ac.item():.3e} / t0 {ds_t0.item():.3e} vs fp32)')

# ---- value-domain stress: near-zero x (rstd ~ 1/sqrt(eps), the eps-limited
# regime) and saturated scale on top of the structured base
for tag, x_scale, scale_sat in (('near-zero x', 0.01, None), ('saturated scale', 1.5, (0.2, 0.2, 0.2))):
    xz = (torch.randn(SB2, E, H, device=DEV) * x_scale).to(torch.bfloat16)
    xzg = xz.detach().view(SB2, 1, E, H).requires_grad_()
    szh = torch.full((3,), 0.2 if scale_sat else 0.011, device=DEV).requires_grad_()
    bz = base_h.clone().requires_grad_()
    yz, hpz, hrz = lite_pre_ascendc(xzg, wg, gg, szh, bz, perm_flat, perm_t, EPS)
    gz = torch.autograd.grad((yz, hpz, hrz), (xzg, wg, gg, szh, bz), (gy, ghp2, ghr2))

    zr = xz.detach().float().view(SB2, 1, E, H).requires_grad_()
    szr = szh.detach().requires_grad_()
    bzr = bz.detach().requires_grad_()
    wrz = wg.detach().float().requires_grad_()
    grz = gg.detach().float().requires_grad_()
    rstdz = torch.rsqrt(zr.view(SB2, E * H).pow(2).mean(-1, keepdim=True) + EPS)
    lz = (zr.view(SB2, E * H) * rstdz * grz) @ wrz.t()
    pre_z, post_z, res_z = torch.split(lz, [E, E, NP], -1)
    hpz_r = 2 * torch.sigmoid(post_z * szr[1] + bzr[E:2 * E])
    cfz = torch.softmax(res_z * szr[2] + bzr[2 * E:], -1)
    hrz_r = cfz @ perm_flat
    hpre_z = torch.sigmoid(pre_z * szr[0] + bzr[:E])
    yz_r = (hpre_z.unsqueeze(-1) * zr.view(SB2, E, H)).sum(1)
    rgz = torch.autograd.grad((yz_r, hpz_r, hrz_r), (zr, wrz, grz, szr, bzr),
                              (gy.view(SB2, H), ghp2.view(SB2, E), ghr2.view(SB2, E * E)))
    rels = [((m.float() - r).abs().max() / r.abs().max().clamp_min(1e-6)).item()
            for m, r in zip(gz, rgz)]
    print(f'{tag:16s} leaf-grad rel: '
          f'{" ".join(f"{n}:{v:.1e}" for n, v in zip(names, rels))}')
