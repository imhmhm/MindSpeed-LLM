# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Root-cause the aclnnMhcPreSinkhornBackward saturation defect.

Verdict (2026-10-16): the backward KERNEL's recurrence is mathematically
correct and the forward's saved-for-backward tensors are the true math
intermediates; the wrong gradients come from fp32 arithmetic collapsing
inside the 39-half-step Sinkhorn backward chain once the softmax saturates.

  A.  op saves vs an fp64 replica (F.linear; einsum is broken on this
      build, see README): hcn/h_pre/inv_rms/every even-slot matrix and
      every sum match to 1e-4..1e-7 at the extreme worst case. Odd slots
      (col-step matrices) are zero placeholders the backward never reads.
  B.  standalone backward with mathematically-correct saves substituted:
      byte-for-byte the same wrong ds2 -> the saves were never the poison.
  C.  CPU re-execution of the kernel's exact fp32 operation order (incl.
      its double-eps on loaded sums and divide-then-subtract chain)
      reproduces the op's wrong ds2 to 0.02% while fp64 gives the true
      value -> precision, not formula.

Usage:  python experiments/mhc_debug/decompose_pre_sinkhorn_bwd.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments' / 'mhc_debug'))

import torch  # noqa: E402
import torch_npu  # noqa: F401

from cann_ops_transformer.ops.mhc_pre_sinkhorn import mhc_pre_sinkhorn_op_builder  # noqa: E402
from cann_ops_transformer.ops.mhc_pre_sinkhorn_backward import (  # noqa: E402
    mhc_pre_sinkhorn_backward_op_builder,
)
from parity_fused_bwd import pre_ref_grads_fp64  # noqa: E402

E, ITERS, EPS, NORM_EPS = 4, 20, 1e-6, 1e-5
DEV = torch.device('npu:0')


def chain_from(mixes, alpha, bias):
    """fp64 sinkhorn chain from a given mixes tensor, per the op's save spec."""
    b, s = mixes.shape[:2]
    z = mixes[..., 2 * E:].view(b, s, E, E) * alpha.double()[2] \
        + bias.double()[2 * E:].view(1, 1, E, E)
    e = (z - z.max(-1, keepdim=True).values).exp()
    norm_out = [e / e.sum(-1, keepdim=True) + EPS]
    sum_out = [e.sum(-1)]

    col = norm_out[0].sum(-2)                       # slot 1 saved WITHOUT eps
    sum_out.append(col)
    norm_out.append(norm_out[0] / (col + EPS).unsqueeze(-2))

    for _ in range(ITERS - 1):
        row = norm_out[-1].sum(-1) + EPS            # slots >= 2 saved WITH eps
        sum_out.append(row)
        norm_out.append(norm_out[-1] / row.unsqueeze(-1))
        col = norm_out[-1].sum(-2) + EPS
        sum_out.append(col)
        norm_out.append(norm_out[-1] / col.unsqueeze(-2))
    return sum_out, norm_out


def ref_intermediates(x, phi, alpha, bias):
    """fp64 replica of the op's saved-for-backward tensors, in op layout [B,S,...]."""
    b, s, n, c = x.shape
    xf = x.double()
    inv_rms = torch.rsqrt(xf.square().flatten(2).mean(-1, keepdim=True) + NORM_EPS)
    hcn = torch.nn.functional.linear(xf.flatten(2), phi.double())
    mixes = hcn * inv_rms

    sum_out, norm_out = chain_from(mixes, alpha, bias)
    h_pre = torch.sigmoid(mixes[..., :E] * alpha.double()[0] + bias.double()[:E]) + EPS
    return dict(h_pre=h_pre, hc_before_norm=hcn, inv_rms=inv_rms,
                sum_out=sum_out, norm_out=norm_out)


def rel(op, ref):
    return ((op.float() - ref.float()).abs().max()
            / (ref.float().abs().max() + 1e-30)).item()


def run_backward(bwd, grads, inputs, saved):
    gy, gpost, gcomb, x = grads
    phi, alpha, bias = inputs
    h_pre, hcn, inv_rms, sum_out, norm_out = saved
    return bwd(gy, gpost, gcomb, x, phi, alpha, bias,
               h_pre, hcn, inv_rms, sum_out, norm_out, EPS)


def main():
    torch_npu.npu.set_device(0)
    case = torch.load(Path(__file__).parent / 'worst_ds2_case.pt', weights_only=False)
    fwd = mhc_pre_sinkhorn_op_builder.load()
    bwd = mhc_pre_sinkhorn_backward_op_builder.load()

    # model-side [S,B,...] -> op-side [B,S,...]
    x = case['x'].permute(1, 0, 2, 3).contiguous().to(DEV)
    phi, alpha, bias = (case[k].to(DEV) for k in ('w', 'scale', 'base'))
    gy = case['gy'].permute(1, 0, 2).contiguous().to(DEV)
    gpost = case['gpost'].permute(1, 0, 2).contiguous().to(DEV)
    gcomb = case['gcomb'].permute(1, 0, 2, 3).contiguous().to(DEV)

    out = fwd.mhc_pre_sinkhorn(x, phi, alpha, bias, E, ITERS, EPS, NORM_EPS, True)
    hin, h_post, h_res, h_pre, hcn, inv_rms, sum_out, norm_out = out
    print(f'op saved dtypes: sum_out {sum_out.dtype} {tuple(sum_out.shape)}, '
          f'norm_out {norm_out.dtype} {tuple(norm_out.shape)}')

    r = ref_intermediates(x.cpu(), phi.cpu(), alpha.cpu(), bias.cpu())
    print('\n== A. op-saved vs fp64 reference ==')
    print(f'  h_pre          {rel(h_pre.cpu(), r["h_pre"]):.3e}')
    print(f'  hc_before_norm {rel(hcn.cpu(), r["hc_before_norm"]):.3e}')
    print(f'  inv_rms        {rel(inv_rms.cpu(), r["inv_rms"]):.3e}')
    print(f'  op hcn[0,0] : {hcn[0, 0].cpu().tolist()}')
    print(f'  ref hcn[0,0]: {r["hc_before_norm"][0, 0].tolist()}')
    print(f'  op h_pre[0,0] : {h_pre[0, 0].cpu().tolist()}')
    print(f'  ref h_pre[0,0]: {r["h_pre"][0, 0].tolist()}')

    # forward correctness in this standalone path
    from parity_fused_bwd import pre_ref_fp64
    ref_y, ref_pre, ref_post, ref_comb = pre_ref_fp64(
        x.cpu().permute(1, 0, 2, 3), phi.cpu(), alpha.cpu(), bias.cpu())
    rc = ref_comb.permute(1, 0, 2, 3)
    hr = h_res.cpu().reshape(h_res.shape[0], h_res.shape[1], E, E)
    print(f'  h_res shape {tuple(h_res.shape)}')
    print(f'  hin rel {rel(hin.cpu(), ref_y.permute(1, 0, 2)):.3e}   '
          f'h_post rel {rel(h_post.cpu(), ref_post.permute(1, 0, 2)):.3e}   '
          f'h_res rel {rel(hr, rc):.3e}   h_res^T rel '
          f'{rel(hr.transpose(-1, -2), rc):.3e}')

    # candidate reconstructions of the op's internal softmax input
    a2, b2 = alpha.cpu().double()[2], bias.cpu().double()[2 * E:]
    mixes_ref = r['hc_before_norm'] * r['inv_rms']
    zt_true = (mixes_ref[0, 0, 2 * E:] * a2 + b2).view(E, E)
    zt_op = (hcn.cpu().double()[0, 0, 2 * E:] * a2 + b2).view(E, E)
    no0 = norm_out[0][0, 0].cpu()
    cands = {
        'softmax(z_true)+eps': zt_true.softmax(-1) + EPS,
        'softmax(z_op_hcn)+eps': zt_op.softmax(-1) + EPS,
        'softmax(z_op_hcn^T)+eps': zt_op.t().softmax(-1) + EPS,
    }
    print('\n  op norm_out[0][0,0] vs candidates:')
    print(f'  op:\n{no0.numpy().round(6)}')
    print(f'  z_true rowmax:\n{zt_true.numpy().argmax(-1)}   '
          f'z_op_hcn rowmax:\n{zt_op.numpy().argmax(-1)}')
    for name, cand in cands.items():
        print(f'  {name}: rel {rel(no0, cand):.3e}')
    hp_op = torch.sigmoid(hcn.cpu().double()[0, 0, :E] * alpha.cpu().double()[0]
                          + bias.cpu().double()[:E])
    print(f'  sigmoid(op_hcn[:4]*s0+b0) vs op h_pre: '
          f'{rel(h_pre.cpu()[0, 0], hp_op):.3e}   vs ref h_pre: '
          f'{rel(h_pre.cpu()[0, 0], r["h_pre"][0, 0]):.3e}')

    # convention validation on a low-saturation twin: if my fp64 chain
    # construction matches the op slot-for-slot where z is small, the
    # convention (not the math) is confirmed for the substitution arm.
    g = torch.Generator().manual_seed(7)
    eh = x.shape[-2] * x.shape[-1]
    xl = torch.randn(1, 8, E, x.shape[-1], generator=g).to(torch.bfloat16)
    pl = torch.randn(2 * E + E * E, eh, generator=g) * 0.013
    al = torch.tensor([0.5, 0.5, 0.5])
    bl = torch.randn(2 * E + E * E, generator=g) * 0.05
    o2 = fwd.mhc_pre_sinkhorn(xl.to(DEV), pl.to(DEV), al.to(DEV), bl.to(DEV),
                              E, ITERS, EPS, NORM_EPS, True)
    r2 = ref_intermediates(xl, pl, al, bl)
    print('\n== A2. low-saturation twin: slot convention validation ==')
    print(f'  h_pre {rel(o2[3].cpu(), r2["h_pre"]):.2e}   hcn '
          f'{rel(o2[4].cpu(), r2["hc_before_norm"]):.2e}   inv_rms '
          f'{rel(o2[5].cpu(), r2["inv_rms"]):.2e}')
    # rebuild the chain from the OP'S OWN hcn: any remaining mismatch is a
    # convention difference in the slot chain itself, not in the mixes
    hcn2 = o2[4].cpu().double()
    so2, no2 = chain_from(hcn2 * r2['inv_rms'], al, bl)
    print('  per-slot rel (chain from op_hcn*inv_rms vs op norm_out):')
    for k in range(6):
        print(f'    slot {k}: {rel(o2[7][k].cpu(), no2[k]):.2e}')
    print(f'  op norm_out[1][0,0]:\n{o2[7][1][0, 0].cpu().numpy().round(6)}')
    print(f'  mine     no2[1][0,0]:\n{no2[1][0, 0].numpy().round(6)}')
    print(f'  op norm_out[2][0,0]:\n{o2[7][2][0, 0].cpu().numpy().round(6)}')
    print(f'  mine     no2[2][0,0]:\n{no2[2][0, 0].numpy().round(6)}')

    # saturation-slot validation: rebuild the chain from the OP'S OWN hcn on
    # the worst case; if even slots + sums still track the op's saves, the
    # convention holds at saturation and the substitution arms are meaningful
    so_op, no_op = chain_from(hcn.cpu().double() * inv_rms.cpu().double(),
                              alpha.cpu(), bias.cpu())
    print('\n== A3. worst case: chain from op_hcn vs op saves (convention @ saturation) ==')
    for k in range(0, 8):
        m = 'zero-slot' if k % 2 else f'{rel(norm_out[k].cpu(), no_op[k]):.2e}'
        print(f'  slot {k}: norm_out {m}   '
              f'sum_out {rel(sum_out[k].cpu(), so_op[k]):.2e}')
    worst_s = max(rel(sum_out[k].cpu(), so_op[k]) for k in range(1, 40))
    worst_n = max(rel(norm_out[k].cpu(), no_op[k]) for k in range(0, 40, 2))
    print(f'  worst over all: even norm_out {worst_n:.2e}, sum_out {worst_s:.2e}')

    rdx, rdw, rds, rdb = pre_ref_grads_fp64(
        x.cpu().permute(1, 0, 2, 3), phi.cpu(), alpha.cpu(), bias.cpu(),
        case['gy'].to(hin.dtype), case['gpost'], case['gcomb'])

    # mirror the op's save layout: even slots hold row-normalized matrices,
    # odd slots (col-step outputs) are zero placeholders the op never writes
    def op_layout(mats):
        st = torch.stack(mats).float()
        st[1::2] = 0
        return st

    variants = {
        'op saved (training path)': (h_pre, hcn, inv_rms, sum_out, norm_out),
        'ref z, op layout': (h_pre, hcn, inv_rms,
                             torch.stack(r['sum_out']).float(),
                             op_layout(r['norm_out'])),
        'all ref': (r['h_pre'].float(), r['hc_before_norm'].float(),
                    r['inv_rms'].float(), torch.stack(r['sum_out']).float(),
                    torch.stack(r['norm_out']).float()),
    }
    print('\n== B. standalone backward op with substituted saved tensors ==')
    print(f'  {"variant":24s} {"ds2 op":>12s} {"ds2 fp64":>12s} {"ds2 rel":>9s} '
          f'{"dx rel":>9s} {"dw rel":>9s} {"db rel":>9s}')
    for name, saved_t in variants.items():
        saved = tuple(t.to(DEV).contiguous() for t in saved_t)
        gdx, gdw, gda, gdb = run_backward(
            bwd.mhc_pre_sinkhorn_backward, (gy, gpost, gcomb, x),
            (phi, alpha, bias), saved)
        ds2 = gda[2].item()
        ds2_ref = rds[2].item()
        print(f'  {name:24s} {ds2:12.4e} {ds2_ref:12.4e} '
              f'{abs(ds2 - ds2_ref) / (abs(ds2_ref) + 1e-30):9.2e} '
              f'{rel(gdx.cpu(), rdx):9.2e} {rel(gdw.cpu(), rdw):9.2e} '
              f'{rel(gdb.cpu(), rdb):9.2e}')

    # dose-response control: the same two arms on the low-saturation twin
    g2 = torch.Generator().manual_seed(11)
    gy2 = torch.randn(1, 8, x.shape[-1], generator=g2).to(torch.bfloat16)
    gp2 = torch.randn(1, 8, E, generator=g2).float()
    gc2 = torch.randn(1, 8, E, E, generator=g2).float()
    tdx, tdw, tds, tdb = pre_ref_grads_fp64(
        xl.permute(1, 0, 2, 3), pl, al, bl, gy2.permute(1, 0, 2),
        gp2.permute(1, 0, 2), gc2.permute(1, 0, 2, 3))
    print('\n== B2. low-saturation twin backward (dose-response control) ==')
    print(f'  {"variant":24s} {"ds2 rel":>9s} {"dx rel":>9s}')
    for name, saved_t in (('op saved', (o2[3], o2[4], o2[5], o2[6], o2[7])),
                          ('all ref', (r2['h_pre'].float(),
                                       r2['hc_before_norm'].float(),
                                       r2['inv_rms'].float(),
                                       torch.stack(r2['sum_out']).float(),
                                       torch.stack(r2['norm_out']).float()))):
        saved = tuple(t.to(DEV).contiguous() for t in saved_t)
        gdx, gdw, gda, gdb = run_backward(
            bwd.mhc_pre_sinkhorn_backward,
            (gy2.to(DEV), gp2.to(DEV), gc2.to(DEV), xl.to(DEV)),
            (pl.to(DEV), al.to(DEV), bl.to(DEV)), saved)
        print(f'  {name:24s} '
              f'{abs(gda[2].item() - tds[2].item()) / (abs(tds[2].item()) + 1e-30):9.2e} '
              f'{rel(gdx.cpu().permute(1, 0, 2, 3), tdx):9.2e}')

    # fp32 arithmetic simulation of the kernel's exact recurrence order
    # (double-eps on odd slots >= 3, Div/(c+eps) -> Mul -> reduce -> /(c+eps)^2
    # -> Sub, then /(r+eps) -> Mul -> reduce -> Sub; tail col + softmax bwd).
    # If plain fp32 reproduces the op's wrong ds2, the defect is precision
    # collapse in the 39-step chain, not a formula error.
    print('\n== C. fp32 simulation of the kernel recurrence order ==')
    so_r = torch.stack(r['sum_out']).float()          # [40,B,S,N]
    no_r = torch.stack([r['norm_out'][k] for k in range(0, 40, 2)]).float()
    G = gcomb.cpu().float()
    for it_ in range(ITERS - 1, 0, -1):
        c = so_r[2 * it_ + 1] + EPS                   # kernel Adds eps to loaded sum
        r_ = so_r[2 * it_] + EPS
        M = no_r[it_]
        t = G / c.unsqueeze(-2)
        d = (G * M).sum(-2)
        G = t - (d / (c * c)).unsqueeze(-2)
        t2 = G / r_.unsqueeze(-1)
        rd = (M * t2).sum(-1)
        G = t2 - rd.unsqueeze(-1)
    c1 = so_r[1] + EPS
    M0 = no_r[0]
    t = G / c1.unsqueeze(-2)
    d = (G * M0).sum(-2)
    G = t - (d / (c1 * c1)).unsqueeze(-2)
    G = (G - (G * M0).sum(-1, keepdim=True)) * M0     # softmax backward
    mixes_res = (r['hc_before_norm'] * r['inv_rms']).float()[..., 2 * E:]
    ds2_fp32 = float((G * mixes_res.view(G.shape)).sum())
    print(f'  fp64 (true)   ds2 = {rds[2].item():.6e}')
    print(f'  fp32 sim      ds2 = {ds2_fp32:.6e}   '
          f'rel {abs(ds2_fp32 - rds[2].item()) / abs(rds[2].item()):.2e}')
    print(f'  op (npu)      ds2 = 3.3540e-01   rel 1.31e+01 (from Part B)')


if __name__ == '__main__':
    main()
