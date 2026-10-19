# mhc_sinkhorn 云上 ~1500 步 loss 崩溃：离线定位（2026-10-15）

## 现象与背景

云上 `ailab_slm_mhc`（`use-fused-mhc: true`，即 deepseek4 主线实现，前向/反向走
CANN 融合算子 `aclnnMhcPreSinkhorn` / `aclnnMhcPost`）长跑在 ~1500 步附近 loss
**直接跳变崩溃**（非缓漂；实测崩溃点 1088，见下节）。同数据同环境：无 MHC 的
模型不崩；2.1.0 手写 mhc_lite 不崩（其 res 头不经过该算子）。

## 结论（TL;DR）

**`aclnnMhcPreSinkhornBackward` 的反向梯度在 res 头 logits 幅值 |z_res| ≳ 5 起
系统性错误，|z_res| ≳ 20 梯度符号翻转、幅度错 10~74×；前向在同样输入下始终
正确（y/comb rel ~1e-3 量级，bf16 地板）。** 同一数学的 eager fp32 实现
（NPU 与 CPU）在全部测试点保持 1e-6~1e-4 —— 缺陷在算子反向，不是
sigmoid/sinkhorn 参数化的固有病态。

崩溃机制与观测完全吻合：hc 参数零初始化 ⇒ z_res 从 0 出发；训练中
hc_fn/hc_scale/hc_base 增长 ⇒ z_res 分布展宽；越过 ~5 后 `hc_scale[2]`
（ds2）等梯度开始错误，~20 后符号翻转 ⇒ 错误梯度正反馈推高饱和度 ⇒
前向一直正常、某一步突然崩（"直接蹦"）。dx 梯度在深饱和时 rel 达 0.9，
即整网穿过 MHC 的梯度一并被污染。

## 证据链

三份脚本，一份比一份小：

| 脚本 | 职责 | 关键结果 |
| --- | --- | --- |
| `parity_fused_bwd.py` | pre/post 两算子 fwd+bwd vs fp64 参考；幅值/退化输入扫描；soak | 常规域 op 干净（dx rel 2.9e-3 = bf16 地板）；scale≥5 出现 dscale rel ~1；post 算子全部 case 干净（含 B=2、近置换 comb、saturated post）；soak 2000 次零 NaN、逐位确定；全部 case 双调逐位可复现 |
| `parity_scale_cliff.py` | 判别"算子缺陷 vs 固有病态"：op / NPU-fp32-eager / CPU-fp32 三方 vs 同一 fp64 参考 | fp32 两臂在全部 case（含 Wx10_s3）保持 1e-6~1e-4；op 的 ds2 从 z~3 劣化（8e-2）、z~5 起 rel 0.2~67 且 bias 同号系统偏置；dx 恶化至 5-8e-2（fp32 臂 2-3e-3）。三 seed 一致 |
| `minrepro_pre_sinkhorn_ds2.py` | 8-token 最小复现，z 幅值唯一变量 | z≤4 一致（≤3e-4）；z=6.6 起 3.4e-2；z=26 rel 2.15 且符号翻转；z=65 rel 74、dx rel 0.93。W×10 组证实触发器是 z 幅值（softmax 饱和度），与 s2/mix 谁大无关。`--save` 产出 `worst_ds2_case.pt` |
| `decompose_pre_sinkhorn_bwd.py` | 根因二分：保存量对拍 / 全真值替换 / fp32 运算序复现 | 保存量=真值（≤1e-4）；替换不改变输出；CPU fp32 按 kernel 运算序重执行复现错误 ds2 至 0.02% → fp32 精度崩溃 |

排除项：

- **post 算子（aclnnMhcPost/Backward）无嫌疑**：全 case rel 1e-7~3e-3，bias ~1e-6；
- **偶发坏值/不确定性无嫌疑**：soak 2000 次逐位确定、零非有限值；
- **退化输入无嫌疑**：x×1e-4（RMS 近 eps 区）、res_base ±50（近 one-hot）、
  x×10、B=2 全部干净；
- 已知的首调 EZ9903 缺陷家族与本案无关（那是同步 loud error，本案是静默数值错）。

## 云上 ckpt 实测闭环（2026-10-15，`ckpt_z_analysis.py`）

用已崩 run（`mtp_260923_201235__ailab_slm_mhc__`，ckpt+日志拷贝在 pfs）直接验证：

**崩溃点精确定位在 iter 1088**（非记忆中的 ~1500）：1086 步 loss 2.377 / gn 0.49
→ 1088 步 loss 4.125 / gn 58 → 1090 步 8.50 / 195 → 1092 步 11.71；此前 1062 步
刚到最低 loss 2.2846。全程 25000 步 **0 NaN、0 skipped**（有限值突变，非 NaN
保护可拦截）；崩溃后 loss 永不恢复（~7 平台），后续 grad norm 反复冲到 1e9~1e11
（最大 1.017e11 @21379）。

**参数饱和度与离线缺陷阈值严丝合缝**（zmax_est = |s2|·max‖W_res行‖ + max|b2|；
首个 ckpt 是 iter 2000，即崩后 912 步，是崩前饱和度的保守下界）：

| iter | median | max | n>5 | n>10 | n>20 |
| --- | --- | --- | --- | --- | --- |
| 2000  | 2.01 | 22.02 (L19.attn) | 14/56 | 6 | 1 |
| 6000  | 3.06 | 14.65 | 14/56 | 3 | 0 |
| 14000 | 3.26 | 14.21 | 15/56 | 3 | 0 |
| 25000 | 2.94 | 15.62 |  9/56 | 2 | 0 |

崩后饱和带持续存在——错误梯度一直在喂，与 loss 平台 + 反复 gn 尖峰互为因果。

**用 iter_2000 的真实训练参数直接对拍 op vs fp64**（同张量，S=512）——
生产参数此刻就落在毒区，剂量-响应完全成立：

| 模块 | zmax_est | ds2 rel（符号） | dx rel |
| --- | --- | --- | --- |
| L19.attn | 22.0 | **34.5×（反号）** | 0.147 |
| L17.attn | 15.6 | 1.85（反号） | 0.116 |
| L24.mlp  | 13.5 | 2.18（反号） | 0.074 |
| L16.attn | 11.4 | 1.12（反号） | 0.042 |
| L10.attn | 11.4 | 0.13 | 0.007 |
| L18.mlp  | 10.3 | 0.088 | 0.032 |
| L0.attn（对照） | 0.00 | 6.6e-5 | 3.4e-3 |
| L27.attn（对照） | 0.34 | 6.0e-5 | 2.1e-3 |

`hc_head` 为 pre-only 头变体（scale(1)/base(4)/W(4,4096)，无 res 通道），
不经 sinkhorn 反向，不受本缺陷影响。

复现：`python experiments/mhc_debug/ckpt_z_analysis.py <ckpt_root>`。

## 根因定位：算子源码级分析 + fp32 精度崩溃复现（2026-10-16，`decompose_pre_sinkhorn_bwd.py`）

**算子源码在本地 ops-transformer 克隆里**（`~/github/ops-transformer`，origin
gitcode.com/cann/ops-transformer，`mhc/` 目录与 v9.1.1 tag 一致 = CANN 9.1.1
libopapi 内置版）。调用链：mindspeed wrapper（`mindspeed_llm/ops/npu_mhc.py`）→
pip 包 autograd Function → JIT csrc 薄绑定 → libopapi 内置二进制。反向核为
`mhc/mhc_pre_sinkhorn_backward/op_kernel/mhc_pre_grad_kernel.h`（1143 行）。

**三步排除法把缺陷钉死在"fp32 精度"上**：

1. **保存量无罪**：前向 needBackward=true 暴露全部反向输入（h_pre /
   hc_before_norm / inv_rms / sumOut[40] / normOut[40]）。fp64 参考逐槽对拍：
   极端饱和 case 下全部一致（hcn 1.1e-4、h_pre 3e-6、softmax+eps 7.8e-6、
   39 个 sum 与全部行归一化矩阵 ≤6.6e-7）。解码出保存布局：norm_out 偶数槽 =
   行归一化后矩阵，奇数槽（列步矩阵）为零占位、反向从不读取；sum_out 奇数槽 =
   列分母、偶数槽 = 行分母，槽 1 保存时不带 eps（kernel 侧补加）。
2. **替换实验无罪**：把全真值保存量喂给独立反向算子，输出与训练路径逐位相同
   （ds2 0.33533 vs 0.33540，rel 均 13.1）——毒不在前向保存路径。
3. **fp32 复现**：按 kernel 源码的精确运算序（对载入 sum 补 eps → 除 → 乘 →
   归约 → 除平方 → 减，39 个半步）在 CPU 上以 fp32 重执行，得到 ds2=3.3533e-1，
   与 NPU 算子输出吻合到 0.02%；同一递推 fp64 执行得真值 2.3715e-2。
   SinkhornGrad 递推公式逐行推演数学正确（含用除后矩阵+单次除 r 的等价形式）。

**结论：算法正确、输入正确，错在算术精度。** 饱和时 softmax 输出近似 one-hot
（非主元落在 1e-6 eps 地板）、列分母可低至 ~4e-3，反向每半步"除→减"链产生
灾难性抵消，39 步级联把 fp32 的 7 位有效数字吃光（错误 ~13×，可反号）；
低饱和时矩阵近均匀、链良态，fp32 干净（1.7e-4）——与全部剂量-响应观测吻合。
前向只做除法无相减抵消，故始终正确。修复方向：SinkhornGrad 递推内用 fp64
（或补偿求和/数学重排的稳定形式）；我方规避：`use-fused-mhc: false` 或
mhc_lite 最终栈（均不经过该算子反向）。**可开工的算子修复方案（含行号落点、
float-float 补偿算术设计、验证清单）见 `FIX_PLAN.md`。**

**环境陷阱（本机构实锤）**：`torch.einsum('mc,bsk->bsm', ...)` 在此
torch_npu 2.7.1 构建上 CPU fp64 也给出完全错误的结果（同缩并换成标签
`'mk,bsk->bsm'` 即精确；maxdiff 可达 1e3）。本目录所有参考实现一律用
`F.linear`/`matmul`，勿用 einsum。

## 复现

```bash
# 完整对拍（3 seed、含 post、soak 2000，约 8 min）
python experiments/mhc_debug/parity_fused_bwd.py --seeds 3 --soak 2000 --save-failing
# 悬崖判别（三方对照，约 6 min）
python experiments/mhc_debug/parity_scale_cliff.py --seeds 3
# 最小复现 + 工单附件张量
python experiments/mhc_debug/minrepro_pre_sinkhorn_ds2.py --save
```

工单要点：`aclnnMhcPreSinkhornBackward`（CANN 9.1.1 libopapi 内置版）在
res 头 softmax 饱和（|z|≳5）时 backward 输出错误（符号翻转/幅度错），forward
正确；根因为 Sinkhorn 反向递推的 fp32 算术在饱和矩阵上灾难性抵消（见上节，
CPU fp32 复现脚本可直接附件）；8-token 复现张量在 `worst_ds2_case.pt`，预期值
由脚本内 fp64 参考给出。

## 云上动作建议

1. **直接验证**（用已崩 run 的现成 ckpt，零训练成本）：**已完成，见上节**——
   真实参数毒区对拍剂量-响应成立，闭环定案；
2. **过渡方案**：`use-fused-mhc: false` 走 eager fp32 回退（本仓验证干净；
   mhc 模块级慢 ~5×，端到端约 +11%/iter）；或切 mhc_lite 最终栈
   （res 头为单 softmax+置换凸组合，不经该算子，见 `../mhc_lite/README.md`）；
3. 向 CANN 提缺陷工单（附 `worst_ds2_case.pt` 与三份脚本）。

## 顺带发现

- `aclnnMhcPreSinkhornBackward` tiling 要求 hidden 维能被 128 整除：
  H=64 直接 `EZ1008 Failed to execute tiling function`（"c must be > 0,
  < 100000 and divisible by 128"）——单测该算子时 hidden 至少 128。
