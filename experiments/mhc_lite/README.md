# mhc_lite 实验（26.1.0）

在 26.1.0 主线 MHC（deepseek4/mhc + CANN fused 算子）之外，隔离实现 2.1.0 版
mhc_lite（V1 语义：带学习 gamma 的 RMSNorm + h_pre/h_post/h_res 三头），
目标是最大程度复用 CANN 算子能力，并与主线 full MHC 对比端到端训练性能。

## 隔离方式

不改动任何主线文件，通过新增文件接入：

| 文件 | 作用 |
| --- | --- |
| `mindspeed_llm/tasks/models/transformer/mhc_lite.py` | MHCLite 模块（系数参数化 + pre/post/head 协议） |
| `mindspeed_llm/tasks/models/spec/gpt_mhc_lite_spec.py` | layer spec（attn_mhc/mlp_mhc 指向 MHCLite，其余同主线） |
| `mindspeed_llm/core/models/gpt/gpt_model_mhc_lite.py` | GPTModelMHCLite（仅替换 hc_head_spec） |
| `pretrain_gpt_mhc_lite.py` | 入口脚本（enable_mhc 时切换 model 类） |
| `wisemlops/configs/ailab_slm_mhclite_0_5b_pretrain.yaml` | 训练配置（`entry:` 指向 lite 入口） |
| `wisemlops/jobs/webstudio_pretrain_ailab_slm_mhclite_0_5b.sh` | 30-iter 冒烟 job |

## 算法（V1）

- RMSNorm(x·e 展开) 带学习 gamma，`npu_rms_norm` 实现；
- 单个 GEMM（`LinearNoTP`，无 TP 切分）输出 `[pre 4 | post 4 | res 24]` logits；
- `h_pre = sigmoid(s·logit + b)`，`h_post = 2·sigmoid(...)`，
  `h_res = softmax(...) @ 24 个置换矩阵`（凸组合、Birkhoff 多面体、零迭代）；
- 初始化：权重全零，`b_pre` 仅本层流为 +8，`b_res` 仅恒等置换为 0，
  因此初始 h_pre 为 one-hot、h_res ≈ 0.9923·I、h_post ≈ 1；
- hc_post 侧复用 CANN `mhc_post`（前向），h_post/h_res 以 fp32 送入算子
  （算子要求 DT_FLOAT，与主线 fused 路径一致）。

## CANN `mhc_post` backward 的值域缺陷（重要）

`aclnnMhcPostBackward` 对部分合法取值组合会触发内部 Transpose launch 失败
（errno 361001，异步报错；`ASCEND_LAUNCH_BLOCKING=1` 下直接段错误）：

- 与 shape 无关：`post_backward_shape_matrix.py` 遍历 b∈{1,2,4}、s∈{512,1024,4096}
  全部通过；
- 值相关且确定性：`post_backward_repro.py` 固定 seed=1 可 100% 复现，seed=2/3 通过；
  单替换任一输入（sub/streams/post/comb/g）都不复现，需多输入同时取到坏值；
- 主线 full MHC 路径同样经过该算子 backward，存在相同的潜在风险。

规避：lite 的 `_MhcPostFn` 保留算子前向，backward 改为 fp32 torch 简约
（4 个 einsum/matmul，开销远小于一次 streams 读写）。

## 验证（`parity_test.py`，单卡 NPU）

fp32 全部通过：y/h_post/h_res vs 纯 fp32 参考 ≤2.4e-07；7 组梯度（dx/dgamma/
dbase/dscale/dW_pre/dW_post/dW_res）≤7.6e-06；hc_head 2.4e-07。

bf16：前向 vs fp32 参考 ≤3.9e-03；梯度 atol 2e-2 通过（dscale 最大 6.3e-02，
标量参数对数扰动敏感所致）；mhc_post 算子前向 vs torch 公式 3.1e-03，
经 `_MhcPostFn` 的梯度 vs torch 公式 autograd：dx 1.5e-03 / dres 6.0e-03 /
dpost 6.0e-07 / dcomb 0。

## 30-iter 冒烟（4×910B4，TP/PP/CP=1，seq 4096，gb 32）

| 版本 | 稳态 ms/iter | TFLOP/s/GPU | iter30 loss | grad norm |
| --- | --- | --- | --- | --- |
| full MHC（fused CANN） | ~3680 | 42.6–42.9 | 5.859 | 0.311 |
| mhc_lite（本实现） | ~4210 | 37.1–37.7 | 5.857 | 0.314 |

lite 比主线 full MHC 慢约 14%。原因：full 的 fused pre 算子把
norm+logits+sinkhorn+三头+y 合成单个 kernel；lite 的 pre 侧是
6–8 个 kernel（npu_rms_norm / GEMM / sigmoid / softmax / perms matmul / y bmm），
且 `_MhcPostFn` 的 torch backward 在反向新增数次 streams 级读写。

## 后续（tier 计划）

- Tier 1：triton 单 kernel 计算 lite 三头（一次读入 streams，输出
  pre/post/h_res/y，含 backward），消除 pre 侧多 kernel 开销；
  （现有 `mhc_pre_only.py` 的 `hc_pre_bmm_forward` 仅覆盖 N=4 的 pre 头。）
- backlog：调研比 lite 更高效的开源 MHC 变体（TileKernels/sglang/vLLM/CANN
  上游）并择优吸收。

## 复现

```bash
# parity + autograd
python experiments/mhc_lite/parity_test.py
# mhc_post backward 稳定性矩阵 / 值域复现（会生成 post_backward_case.pt）
python experiments/mhc_lite/post_backward_shape_matrix.py
python experiments/mhc_lite/post_backward_repro.py
# 30-iter 冒烟
bash wisemlops/jobs/webstudio_pretrain_ailab_slm_mhclite_0_5b.sh
```
