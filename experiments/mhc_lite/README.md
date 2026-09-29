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

## CANN `mhc_post` backward 的首调缺陷（重要，结论已修正）

`aclnnMhcPostBackward` 在**进程内第一次调用、且 batch≥2** 时必然失败
（异步 `AclNN_Runtime_Error` EZ9903，错误串含 `InitTilingParseCtx failed`，
tiling 冷启动），第二次起全部正常；**与输入取值/seed 完全无关**。定性证据
（`post_backward_order_probe.py`，每个场景一个全新子进程）：

| 场景（S=512 B=2 除注明；bad/good = seed 1/2 的值） | 结果 |
| --- | --- |
| bad / good / seed0 单跑 | 首调 RAISE（与取值无关） |
| bad,bad / seed0,seed0 / good,good | RAISE,OK（同值第二次必过） |
| bad@4096x1 / bad@512x1（同 seed 1 值，B=1） | **OK**（B=1 首调免疫） |
| bad@512x4 | RAISE（B≥2 首调必挂） |
| bad@512x1,bad | OK,OK（B=1 首调把 B=2 也预热） |
| fwd1 / fwd35 后 bad | WARM,RAISE（**mhc_post 前向不预热**） |
| prefwd 后 bad | WARM,RAISE（**pre 仅前向也不预热**） |
| prewarm（pre fwd+bwd）后 bad | WARM,OK（**任一 backward 预热**） |

- 旧"值域缺陷（seed=1 必挂、seed 2/3 通过）"是**进程内次序伪象**：
  `post_backward_repro.py` 恒把 bad 用例排在进程第一个 backward，其
  pairwise/leave-one-out 混合"全过"只是因为都已不是首调。
- 旧"与 shape 无关"同样是伪象：`post_backward_shape_matrix.py` 恰好先跑
  (b=1,s=512)，把后面的 b=2/4 全部预热。真实规律是 **B=1 首调免疫、
  B≥2 首调必挂**。
- 单进程 seed 扫描不可信（`post_backward_seed_sweep.py`，留档）：首调异步
  报错后 context 被污染（对照用例返回 GRAD-BAD），其"0/100 挂"只是因为
  首调之后再无首调。
- 预热源是**任意一次 backward**（本算子任意 shape，或 mhc_pre_sinkhorn 的
  backward——tiling 解析上下文应为跨算子共享）；前向再多也不触发初始化，
  等待也不行（prefwd 场景进程启动 20+s 后首调仍挂）。
- **主线训练结构上安全**：B=1 免疫；mbs≥2 时反向图序保证首个 mhc_post
  backward 之前已跑过大量其他算子 backward（lm_head/attention/MLP），
  共享上下文已就绪——两次 30-iter 冒烟从未触发与此一致。真正暴露面是
  "孤立进程 + B≥2 + 首个 backward 就是 mhc_post"（单测/独立 bench），
  先跑一次任意 warm-up backward 即可规避。
- lite 的 `_MhcPostFn`（算子前向 + torch 简约反向）仍是稳妥默认；
  `MHC_LITE_NATIVE_POST_BWD=1` 的 ~20 ms/iter 收益在训练时序下可安全获取。

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
| mhc_lite Tier-0（torch 链） | ~4210 | 37.1–37.7 | 5.857 | 0.314 |
| mhc_lite Tier-1（triton） | ~4005 | 37.9–39.5 | 5.856 | 0.313 |
| mhc_lite Tier-2（triton） | ~3767 | 41.8–42.2 | 5.857 | 0.313 |
| **Tier-2 + 原生 post 反向** | **~3583** | 42.8–44.3 | 5.856 | 0.313 |

Tier-2 + 原生 post 反向（`MHC_LITE_TRITON=1 MHC_LITE_NATIVE_POST_BWD=1`）
比 full MHC 快约 2.7%——融合度拉平后，lite 免去 sinkhorn 迭代的优势开始
显现；该组合使用的 `aclnnMhcPostBackward` 缺陷为 B≥2 首调冷启动
（见上节），训练时序下结构性安全。

Tier-0 比主线慢约 14%：full 的 fused pre 算子把 norm+logits+sinkhorn+三头+y
合成单个 kernel，而 lite 的 pre 侧是 6–8 个 kernel，且 `_MhcPostFn` 的 torch
backward 在反向新增数次 streams 级读写。Tier-1 triton 融合收回约 210 ms/iter
（~5%），剩余 ~8.7% 差距主要在 pre 反向（full 为 aclnn 融合反向单 kernel，
lite 为多 kernel 链 + autograd Function 的 python 开销）。

### 原生 aclnn 反向的对照（`bench_post_native_bwd.py` / `bench_full_mhc_ref.py`）

| 环节（S=4096，同布局口径） | aclnn 原生 fwd+bwd | Tier-1 当前 |
| --- | --- | --- |
| post（`mhc_post`） | **0.581 ms** | 0.988 ms（fwd 0.129 + K3 0.86） |
| pre（`mhc_pre_sinkhorn`，仅 full MHC 语义） | 2.177 ms | 3.762 ms |

post 侧若直接用原生反向（`MHC_LITE_NATIVE_POST_BWD=1` 开关，默认关；
B≥2 首调需先 warm-up 一次，见首调缺陷一节）：每次省 ~0.4 ms × 48 调用
≈ **20 ms/iter（~0.5%）**。pre 侧原生反向与 lite 语义不同
（sinkhorn vs 单 softmax、无学习 gamma），不能直接复用；其 1.6 ms/次差距
（≈76 ms/iter）是剩余差距的大头，需更深融合或 CANN 侧 lite 算子。

## 后续（tier 计划）

- Tier 1（已实现）：triton 融合 pre/post 侧 kernel，消除多 kernel 开销；
- Tier 2（已实现）：launch/python 开销与中间量物化的消除（见下节）；
- backlog：调研比 lite 更高效的开源 MHC 变体（TileKernels/sglang/vLLM/CANN
  上游）并择优吸收；CANN 侧 lite 版融合 pre 算子（无 sinkhorn，理论快于
  full 的 mhc_pre_sinkhorn）。

## Tier 2：差距归因与消除（`MHC_LITE_TRITON=1`）

### 为什么"无迭代"却没有更快（归因，L=28 → 56 次 pre + 56 次 post/iter）

1. **迭代本身不耗时**：full 的 20 步 sinkhorn 作用在 [bs,16] 小张量上，
   在融合 kernel 内是 µs 级；两边的主体成本同为 norm+GEMM+三头+y 数据流。
   lite 的省略只占 <5% 计算量，融合度不齐平时完全被淹没。
2. **launch 分发开销（实测探针 `probe_overheads.py`）**：triton JIT launch
   **105µs/次**（torch 原生 21µs，autograd Function fwd+bwd python ~0.3ms）。
   Tier-1 的 pre fwd+bwd 有 18 个 launch（6 个 triton），full 只有 2 个
   aclnn；仅分发 ≈1.1ms/调用。
3. **中间量物化**：xn/dx_direct/d_x_rms/d_xn 等 32MB 级中间张量多次往返，
   pre fwd+bwd ≈330MB vs full 融合 ≈130MB。

### Tier-2 手段与效果（S=4096 组件级）

| 手段 | 效果 |
| --- | --- |
| `_launch`：CompiledKernel 直调缓存（需 3D grid；105→48.8µs） | 全部 kernel 分发减半 |
| K1a 并入 res 头（softmax+16 置换列和，[16,NP] **连续**转置表） | 16× stride-16 gather 放大消除，K1 全家 0.53→0.275 ms |
| megaK：K4+K2 合一（dW 简约 + 三头 jacobian，寄存器传递） | pre bwd 1.06→0.855 ms，少 1 launch |
| `lite_grad_x`：重算 w·g 取代 dx_direct 物化 | 0.73→0.171 ms，省 64MB/次 |

结果：hc_pre fwd **0.682 ms（比 full 融合算子 fwd 1.049 快 35%）**；
hc_pre fwd+bwd 3.76→**2.74 ms**（full 2.18）；数值 parity 同 Tier-1
（fp32 梯度 ≤1.1e-05）。

### post 原生反向实测（`MHC_LITE_NATIVE_POST_BWD=1`，30-iter）

`bench_post_native_bwd.py`：aclnn 原生 fwd+bwd 0.581 ms vs K3 路径
0.988 ms；**端到端实测 median 3949.6 vs 4005 ms（省 ~55ms/iter，1.4%）**，
loss 5.8566 对齐；30-iter 真实训练未触发首调缺陷（B=1 免疫 + 训练时序
预热，见首调缺陷一节）。

## Tier 1：triton 融合 kernel（`MHC_LITE_TRITON=1`）

`mindspeed_llm/ops/triton/mhc_lite_heads.py`：

| kernel | 覆盖 | 耗时（S=4096） | 替换的 torch 链 |
| --- | --- | --- | --- |
| K1a `lite_heads_fwd_kernel` | pre/post 头 sigmoid → h_pre/h_post | ~0.10 ms | split+sigmoid 链 |
| K1b `lite_y_fwd_kernel` | y = Σ h_preᵢ·xᵢ（从 h_pre 显存重读） | ~0.13 ms | y bmm 0.59 ms |
| res 头（torch） | softmax + 24 置换凸组合 | 0.15 ms | 同左 |
| K2 `lite_heads_bwd_kernel`(+reduce) | 三头 jacobian + dlogits/dscale/dbase | 0.46 ms | sigmoid/softmax 反向链 |
| K4 `lite_y_bwd_kernel` | dh_pre 简约 + d_x_direct 瓦片 | ~0.25 ms | y bmm 反向 0.49 ms |
| K5 `lite_add_cast_kernel` | d_x_direct + d_x_rms 融合加法 | ~0.25 ms | 多次 fp32 cast+add 0.48 ms |
| K3 `lite_post_bwd_kernel` | mhc_post 反向四输出全融合 | 0.86 ms | 4×einsum fp32 链 |

组件级（S=4096,B=1,bf16）：hc_post fwd+bwd 2.275→**1.19 ms**；hc_pre fwd
1.02→**~1.0 ms**（与 full fused 算子 fwd 1.049 ms 持平）；hc_pre fwd+bwd
3.78→**3.76 ms**（反向剩余 ~2.7 ms 主要是 autograd/python 与 rms/GEMM 反向）。

### ascend triton 后端的两处缺陷（bisect 证据：`bench_k1_bisect.py`）

1. `tl.dot` 数值错误（所有 dtype，误差 ~1e+02）→ 全部改用 2D
   load/elementwise/`tl.sum` 惯用法（同 `mhc_pre_only.py`）；
2. **"存小向量 + 同值喂宽瓦片"悬崖**：kernel 内把 [GROUP] 维向量（如 h_pre）
   存入显存、且该值同时参与 [GROUP,BLOCK_D] 瓦片计算时，GROUP≥2 的 kernel
   从 ~0.13 ms 跌到 ~10.2 ms（≈75×），与 sigmoid/存储 dtype 无关
   （V2 0.133 / V5 0.136 / V6 10.2 / V4-线性 10.2）；GROUP=1 可逃离。
   规避：拆双核——K1a 只存不算瓦片、K1b 从显存重读 h_pre 算 y。

数值验证（`triton_parity_test.py`，`MHC_LITE_TRITON=1`）：fp32 前向 ≤1.2e-07、
7 组梯度 ≤1.1e-05；bf16 前向 ≤2.8e-03、梯度与 Tier-0 torch 路径同量级
（dscale 6.3e-02 同为标量扰动敏感）；post triton 反向 vs torch 公式
dcomb 5.5e-07。

## 社区 910B MHC 实现调研（2026-09）

| 实现 | 形态 | 反向 | 对训练的参考价值 |
| --- | --- | --- | --- |
| vllm-ascend `csrc/moe/hc_pre`/`hc_post` | Ascend C 融合算子（rms+GEMM+sinkhorn 20 迭代单 kernel，910B tiling，HF32 matmul，2026-09 仍活跃） | 无（fwd-only，aclnn 注册） | 若做 CANN 侧 lite 融合 pre 算子，其 Ascend C 写法是最直接模板；PyTorch 回退实现可做对拍 |
| CANN ops-transformer `experimental/mhc`（智子芯元 KernelCAT 生成） | Ascend C 三算子（pre/post/res），sinkhorn 归上层 | 无 | 仅 tiling/精度写法；官方 `mhc/mhc_sinkhorn` aclnn 版仅支持 950PR/DT，910B 不可用 |
| sgl-kernel-npu | 无实现（CI 中仅有 950 实验 `hc_post` 构建项） | — | 无 |
| tilelang-ascend `examples/mhc_post` | TileLang，910B 实测 bf16 4.63→0.38ms（纯 Vector 路径，vs CANN 基线 5.98×） | 无（mhc_pre/mhc_bwd 示例 2026-09 曾合入即回退） | mhc_post 的 Vector 优化路径值得对照；backward 不可用 |
| deepseek TileKernels `tile_kernels/mhc` | TileLang（**仅 SM90/SM100 GPU**） | 有（sinkhorn 自定义 bwd + autograd 感知训练路径） | 反向语义/训练路径的最好范本（GPU-only）；lite 无迭代，仅 full MHC 适用 |
| yixuan/mHC-proj（arXiv:2606.07574） | CUDA warp 级 | 有（Newton 代 sinkhorn + 隐函数微分免存中间量） | full MHC 反向设计思路参考（GPU-only） |
| FFTYYY/mhc-lite | 纯 PyTorch 训练实验（同为"去 sinkhorn 迭代"思路） | 有（autograd） | 印证 lite 方向；无 910B kernel |

结论：910B 上带反向的训练级 MHC/lite 实现**开源社区为零**；本仓的 aclnnMhcPreSinkhorn
（fwd+bwd）+ Tier-2 triton 组合已是唯一可用训练路径。可吸收的外部经验只有写法：
vllm-ascend 的 Ascend C 融合 pre 模板（上游 lite 算子诉求）、tilelang-ascend 的
mhc_post Vector 路径。

## torch.compile 在 NPU 上的机制与 naive 实测（`bench_compile.py`）

三条可用编译路径（torch 2.10 + torch_npu 2.10.post6）：

- `aot_eager`：dynamo 捕 FX 图，仍用原 aten 算子执行，只省 python 分发；
- `inductor`：`torch_npu._inductor` 将 codegen 打到 triton ascend 后端
  （`npu_triton_heuristics.py`），pointwise/reduction 生成 triton kernel、GEMM 走 extern；
- `torchair`（`torch_npu.dynamo.torchair.get_npu_backend(compiler_config=...)`）：
  FX→分解→CANN GE 整图捕获，图级融合 + 单图下发，前向/反向均可成图。

组件级实测（S=4096, B=1, bf16，Tier-0 torch 链复刻为纯函数，`fwd / fwd+bwd` ms）：

| 链 | eager | aot_eager | inductor | torchair |
| --- | --- | --- | --- | --- |
| post 公式 | 0.585 / 1.372 | 0.593 / 1.435 | 1.142 / 2.911（慢 ~2×） | **0.462 / 1.096（−21%/−20%）** |
| pre 全链 | 0.853 / 2.911 | 1.331 / 4.149（慢 ~56%） | 编译失败 | 编译失败 |

- inductor pre 失败：npu codegen 对 0-dim/单元素张量的 broadcast 生成
  `tl.broadcast_to` rank mismatch（`NoTritonConfigsError`），把 scale/base 切片外提也绕不开；
- torchair pre 失败：GE `matmul_backward currently only support mask == [True, True]`；
- inductor post 生成的 triton kernel 全面慢于 CANN 原生算子（与已知 triton-ascend
  launch 105µs、codegen 质量问题一致）；
- aot_eager 反而更慢：guard + boxed-arg 包装开销 > 该规模链的 python 分发收益。

**naive 结论**：全链 compile 无增益。唯一可用点是 torchair 编译 post 公式：
单次省 ~0.28ms × 56 调用 ≈ **15 ms/iter（~0.4%）**，语义为 torch 公式（不依赖
`aclnnMhcPostBackward`），且比 Tier-2 的 aclnn 前向 + K3 反向（b02 1.211ms）
还快 ~10%——可作 Tier-3 候选。

## 参考实现精读与借鉴实验（`bench_borrowed.py`，2026-09）

精读三个参考仓后的可借鉴点与实测（S=4096, B=1, bf16，基线 = ms-llm 原有
mhc_sinkhorn fused 路径）：

**vllm-ascend `hc_pre`/`hc_post`（Ascend C，fwd-only）**
- hc_pre 为 AIC/AIV 协同单 kernel 两段式：AIV 把 x cast fp32 写 GM workspace、
  AIC 做 cube GEMM，Part2 在 vector 上完成 K-split 归约 + RMS（squareSum 走
  `GatherMaskByDiagonal` 取对角）+ 三头 + y（cast→Brcb 广播乘→ReduceSum，无矩阵
  引擎）+ **sinkhorn 20 迭代全在 kernel 内**（每步 row/col sum 后 `+eps` 再除）；
- hc_post 用 **MicroAPI 寄存器级**实现：每 token 的 post/comb 标量以
  `DIST_BRC_B16` 广播进寄存器，x/residual 单遍 unpack，fp32 `MulAddDst` FMA
  累加——post 是访存受限算子，最优形态 = 单遍向量 FMA，不走 cube；
- cube GEMM 在 fp32 上做（x 先 cast），精度优先。

**KernelCAT `ops-transformer/experimental/mhc`（Ascend C，fwd-only）**：mHC
（流形约束、**逐层静态权重**）三算子分解 pre/post/res，UB 动态 tiling
（192KB 预算 + buffer 数算术），bf16 走 cast-fp32-计算-cast 回。分解思路与
lite 的三头结构同源，但其"权重静态"设定不含我们的动态 per-token 场景。

**tilelang-ascend `examples/mhc_post`（910B, CANN 9.0, V0→V10 记录）**：
- V0 cube 双 kernel 4.63ms → V10 纯 Vector 单 kernel 0.38ms（n=4096,h=2560，
  vs CANN 基线 5.98×）。关键教训：hc=4 的 [4,4]@[4,h] 用 cube 需 pad 到 16、
  浪费 93.75% MAC；**AXPY（`T.tile.axpy` 标量乘加）替代 broadcast+mul+reduce_sum
  是最大单项突破**（消除 7 个 2D fp32 UB buffer，UB 省 3/4 使 h_blk 提到 2048）；
- 其余：双 V 核按 token 划分、循环不变量（comb）外提、2D merged copy/store、
  自适应 h_blk 取 h 的最大因数消除 padding、`T.Pipelined` 双缓冲、kernel 缓存。

### 借鉴实验结果

| # | 对照（同 shape 同 dtype） | 结果 | 结论 |
| --- | --- | --- | --- |
| A1 | 主线 torch 回退 post（broadcast+sum 形式，`mhc.py:286`） | 1.529 ms | 正是 tilelang V1 判定慢的形式，还物化 [s,b,4,d] fp32 中间量 |
| A2 | bmm 形式（Tier-0 同构） | 0.573 ms | cube bmm 仍优于 eager broadcast 链 |
| A3 | AXPY 形式（`addcmul_` 累加，tilelang V4 思想的 eager 近似） | 1.469 ms | **AXPY 收益只在 kernel 内**：eager 下 17 次 launch 抵消算法优势 |
| A4 | aclnn `mhc_post` 裸调（op 原生布局） | **0.114 ms** | CANN 算子已是 MicroAPI 向量路径，比最好的 eager 形式快 5× |
| A5 | 经 `npu_mhc` wrapper（[s,b]↔[b,s] 布局适配） | 0.179 ms | wrapper 的输出 `clone()` 代价 **0.065 ms/次 ≈ 3.6 ms/iter**（B=1；B>1 时输入也要拷贝，代价更大）——主线可优化点 |
| B1 | lite res 头 triton kernel（softmax+置换表） | 0.114 ms | 与 torch 参考（0.151ms）对齐 |
| B2 | full sinkhorn-20 头 triton kernel（行/列迭代全在 kernel 内，vllm-ascend 借鉴） | **0.228 ms** | 迭代使头部成本 ×2；eager sinkhorn 循环要 3.294ms（kernel 内快 14×）。**lite 免迭代在等融合度下省 ~0.11 ms/次 ≈ 6 ms/iter（0.17%）**——理论优势存在但幅度小，印证 Tier-2 的归因 |
| C1/C2 | logits GEMM bf16 vs fp32（含 cast） | 0.035 / 0.087 ms | fp32 GEMM 对数 logits 误差 1.6e-02→0（vs fp64）；+0.05ms/次（×56≈3ms/iter）可换精度，是 lite 可选的精度开关（vllm-ascend 选 fp32 cube） |

综合：社区参考在 torch 层可吸收的只有"**别用 broadcast 链**"（主线回退路径
1.53→0.57ms，若走非 fused 回退值得改）；性能上限仍取决于算子级融合
（aclnn/AIC-AIV 协同/MicroAPI），这正是 Tier-2 triton 与上游 lite 融合算子
诉求的方向。wrapper 布局适配的 3.6ms/iter 与 fp32 GEMM 精度开关是两个
立即可落地的主线优化点。

### tilelang 直测（`bench_tilelang_post.py`，源码构建后补齐）

tilelang-ascend 在本机只能源码构建：pypi 0.1.4 wheel 需 glibc 2.38
（本机 2.34），GitHub release wheel 被网络代理截断（20MiB 上限）。
源码构建（`github/tilelang-ascend` @ascendc_pto，`USE_ASCEND=true python
setup.py build_ext --inplace`）产物只依赖 GLIBC_2.34/GLIBCXX_3.4.26，
经 `PYTHONPATH=<clone> LD_LIBRARY_PATH=<conda>/lib` 直接可用，无需安装。

三方对照（n=4096, hc=4, bf16，本机 CANN 9.1.1）：

| shape | tilelang V10 | eager bmm | aclnn `mhc_post` 裸调 |
| --- | --- | --- | --- |
| h=1024（本仓 0.5B shape） | 1.086 ms | 1.207 ms | **0.114 ms** |
| h=2560（tilelang 仓文档 shape） | 0.801 ms | 3.041 ms | **0.168 ms** |

tilelang 仓自称的 "vs CANN 5.98×" 基线是 eager bmm；**与 aclnn 融合算子
对比它是慢 7~10× 的**——aclnn 的 MicroAPI 单遍向量路径在 post 这种访存
受限算子上仍是上限，tilelang 中间层换不来超越。

### sinkhorn（full MHC res 头）各实现对比（`bench_sinkhorn.py`）

同口径对比：logits [4096,16] fp32 → softmax(s·l+b)+eps → 初始列归一 →
19×(行归一,列归一) → out [4096,16]，即 `torch_hc_split_sinkhorn` res 段：

| 实现 | ms | maxdiff vs fp64 |
| --- | --- | --- |
| eager torch 循环（主线回退形态） | 3.017 | 0 |
| aot_eager | 2.991 | 0 |
| inductor（npu triton codegen） | 2.987 | 0 |
| torchair（CANN GE 图） | 2.994 | 0 |
| tilelang UB kernel（源码构建，双 V 核） | 0.735 | 1.2e-07 |
| **triton 融合 kernel**（16 值驻留寄存器） | **0.227** | 1.2e-07 |
| （参照）aclnnMhcPreSinkhorn 整算子 fwd | 0.633 | 含 norm+GEMM+3 头+sinkhorn+y |

结论：
- **sinkhorn 单算最快是 triton 融合 kernel（0.227ms，eager 的 13×）**：
  每 token 的 16 个值全程驻留寄存器，38 次归一化全是寄存器内逐元素运算，
  显存只碰一次读一次写；
- **torch.compile 三后端对这个迭代小张量全部无效**（都 ~3ms）：dynamo/
  inductor/GE 图都无法把 19 次 `sum+div` 循环融合成单 kernel，每次迭代
  仍是独立小 kernel；
- tilelang 版 0.735ms：tile 原语每步是独立向量指令（约 580 条 [256,8]
  tile op），无寄存器级融合，且 fp32 行宽须 pad 到 8 通道（32B 对齐），
  一半算力花在 pad 通道上——比 triton 慢 3.2×。tilelang 的甜区是大 tile
  数据流（如 mhc_post），不在这种 16 宽迭代小矩阵；
- aclnn 融合算子里 sinkhorn 被摊销（整算子含全部 pre 计算才 0.63ms），
  印证"迭代进 kernel"是正解（vllm-ascend 同路线）。

### 完整 mhc_sinkhorn 端到端对比（`bench_sinkhorn_e2e.py`）

实验对象是**完整语义的 mhc_sinkhorn 模块**（deepseek4/mhc.py：pre 全链
fp32 RMS→logits GEMM→三头→20 迭代 sinkhorn→y，加 post 全链
`out = h_post⊗y + h_resᵀ@residual`），fwd 与 fwd+bwd 双口径，S=4096/B=1/
hc=4/h=1024、bf16 输入 fp32 参数，不含 mhc_lite 维度。

**pre 侧**（fwd / fwd+bwd ms，数值 vs 回退参考）：

| 实现 | fwd | fwd+bwd | y/post/comb maxdiff |
| --- | --- | --- | --- |
| aclnn fused（主线 fused 路径，经 wrapper） | **0.638** | **2.145** | 1.6e-02/5.0e-06/4.5e-06 |
| torch 回退（主线非 fused 分支，eager 循环） | 3.772 | 14.799 | 0 |
| 回退 + triton sinkhorn 头（自写 fwd/bwd kernel） | 1.262 | 5.151 | 0/0/8.9e-08 |

**post 侧**：

| 实现 | fwd | fwd+bwd | out maxdiff |
| --- | --- | --- | --- |
| aclnn fused（主线 wrapper，含输出 clone） | **0.185** | **0.840** | 3.1e-02 |
| torch broadcast（主线回退形式） | 1.590 | 5.983 | 0 |
| torch bmm | 0.575 | 1.362 | 3.1e-02 |
| torchair bmm（整图编译） | 0.570 | 1.363 | 3.1e-02 |

**端到端链**（pre+post，residual = 输入流）：

| 链 | fwd | fwd+bwd | out maxdiff |
| --- | --- | --- | --- |
| 主线 fused（aclnn pre + aclnn post） | **0.909** | **2.918** | 3.1e-02 |
| 主线回退（eager pre + broadcast post） | 4.349 | 15.151 | 0 |
| triton 头 pre + bmm post（全程不碰 aclnn） | 1.758 | 6.635 | 3.1e-02 |
| fused pre + bmm post | 1.281 | 3.699 | 3.1e-02 |
| fused pre + torchair post | 1.287 | 3.707 | 3.1e-02 |

结论：

- **主线 fused 路径在完整 mhc_sinkhorn 维度上没有对手**：端到端 fwd+bwd
  2.92 ms，第二名（fused pre + torch/bmm post）3.70 ms 还慢 27%——post 侧
  aclnn 原生反向（0.84 ms 含 wrapper 开销）比最好的 torch 形式（bmm 1.36）
  快 40%。若要改用 torch 反向而保留 fused pre，
  代价是 +0.78 ms/次（×48 ≈ 37 ms/iter，~1%）。
- **主线回退路径慢 5.2×**（15.15 vs 2.92）：大头是 eager sinkhorn 循环
  （pre fwd 3.77 ms 里 ~3 ms）与 broadcast post（fwd+bwd 5.98 ms）。
  非融合场景下至少应把 post 换成 bmm 形式（−4.6 ms）。
- **自研 triton sinkhorn 头（含反向）是 aclnn 之外唯一带正确反向的融合
  实现**：把回退 pre fwd+bwd 从 14.80 拉到 5.15 ms（2.9×），梯度与 eager
  autograd 对齐（dlogits 3.3e-09，d 3.1e-02 的 e2e 差全部来自 post 侧
  bf16 收缩，与 aclnn 链同量级）。纯 torch+triton 组合 6.64 ms 可作
  "不依赖缺陷算子"的安全回退，训练仍应走 fused。
- wrapper 布局适配代价在本形状（B=1）：pre 侧可忽略（fwd 0.638 vs 裸
  0.633）；post 侧输出 clone +0.07 ms/次（0.185 vs 裸 0.114，×48 ≈
  3.4 ms/iter），是主线的直接可优化点。
- torchair 对本节 post 形式（含 `comb.transpose(-1,-2)`）无增益
  （1.363 vs bmm 1.362）——早前 `bench_compile.py` 里 torchair post
  −20% 的口径无转置；带转置后 GE 图不再占优。

triton 头反向的实现：前向 kernel 把 40 个归一化阶段的 m 值 checkpoint 到
GM（[40, bs, 16] fp32，10.5 MB），反向 kernel 用运行时循环按
`(g − ⟨g, m_out⟩_axis)/(S+eps)` 逐阶段回放（行归一步 dot 按行、列归一步
按列，softmax 头标准 vjp）。两个 triton-ascend 工程坑记录：

1. **完全展开的大 kernel 会让 ascend backend 编译病态**：40 阶段 × 160
   条 store 的展开版 ttir 217 KB，backend 编译 >10 min 不出 npubin；同样的
   循环体改成运行时 `for`（scf.for）后 IR 体量恒定，7 s 编完，数值不变。
2. **triton 不校验输入连续性**：`mixes[..., 8:].reshape(bs, 16)` 挤掉
   size-1 维时返回行 stride 24 的视图，kernel 按连续布局静默读错（token 0
   对、其余全错）；喂 kernel 前必须显式 `.contiguous()`。

## 复现

```bash
# parity + autograd（Tier-0 torch 路径）
python experiments/mhc_lite/parity_test.py
# Tier-1 triton 路径 parity
python experiments/mhc_lite/triton_parity_test.py
# kernel 计时 / K1 悬崖 bisect / 反向分解 / full 算子参照
python experiments/mhc_lite/bench_kernels.py
python experiments/mhc_lite/bench_k1_bisect.py
python experiments/mhc_lite/bench_pre_bwd_pieces.py
python experiments/mhc_lite/bench_full_mhc_ref.py
# torch.compile 三后端 naive 实测
python experiments/mhc_lite/bench_compile.py
# tilelang mhc_post 三方对照（需源码构建 tilelang-ascend，见上节）
PYTHONPATH=<tilelang-ascend clone> LD_LIBRARY_PATH=<conda env>/lib:$LD_LIBRARY_PATH \
  python experiments/mhc_lite/bench_tilelang_post.py
# sinkhorn 各实现对比（tilelang 段同样需要源码构建，缺失时自动跳过）
PYTHONPATH=<tilelang-ascend clone> LD_LIBRARY_PATH=<conda env>/lib:$LD_LIBRARY_PATH \
  python experiments/mhc_lite/bench_sinkhorn.py
# 完整 mhc_sinkhorn 端到端对比（pre/post/链路，fwd 与 fwd+bwd）
python experiments/mhc_lite/bench_sinkhorn_e2e.py
# mhc_post backward 稳定性：shape 矩阵 / 值混合（会生成 post_backward_case.pt）
python experiments/mhc_lite/post_backward_shape_matrix.py
python experiments/mhc_lite/post_backward_repro.py
# 首调次序定性（每个 --order 一个全新子进程，是首调缺陷一节的证据）
for o in bad good,bad bad,bad bad@4096x1 bad@512x1,bad fwd35,bad prefwd,bad prewarm,bad; do
  python experiments/mhc_lite/post_backward_order_probe.py --order $o
done
# 单进程 seed 扫描（方法论留档：首调报错后 context 被污染，结果不可信）
python experiments/mhc_lite/post_backward_seed_sweep.py
# 30-iter 冒烟（Tier-0 / Tier-1=triton）
bash wisemlops/jobs/webstudio_pretrain_ailab_slm_mhclite_0_5b.sh
MHC_LITE_TRITON=1 bash wisemlops/jobs/webstudio_pretrain_ailab_slm_mhclite_0_5b.sh
```
