# aclnnMhcPreSinkhornBackward 精度缺陷修复方案（2026-10-19）

配套文档：`README.md`（缺陷定位与证据链）、`DIAGNOSTIC_REPORT.md`（详细报告）。
本文给出算子源码级的改动方案，目标是可以直接开工的工程方案，不是方向性讨论。

## 0. 问题回顾（一段话说清）

反向核 `mhc/mhc_pre_sinkhorn_backward/op_kernel/mhc_pre_grad_kernel.h` 的
`SinkhornGrad()`（第 373–559 行）按前向相反顺序回放 39 个"半步"（19 轮迭代
× 行/列归一化各一次 + 收尾半步 + softmax 反向）。每个半步的数学形式是
"梯度除以归一化分母，再减去它在当前矩阵方向上的投影"——**减法两边的数值
在 softmax 饱和时几乎相等，fp32 的 7 位有效数字在相减中损失殆尽**；且分母
可低至 ~4e-3，每半步把绝对误差放大 ~250 倍，39 步级联后梯度错 10~74 倍、
可反号。前向只做除法无此相减，故前向始终正确。CPU 上用 fp64 按同一运算序
重执行即得真值——**数学正确，纯算术精度问题**。

## 1. 硬件约束（决定方案选型的实测事实）

- 910B 的向量核只暴露 fp16/bf16/fp32/int 指令。实测 CANN 9.1.1 全部
  Ascend C 头文件（`/usr/local/Ascend/cann-9.1.1/aarch64-linux/ascendc/`）
  中 **没有任何 double 类型/指令**（全文检索 0 处）→"整段改 fp64"这条最短
  路在当前硬件 + CANN 版本上不可行。
- 现有 fp32 向量指令（Add/Sub/Mul/Div/Brcb/BlockReduceSum/DataCopyPad）
  齐全，且该段的张量很小（[tileCoreBS×16, 16]），指令余量充足。

## 2. 推荐方案：float-float 补偿算术（双 float 表示）

**思路**：递推链的每个中间张量从"一个 fp32"改为"一对 fp32（hi, lo）"，
hi+lo 合计约 48 位尾数有效数字（fp64 是 53 位，同一运算序下足够：decompose
实验里 fp32 底噪 1e-7 被链放大 ~10⁵ 变成 1300% 误差；48 位表示的底噪
1e-14~1e-15，同样放大后仍在 1e-9 量级，余量 7 个数量级）。

所有运算只用现有 fp32 指令实现，对 hi/lo 两个分量各做一遍：

| 基础算子 | 实现要点 | 向量指令数 |
|---|---|---|
| two_sum（Knuth） | s=a+b; t=s−a; e=(a−(s−t))+(b−t); 结果 (s, e+a_lo+b_lo) | ~7 |
| two_product（Veltkamp 拆分 + Dekker） | 用常数 4097（2¹²+1）把操作数拆成 12+12 位两半，先乘后补残差 | ~9 |
| two_div | q_hi=Div(a_hi,b)（现有 Div）；用 two_product+two_sum 求 a−q_hi·b 的残差；q_lo=Div(r,b) | ~12 |
| 行/列归约（Σ 16 项） | Neumaier 补偿累加；文件里现成的 `kahanCustom`（第 42 行）就是雏形，目前只用于 scale/bias 归约，不在 Sinkhorn 链上 | ~2×朴素 |

布局类指令（Brcb 广播、BlockReduceSum、DataCopyPad、跨步 repeat 参数）
**原样保留**——float 布局一个字节都不动，只是对 hi/lo 各执行一次。

### 2.1 改动落点（真实行号）

| 位置 | 现状 | 改动 |
|---|---|---|
| `SinkhornGrad()` L373–559 | 单 float 递推 | hi/lo 成对递推 |
| L391–480 主循环（38 个半步） | 每半步 ~20 条指令 | 换成上表基础算子组合；两次相减点 L445–451（列步）、L474（行步）是抵消核心，必须走补偿路径 |
| L415/L440/L460 三个 Div | 除以分母（放大器） | two_div |
| L481–545 收尾半步 + softmax 反向 L532–545 | 同样有"减投影"（L543） | 一并纳入补偿区（ds2 的最后一段相减就在这里） |
| 载入的 saved norm/sum（L397–405） | fp32 | 升位为 (x, 0) 对，视为精确值（其自身误差 ≤1e-4，已排除嫌疑） |
| 链尾输出（L545 之后） | fp32 | hi+lo 合并回单 fp32，后续 `ComputeGradHHat2` 等段落不动 |
| `InitGradPreStageBuffer()` L252–330 | 手工 offset 账本 | 递推段缓冲 ×2（gradHRes 对 + 两个 temp 对）；若 UB 顶满，下调 tileCoreBS（host tiling 同步改），只影响性能不影响正确性 |

### 2.2 性能预期与门槛

该段作用在 [tileCoreBS×16, 16] 小张量上，每半步 ~20 条 → ~150–200 条向量
指令，39 步共 ~8k 条，仍是微秒级；本算子的耗时主体是两个 GEMM（mm1/mm2）。
**门槛：算子级（pre fwd+bwd）回归 ≤15%，端到端 iter 回归 ≤1%。**

## 3. 备选方案（及不选的原因）

| 方案 | 结论 |
|---|---|
| 整段 fp64 | 910B Ascend C 无 double 向量指令（§1 实测），不可行；若未来 CANN 版本开放可整体替换本方案且更简单 |
| 数学重排消除相减 | 归一化的 Jacobian 向量积本质是"减去投影"，残差本身就是输出量，相减无法绕开，只能加有效数字 |
| 标量（SIMT）路径用 C 的 double | arch35 目录有 `simt_vf.h` 提示标量实现风格，标量语义下 double 可编译（软件仿真）；但整段重写且慢一个量级，仅作兜底 |
| 我方规避（不改算子） | `use-fused-mhc: false` eager 回退 / mhc_lite 最终栈（已验证无此缺陷）——上游修复前的主线方案，与本方案互补，见 `../mhc_lite/README.md` |

## 4. 构建与验证（全部现成资产）

构建：改 `~/github/ops-transformer`（gitcode.com/cann/ops-transformer 本地
克隆）→ 用我们现有的 build_out 构建流程（lite_pre 三算子已跑通）→
`ASCEND_CUSTOM_OPP_PATH` 指向 build_out 即可生效（优先于 CANN 内置
libopapi），验证脚本无需改动。不动 CANN 安装。

验收标准（每条都有现成脚本与基准数据）：

1. `minrepro_pre_sinkhorn_ds2.py` + `worst_ds2_case.pt`：ds2 从 0.3353（错）
   变为 ≈2.37e-2（fp64 真值），相对误差 ≤1e-3；
2. `parity_scale_cliff.py`（3 seeds，|z| 阶梯至 65）：op 对 CPU fp64 全程
   ≤1e-3，不再出现 10~74× 分离（当前 z≥5 即劣化、z≥20 反号）；
3. `ckpt_z_analysis.py` 用崩溃 run（mtp_260923_201235）iter_2000 真实参数
   实测：L19.attn（z=22）ds2 从 34.5×（反号）→ ≤1e-3，dx 从 0.147 → ≤3e-3；
   对照模块（z≈0）保持 6e-5 水平不劣化；
4. `parity_fused_bwd.py`：前向逐位不变（不触碰前向核）；soak 2000 次零
   NaN、逐位确定；hidden%128 约束不变；
5. 性能门槛见 §2.2；30-iter 冒烟 iter 时间在跨进程噪声内。

## 5. 上线路径与工作量

1. 本地改内核 + build_out 构建 + 上述 5 条验证（**约 2–3 天内核改造 +
   1 天验证**；风险点：UB 预算与跨步布局细节，兜底是降 tileCoreBS）；
2. （可选）云上 `use-fused-mhc: true` 小规模冒烟对照历史崩溃点；
3. 提 MR 到 gitcode.com/cann/ops-transformer（v9.1.1 与 CANN 内置版同源，
   修一处两边受益），并附 CANN 缺陷工单：三份复现脚本 +
   `worst_ds2_case.pt` + CPU fp32 复现脚本（decompose）已备齐；
4. 上游合入前，主线继续走 mhc_lite 最终栈（不经该算子，已通过对抗值域
   压测，见 `../mhc_lite/report_finalstack_stress.md`）。
