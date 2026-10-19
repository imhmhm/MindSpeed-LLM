# mhc_lite 快速上手（分支保留版）

mhc_lite：用"24 个置换矩阵的 softmax 凸组合"精确构造双随机混合矩阵（零迭代，
不经 CANN sinkhorn 算子），模块/封装/算子源码全部在本仓库。设计与验证细节见
本目录 `README.md`（研究记录）。

## 三档运行（依赖从零到全）

1. **纯 PyTorch**：不设任何 `MHC_LITE_*` 环境变量。零额外依赖、零编译。
2. **triton**：`export MHC_LITE_TRITON=1`。无需外部仓库/预编译。
3. **最终栈（正式训练推荐）**：需要一次性构建的算子产物 + 5 个开关。

## 最终栈算子：构建一次，处处拷贝（仅第 3 档需要）

```bash
git clone --branch v9.1.1 https://gitcode.com/cann/ops-transformer.git /cache/ops-transformer
cd <本仓库>
bash experiments/mhc_lite/ascendc_lite_pre/sync_and_build.sh
bash experiments/mhc_lite/ascendc_lite_pre_fused/sync_and_build.sh
bash experiments/mhc_lite/ascendc_lite_pre_grad/sync_and_build.sh   # 这一条把全部算子打进同一个包
# 产物：/cache/ops-transformer/build_out（~15 MB，910B1–B4 通用；CANN 换大版本需按对应 tag 重编）
```

`/cache` 是 webstudio 与 mtp 云上主机都有的本地大盘，三个构建脚本不传参数时
默认就用 `/cache/ops-transformer`（也可传参指定其他克隆目录）。

运行环境（torch_npu 导入前）：

```bash
export ASCEND_CUSTOM_OPP_PATH=/cache/ops-transformer/build_out
export MHC_LITE_ASCENDC=1 MHC_LITE_ASCENDC_GRAD=1 MHC_LITE_ASCENDC_CHAIN=1
export MHC_LITE_NATIVE_POST_BWD=1 MHC_LITE_POST_DIRECT=1
```

## 起训练作业

- 配置模板：`wisemlops/configs/ailab_slm_mhclite_0_5b_pretrain.yaml`
  （`entry: pretrain_gpt_mhc_lite.py`，`model_name: ailab_slm_mhclite`，
  `enable-mhc: true`，`spec: ...gpt_mhc_lite_spec, layer_spec`）；
- 作业脚本：`wisemlops/jobs/webstudio_pretrain_ailab_slm_mhclite_0_5b.sh`
  （常规）与 `webstudio_pretrain_ailab_slm_mhclite_finalstack.sh`
  （最终栈 30 步冒烟，环境变量已配好）；
- 检查点转 HF：`ailab_slm_mhclite` 已注册到 `task_ckpt_mcore2hf.py`。

## 自检脚本

正确性/精度/压测脚本（`parity_test.py`、`probe_finalstack_accuracy.py`、
`probe_finalstack_stress.py`）不在本分支，见研究分支
`zhanghengheng/dev_26.1.0_mhc` 的 `experiments/mhc_lite/`。

## 两个已知怪癖

- 进程内第一次调用 `aclnnMhcPostBackward` 可能报错（需任意一次 backward
  预热；训练作业结构上安全，孤立基准才需要注意）；
- `dscale/dbase` 两个梯度是多核原子累加，多次运行允许 ~1e-7 相对重排，
  其余输出逐位可复现。
