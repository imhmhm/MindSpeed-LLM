# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""SwanLab tracking, installed into Megatron's wandb writer slot.

Megatron reaches its experiment tracker only through get_wandb_writer(), and every
consumer uses just two methods -- writer.log(data, step) and writer.finish(). SwanLab's
Run exposes both with the same signatures, so attaching a Run to _GLOBAL_WANDB_WRITER
reuses all existing metric call sites instead of duplicating them.
"""
import os
from functools import wraps


## zhh: 0.9.0 的 Literal["disabled", "online", "local", "offline"]; 旧名 "cloud" 会被
## Settings.validate_mode 映射到 "online", 这里只暴露规范名, 避免同一模式两种拼法
SWANLAB_MODES = ['online', 'local', 'offline', 'disabled']


def set_wandb_writer_wrapper(fn):
    """Put a SwanLab run in the wandb writer slot when --swanlab-project is set."""
    @wraps(fn)
    def wrapper(args):
        if not getattr(args, 'swanlab_project', ''):
            return fn(args)

        ## zhh: 与 wandb 同样只在最后一个 rank 建 writer -- 打点处统一用 is_last_rank()
        ## 取 writer, 其余 rank 拿到 None 而跳过; 若每个 rank 都 init 会开出 world_size 个实验
        if args.rank != (args.world_size - 1):
            return

        from megatron.training import global_vars

        log_dir = args.swanlab_save_dir
        if not log_dir:
            ## zhh: 对齐 wandb 的默认位置 (args.save/wandb); args.save 可能为 None,
            ## 此时退回 swanlab 自己的默认目录
            log_dir = os.path.join(args.save, 'swanlab') if args.save else 'swanlog'

        import swanlab
        global_vars._GLOBAL_WANDB_WRITER = swanlab.init(
            project=args.swanlab_project,
            name=args.swanlab_exp_name or None,
            log_dir=log_dir,
            mode=args.swanlab_mode,
            config=vars(args),
        )
    return wrapper


## zhh: 以下两个是 megatron/training/wandb_utils.py 的空实现替身. 那边用
## writer.Artifact / run.log_artifact / run.use_artifact 做 ckpt 版本登记, 是 wandb
## 独有的 Artifact 体系, swanlab 无对应物. on_load 那侧原本裹了 try/except 不会炸,
## 但 checkpointing.py 里的 on_save 没有保护, 首次存盘就会 AttributeError, 所以两个都换掉.


def swanlab_noop_on_save_checkpoint_success(checkpoint_path, tracker_filename, save_dir, iteration):
    """No-op stand-in: SwanLab has no wandb-style artifact registry."""


def swanlab_noop_on_load_checkpoint_success(checkpoint_path, load_dir):
    """No-op stand-in: SwanLab has no wandb-style artifact registry."""
