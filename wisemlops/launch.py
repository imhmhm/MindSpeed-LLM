import os
import re
import time
import argparse
import subprocess

from omegaconf import OmegaConf

from ma_utils import (BarrierContextManager, copy_from_obs, _download_obs_ckpt_to_load,
                      resolve_data_source, get_ckpt_tag, SyncCKPT, resolve_cluster,
                      SPLIT_NAMES)
from task_ckpt_mcore2hf import run_mcore2hf
import moxing as mox


## top-level config keys that are NOT mindspeed-llm training args (not rendered to the entry script)
STRUCTURAL_KEYS = {
    "entry", "sync_ckpt", "copy_data_to_cache", "mcore2hf_after_training",
    "model_name", "load_name", "load_iter",
    "dataset_name", "data_prefixes_and_weights", "is_pretrain_data", "is_pack_varlen_data",
    "train_dataset_name", "train_data_prefixes_and_weights",
    "valid_dataset_name", "valid_data_prefixes_and_weights",
    "test_dataset_name", "test_data_prefixes_and_weights",
    "tokenizer_tag",
    "huashan_obs_mtp_task", "huashan_obs_ml_data",
    "mount_dataset_mtp_task", "mount_dataset_ml_data",
    "enable_tensorboard", "enable_swanlab",
}



def build_torchrun_launcher():
    """torchrun launcher args from os.environ (canonical after resolve_cluster)."""
    return [
        "--nproc_per_node", os.environ["NPUS_PER_NODE"],
        "--nnodes",         os.environ["NNODES"],
        "--node_rank",      os.environ["NODE_RANK"],
        "--master_addr",    os.environ["MASTER_ADDR"],
        "--master_port",    os.environ["MASTER_PORT"],
    ]


def render_args(cfg):
    """flat config -> entry-script argv. See module docstring for the convention.
    """
    argv = []
    for key, value in cfg.items():
        if key in STRUCTURAL_KEYS:
            continue
        if OmegaConf.is_config(value):
            value = OmegaConf.to_object(value)
        if value is False or value is None or value == "":
            continue
        flag = "--" + str(key)
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            ## shell=False: argparse nargs='*'/'+' args (--spec, --data-path, ...) take one argv element per item
            argv.append(flag)
            argv.extend(str(item) for item in value)
        else:
            argv.extend([flag, str(value)])
    return argv


def validate(cfg):
    required = ["save", "tokenizer-name-or-path"]
    missing = [k for k in required if not cfg.get(k)]
    if not cfg.get("data-path") and not any(
            cfg.get(f"{name}-data-path") for name in SPLIT_NAMES):
        missing.append("data-path (or train/valid/test-data-path)")
    if missing:
        raise ValueError(f"launch.py: missing required args after pre-tasks: {missing}")


def pre_tasks(cfg):
    """
    - determine huanshan(obs) or mount dataset
    - render `--data-path` (or `train/valid/test-data-path`, megatron style)
    - sync the following tasks across all nodes within `BarrierContextManager` 
        -> 910c rerank node (optional)  
        -> copy code to cache (optional)
        -> copy data to cache (optional)
        -> copy loading ckpt to cache 
        -> copy hf model tokenizer to cache
    - render `job_name` using time stamp
    - ckpt saving path
    - tensorboard / swanlab paths
    """
    model_name = cfg.model_name
    load_name = cfg.get("load_name") or ""
    load_iter = cfg.get("load_iter") or ""
    seq_len = cfg.get("seq-length")
    is_pretrain_data = cfg.get("is_pretrain_data", False)
    is_pack_varlen_data = cfg.get("is_pack_varlen_data", False)

    ## data-mode defaults for the attention/position flags; set only when absent.
    if is_pretrain_data:
        data_mode_defaults = {"neat-pack": False, "reset-attention-mask": False,
                              "reset-position-ids": False, "no-pad-to-seq-lengths": False}
    elif is_pack_varlen_data:
        data_mode_defaults = {"neat-pack": True, "reset-attention-mask": True,
                              "reset-position-ids": True, "no-pad-to-seq-lengths": False}
    else:
        data_mode_defaults = {"neat-pack": False, "reset-attention-mask": False,
                              "reset-position-ids": False, "no-pad-to-seq-lengths": True}
    for key, value in data_mode_defaults.items():
        if cfg.get(key) is None:
            cfg[key] = value
    
    use_obs = bool(int(os.getenv("USE_OBS", 0)))
    if use_obs:
        mtp_task_dir_root = cfg.get("huashan_obs_mtp_task")
        ml_data_dir_root = cfg.get("huashan_obs_ml_data")
    else:
        mtp_task_dir_root = cfg.get("mount_dataset_mtp_task")
        ml_data_dir_root = cfg.get("mount_dataset_ml_data")
    if not mtp_task_dir_root or not ml_data_dir_root:
        keys = ("`huashan_obs_mtp_task` / `huashan_obs_ml_data`" if use_obs
                else "`mount_dataset_mtp_task` / `mount_dataset_ml_data`")
        raise ValueError(f"{keys} must be non-empty when USE_OBS={int(use_obs)}; "
                         f"pass them as CLI overrides in the job shell")

    tokenizer_tag = cfg.get("tokenizer_tag")
    if not tokenizer_tag:
        raise ValueError("`tokenizer_tag` must be set "
                         "it selects the ml_data__<tokenizer_tag>[_<seq-length>[_pack]] directory")
    data_dir_suffix = ""
    if not is_pretrain_data:
        data_dir_suffix += f"_{seq_len}"
        if is_pack_varlen_data:
            data_dir_suffix += "_pack"
        if cfg.stage == "reranker":
            data_dir_suffix += "_reranker"

    ## one directory for every split: the tokenizer / seq-length / packing tags are
    ## job-wide, so per-split streams must have been preprocessed the same way and
    ## differ only in their `{split}_dataset_name` subdirectory
    ml_data_dir = f"{ml_data_dir_root}/ml_data__{tokenizer_tag}{data_dir_suffix}"

    copy_data_to_cache = cfg.get("copy_data_to_cache")
    if use_obs:
        if copy_data_to_cache is False:
            print("**** USE_OBS=1 forces copy_data_to_cache=true ", flush=True)
        copy_data_to_cache = True
    elif copy_data_to_cache is None:
        copy_data_to_cache = False
    data_cache_dir_root = "/cache/inputs/data" if copy_data_to_cache else ml_data_dir
    cfg['no-shared-storage'] = copy_data_to_cache

    ## shell=False; also returns what the barrier block below has to fetch
    data_copy_plan = resolve_data_source(cfg, data_cache_dir_root)

    ma_vj_name = os.getenv("MA_VJ_NAME", "")
    if ma_vj_name:
        flag_name = re.sub(r"-vc\d+$", "", ma_vj_name)  # 超节点拓扑亲和调度
        job_name_prefix = "mtp"
    else:
        flag_name = job_name_prefix = "webstudio"
    flag_path = f"{mtp_task_dir_root}/outputs/flags/{flag_name}"
    mox.file.make_dirs(flag_path)

    resolve_cluster()

    with BarrierContextManager(
        flag_path=flag_path, 
        node_rank=int(os.environ["NODE_RANK"]), 
        nnodes=int(os.environ["NNODES"]),
        rerank_node=bool(int(os.getenv("RERANK_NODE", 0)) 
                         and os.getenv("sys_hyper_job") == "false")
    ):
        
        ## copy data to cache (.bin + .idx per prefix)
        ## when use_obs is False and copy_data_to_cache is False, src and cache paths are the same
        tic = time.time()
        
        ## one entry per data source: a single blend, or one per per-split stream.
        ## splits sharing a directory copy the same files, which skip_existing absorbs.
        for _dataset_name, file_name_prefixes in data_copy_plan:
            data_src_dir = os.path.join(ml_data_dir, _dataset_name)
            data_cache_dir = os.path.join(data_cache_dir_root, _dataset_name)

            for prefix in file_name_prefixes:
                for _data_path in mox.file.glob(os.path.join(data_src_dir, f"{prefix}*.bin")):
                    _data_path_prefix, _ = os.path.splitext(_data_path)
                    _data_filename = os.path.relpath(_data_path_prefix, data_src_dir)  ## 兼容两种data输入方式
                    copy_from_obs(f"{_data_path_prefix}.idx",
                                  os.path.join(data_cache_dir, f"{_data_filename}.idx"),
                                  skip_existing=True)
                    copy_from_obs(f"{_data_path_prefix}.bin",
                                  os.path.join(data_cache_dir, f"{_data_filename}.bin"),
                                  skip_existing=True)

        print(f"**** data copy time: {(time.time() - tic) / 60} minutes", flush=True)

        ## copy loading ckpt to cache (pp-shard-aware)
        if load_name:
            ckpt_load_cache_dir_root = "/cache/inputs/ckpt/mcore"
            cfg.load = f"{ckpt_load_cache_dir_root}/{load_name}"
            ckpt_src_dir = f"{mtp_task_dir_root}/ckpt/mcore/{load_name}/iter_{int(load_iter):07d}/"
            ckpt_load_cache_dir = f"{cfg.load}/iter_{int(load_iter):07d}/"
            tp = int(cfg.get("tensor-model-parallel-size", 1) or 1)
            pp = int(cfg.get("pipeline-model-parallel-size", 1) or 1)
            _download_obs_ckpt_to_load(
                tp, pp, 
                int(os.environ["NODE_RANK"]), int(os.environ["NNODES"]), int(os.environ["NPUS_PER_NODE"]), 
                ckpt_src_dir, ckpt_load_cache_dir
            )
            with open(f"{cfg.load}/latest_checkpointed_iteration.txt", "w") as f:
                f.write(str(load_iter))

        ## copy hf_model tokenizer to cache
        tokenizer_cache_dir = f"/cache/inputs/hf_model/{model_name}"
        mox.file.copy_parallel(f"{mtp_task_dir_root}/hf_model/{model_name}", tokenizer_cache_dir)
        cfg["tokenizer-name-or-path"] = tokenizer_cache_dir

    ## job timestamp / name / dirs
    timestamp_all = mox.file.glob(f"{flag_path}/[0-9][0-9][0-9]/*")
    job_timestamp = sorted(os.path.basename(t) for t in timestamp_all)[0]
    print(f"**** job_timestamp: {job_timestamp}", flush=True)
    ckpt_tag = get_ckpt_tag(load_name)
    job_name = f"{job_name_prefix}_{job_timestamp}__{model_name}__{ckpt_tag}"
    
    ckpt_save_cache_dir = f"/cache/outputs/ckpt/mcore/{job_name}"
    os.makedirs(ckpt_save_cache_dir, exist_ok=True)
    cfg.save = ckpt_save_cache_dir
    ckpt_save_dst_dir = f"{mtp_task_dir_root}/ckpt/mcore/{job_name}"

    ## tracker switches
    if cfg.get("enable_tensorboard", True):
        if cfg.get("tensorboard-dir") in (None, ""):
            cfg["tensorboard-dir"] = f"/cache/outputs/tensorboard/{job_name}"
    else:
        cfg["tensorboard-dir"] = ""

    if cfg.get("enable_swanlab", False):
        if cfg.get("swanlab-project") in (None, ""):
            cfg["swanlab-project"] = f"{model_name}__{ckpt_tag}"[:100]  ## swanlab requirement
        if cfg.get("swanlab-exp-name") in (None, ""):
            cfg["swanlab-exp-name"] = job_timestamp
        if cfg.get("swanlab-save-dir") in (None, ""):
            cfg["swanlab-save-dir"] = f"/cache/outputs/swanlab/{job_name}"
    else:
        cfg["swanlab-project"] = ""
        cfg["swanlab-exp-name"] = ""
        cfg["swanlab-mode"] = ""

    return ckpt_save_cache_dir, ckpt_save_dst_dir, job_name, mtp_task_dir_root


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", required=True, help="default config yaml path")
    opts, rest = parser.parse_known_args()

    cfg = OmegaConf.merge(OmegaConf.load(opts.config), OmegaConf.from_cli(rest))
                   
    save_cache, save_dst, job_name, task_root = pre_tasks(cfg)
    validate(cfg)
    cmd = ["torchrun", *build_torchrun_launcher(),
           cfg.entry, *render_args(cfg)]

    uploader = None
    sync_ckpt = cfg.get("sync_ckpt")
    if sync_ckpt is not None and sync_ckpt.get("enable"):
        
        uploader = SyncCKPT(
            sync_ckpt.get("src") or save_cache,
            sync_ckpt.get("dst") or save_dst,
            interval=int(sync_ckpt.get("interval", 60) or 60)
        )
        uploader.start()

    try:
        subprocess.run(cmd, check=True)
    finally:
        if uploader is not None:
            uploader.exit()


    mcore2hf = cfg.get("mcore2hf_after_training")
    if mcore2hf is not None and (iters := mcore2hf.get("iters")):
        if OmegaConf.is_config(iters):
            iters = OmegaConf.to_object(iters)
        iters = list(iters) if isinstance(iters, (list, tuple)) else [iters]
        run_mcore2hf(
            task_names=[job_name],
            iters=[iters],
            model=cfg.model_name,
            task_dir_root=task_root,
            use_obs=bool(int(os.getenv("USE_OBS", 0))),
        )


if __name__ == "__main__":
    main()
