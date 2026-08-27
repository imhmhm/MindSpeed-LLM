import os
from typing import List
import subprocess
import re
import glob
import shutil
import time
import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ProcessPoolExecutor

import moxing as mox


def get_npus_per_node():
    """
    resolve `npus_per_node` in modelarts and notebook
    """
    n = os.environ.get("MA_NUM_GPUS")
    if n:
        return int(n)
    out = subprocess.check_output(
        "npu-smi info -l | grep -c 'NPU ID'", shell=True, text=True)
    return int(out.strip())


def get_master_addr():
    vc_worker_hosts = os.getenv('VC_WORKER_HOSTS')
    if not vc_worker_hosts:
        return "localhost"
    return vc_worker_hosts.split(',')[0].strip()


def get_master_addr_legacy():
    ma_vj_name = os.environ.get("MA_VJ_NAME")
    if not ma_vj_name:
        return "localhost"
    master_addr = (f"{ma_vj_name}-{os.environ.get('MA_TASK_NAME')}-"
                   f"{os.environ.get('MASTER_RANK', '0')}.{ma_vj_name}")
    return master_addr


def resolve_cluster():
    """
    resolve environment variables in modelarts and notebook
    https://support.huaweicloud.com/intl/zh-cn/usermanual-standard-modelarts/develop-models-0104.html
    """
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"]="1"
    os.environ["PYTORCH_NPU_ALLOC_CONF"]="expandable_segments:True"
    
    nnodes = int(os.environ.get("MA_NUM_HOSTS", "1"))
    node_rank = int(os.environ.get("VC_TASK_INDEX", "0")) if nnodes > 1 else 0
    npus_per_node = get_npus_per_node()
    master_addr = get_master_addr_legacy()
    master_port = os.environ.get("MASTER_PORT", "6001")

    os.environ["NNODES"] = str(nnodes)
    os.environ["NODE_RANK"] = str(node_rank)
    os.environ["NPUS_PER_NODE"] = str(npus_per_node)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(nnodes * npus_per_node)

    return {"nnodes": nnodes, "node_rank": node_rank, "npus_per_node": npus_per_node,
            "master_addr": master_addr, "master_port": master_port}


class BarrierContextManager:
    def __init__(self, flag_path, nnodes=8, node_rank=0, sleep=30, rerank_node=False):
        """
        This context manager provides a multi-node barrier to finish pre-tasks
        before initializing distributed gpu programs
        - The barrier adopts files or directories on the shared storage as ready flags
        - flag of each node is the dir ```f"{flag_path}/{node_rank}"```
        - spod_info of each node is the dir ```f"{flag_path}/spod_info/{node_rank}/{super_pod_id}_{server_index}"```
        """
        self.flag_path = flag_path
        self.nnodes = nnodes
        self.node_rank = node_rank  # original node_rank before rerank
        self.sleep = sleep
        self.rerank_node = rerank_node
        if self.flag_path:
            self.flag_node = os.path.join(self.flag_path, f"{self.node_rank:03d}")
        else:
            self.flag_node = None

    def all_nodes_pre_tasks_ready(self):
        ready = True
        for i in range(self.nnodes):
            target_file = os.path.join(self.flag_path, f"{i:03d}")
            if not mox.file.exists(target_file):
                ready = False
        return ready

    def all_nodes_spod_info_ready(self):
        ready = True
        for i in range(self.nnodes):
            target_file = os.path.join(self.flag_path, "spod_info", f"{i:03d}")
            if not mox.file.exists(target_file):
                ready = False
        return ready

    def rerank_node_910c(self):
        """
        rerank NODE_RANK in 910c tasks according to (`super_pod_id`, `server_index`)
        """
        spod_info_folder = os.path.join(self.flag_path, "spod_info", f"{self.node_rank:03d}")

        if mox.file.exists(spod_info_folder):
            mox.file.remove(spod_info_folder, recursive=True)

        ## zhh: spod_info: "[`super_pod_id`, `server_index`]"
        spod_info = subprocess.check_output(
            "npu-smi info -t spod-info -i 0 -c 0 | awk '/Super Pod ID|Server Index/{print $NF}'",
            shell=True,
        ).decode('utf-8').strip().split()

        mox.file.make_dirs(os.path.join(spod_info_folder, f"{spod_info[0]}_{spod_info[1]}"))

        _spod_info_ready = self.all_nodes_spod_info_ready()
        while not _spod_info_ready:
            print(f"Not ready for reranking `NODE_RANK`, sleep for {self.sleep} secs.", flush=True)
            time.sleep(self.sleep)
            _spod_info_ready = self.all_nodes_spod_info_ready()

        spod_info_list = []
        spod_info_dir_all_node = sorted(mox.file.glob(f"{self.flag_path}/spod_info/[0-9][0-9][0-9]/*"))
        for _dir in spod_info_dir_all_node:
            ## zhh: spod_info_tuple: (`super_pod_id`, `server_index`)
            spod_info_tuple = tuple(map(int, os.path.basename(_dir).split("_")))
            spod_info_list.append(spod_info_tuple)
        ## zhh: debug
        print(f"**** spod_info (super_pod_id, server_index):\n{spod_info_list}\n", flush=True)

        rerank_mapping_old_to_new, rerank_mapping_new_to_old = get_sorted_index_mapping(spod_info_list)
        os.environ['VC_TASK_INDEX'] = str(rerank_mapping_old_to_new[self.node_rank])
        os.environ['ORIGINAL_VC_TASK_INDEX'] = str(self.node_rank)
        os.environ['MASTER_RANK'] = str(rerank_mapping_new_to_old[0])
        resolve_cluster()
        print(f"**** NODE_RANK after rerank (old: new):\n{rerank_mapping_old_to_new}\n", flush=True)

    def __enter__(self):
        if self.flag_node:
            if mox.file.exists(self.flag_node):
                mox.file.remove(self.flag_node, recursive=True)
        if self.rerank_node:
            self.rerank_node_910c()
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.flag_node:
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            mox.file.make_dirs(os.path.join(self.flag_node, timestamp))
            ready = self.all_nodes_pre_tasks_ready()
            while not ready:
                print(f"Not ready for executing the following commands, sleep for {self.sleep} secs.", flush=True)
                time.sleep(self.sleep)
                ready = self.all_nodes_pre_tasks_ready()


def _download_obs_ckpt_to_load(tp_size, pp_size,
                               node_rank, nnodes, nproc_per_node,
                               ckpt_load_obs, ckpt_load_local):

    os.makedirs(ckpt_load_local, exist_ok=True)

    tic = time.time()

    if pp_size > 1:
        file_list = mox.file.glob(
            os.path.join(ckpt_load_obs, "mp_rank_[0-9][0-9]_[0-9][0-9][0-9]")
        )
        print(f"Found file list: {file_list}", flush=True)
        assert (nnodes * nproc_per_node) % (tp_size * pp_size) == 0
        dp_size = ((nnodes * nproc_per_node) / (tp_size * pp_size))

        for obs_ckpt_slice in file_list:
            slice_name = os.path.basename(obs_ckpt_slice)
            pp_stage = int(slice_name[-3:])
            tdp_size = tp_size * dp_size

            rank_range_start = node_rank * nproc_per_node
            rank_range_end = (node_rank + 1) * nproc_per_node
            ranks = [rk for rk in range(rank_range_start, rank_range_end)]

            pp_stage_each_rank = [rk // tdp_size for rk in ranks]
            pp_stages_this_node = list(set(pp_stage_each_rank))
            if pp_stage in pp_stages_this_node:
                local_ckpt_slice = os.path.join(ckpt_load_local, slice_name)
                print(f"Copy checkpoint from {obs_ckpt_slice}\nto {local_ckpt_slice}")
                mox.file.copy_parallel(obs_ckpt_slice, local_ckpt_slice)
    else:
        mox.file.copy_parallel(ckpt_load_obs, ckpt_load_local)

    print(f'**** copying checkpoints from obs takes {(time.time() - tic) / 60} minutes', flush=True)
    os.system("du -h {}".format(ckpt_load_local))


def copy_from_obs(obs_file, local_file, skip_existing=True):
    if skip_existing:
        if mox.file.exists(local_file):
            obs_file_size = mox.file.get_size(obs_file)
            local_file_size = mox.file.get_size(local_file)
            if obs_file_size == local_file_size:
                print(f"local file {local_file} exists\n"
                      f"skip copying {obs_file}\n", flush=True)
                return
    local_file_dir = os.path.dirname(local_file)
    os.makedirs(local_file_dir, exist_ok=True)
    mox.file.copy(obs_file, local_file)


class SyncCKPT:
    def __init__(self, src_path, dst_path, interval=-1):
        self.src_path = src_path
        self.dst_path = dst_path
        self.interval = interval

    def init_scheduler(self):
        executors = {'processpool': ProcessPoolExecutor(1)}
        scheduler = BackgroundScheduler(executors=executors)
        return scheduler

    def start(self):
        if self.interval <= 0:
            return
        self.scheduler = self.init_scheduler()
        self.scheduler.add_job(self.copy_ckpt,
                               trigger='interval',
                               seconds=self.interval,
                               id='copy_ckpt_job')
        # logging.getLogger('apscheduler').propagate = False
        logging.getLogger('apscheduler').setLevel(logging.WARNING)
        self.scheduler.start()

    def exit(self):
        if self.interval >= 0:
            self.scheduler.remove_job('copy_ckpt_job')
            self.scheduler.shutdown()
        ## final run
        mox.file.copy_parallel(self.src_path, self.dst_path)
        ## copy MTP logs to destination storage
        mtp_log_dir = "/opt/huawei/schedule-train/log"
        if os.path.exists(mtp_log_dir):
            mox.file.copy_parallel(mtp_log_dir, os.path.join(self.dst_path, '0_logs'))

        ## copy tensorboard to destination storage
        ## mindspeed-rl default tensorboard path
        mtp_rl_tb_dir = "/cache/algorithm/runs"
        ## mindspeed-llm default tensorboard path set in shell scripts
        mtp_tb_dir = "/cache/outputs/tensorboard"
        if os.path.exists(mtp_rl_tb_dir):
            mox.file.copy_parallel(mtp_rl_tb_dir, os.path.join(self.dst_path, 'tensorboard'))
        if os.path.exists(mtp_tb_dir):
            mox.file.copy_parallel(mtp_tb_dir, os.path.join(self.dst_path, 'tensorboard'))

        ## copy swanlab to destination storage
        ## path set by launch.py pre-tasks, kept beside tensorboard rather than under save
        mtp_sl_dir = "/cache/outputs/swanlab"
        if os.path.exists(mtp_sl_dir):
            mox.file.copy_parallel(mtp_sl_dir, os.path.join(self.dst_path, 'swanlab'))

        ## copy ray logs to destination storage
        ray_log_dir = "/tmp/ray/session_latest/logs"
        if os.path.exists(ray_log_dir):
            mox.file.copy_parallel(
                ray_log_dir,
                os.path.join(
                    self.dst_path, 'ray_logs',
                    f"{os.getenv('ORIGINAL_VC_TASK_INDEX', os.getenv('VC_TASK_INDEX', 0))}",
                )
            )

    def copy_ckpt(self):
        ckpt_local_history = glob.glob(f"{self.src_path}/iter_*")
        if len(ckpt_local_history) > 1:
            ckpt_local_history.sort()
            dir_to_copy = ckpt_local_history[0]
            try:
                iter_to_copy = os.path.basename(dir_to_copy)
                mox.file.copy_parallel(dir_to_copy,
                                       os.path.join(self.dst_path, iter_to_copy))
                print(f"**** Successfully copy {dir_to_copy} to\n {self.dst_path}", flush=True)
                shutil.rmtree(dir_to_copy)
                print(f"**** Successfully remove {dir_to_copy}", flush=True)

            except OSError as error:
                raise Exception(f"**** {dir_to_copy} can not be removed") from error


DIR_TO_FILTER = re.compile(r'^[-_]?((tp|pp|ep)[-_]?\d+[-_]?)+$', re.IGNORECASE)

def get_ckpt_tag(load_name):
    """Shorten a `load_name` to the segment that identifies the checkpoint.

    Walks the path right to left and returns the first segment that is not a
    parallel-layout tag, so `Qwen3-0.6B-Base/tp1pp1` yields `Qwen3-0.6B-Base`
    instead of `tp1pp1`.
    """
    segments = [seg for seg in load_name.split('/') if seg]
    if not segments:
        return ""
    for segment in reversed(segments):
        if not DIR_TO_FILTER.match(segment):
            return segment
    ## every segment is a layout tag; keep the last one rather than return empty
    return segments[-1]


## megatron takes one data source per run: a single --data-path blend carved up by
## --split, or independent --train/valid/test-data-path streams (blend_per_split,
## where --split does not apply). See megatron/core/datasets/
## blended_megatron_dataset_config.py __post_init__.
SPLIT_NAMES = ("train", "valid", "test")


def resolve_data_source(cfg, data_cache_dir_root):
    """Render --data-path or the per-split --{train,valid,test}-data-path.

    Returns the copy plan as [(dataset_name, file_name_prefixes), ...]; every
    entry still has to be fetched into the cache. Sets the unused mode's keys to
    "" rather than leaving them as [], which render_args would emit as a bare
    flag that megatron then fails to unpack.
    """
    dataset_name = cfg.get("dataset_name") or ""
    prefixes_and_weights = list(cfg.get("data_prefixes_and_weights", []) or [])
    per_split = {
        name: (cfg.get(f"{name}_dataset_name") or "",
               list(cfg.get(f"{name}_data_prefixes_and_weights", []) or []))
        for name in SPLIT_NAMES
    }

    has_single = bool(dataset_name or prefixes_and_weights)
    has_per_split = any(ds or pw for ds, pw in per_split.values())
    if has_single and has_per_split:
        raise ValueError(
            "`dataset_name` / `data_prefixes_and_weights` are mutually exclusive with their "
            "`{train,valid,test}_` counterparts (megatron takes one data source per run)")
    if not has_single and not has_per_split:
        raise ValueError(
            "no data source: provide `dataset_name` / `data_prefixes_and_weights`, "
            "or their `{train,valid,test}_` counterparts")

    if has_single:
        ## `dataset_name` alone means the prefix repeats the directory name
        if not prefixes_and_weights:
            prefixes_and_weights = [dataset_name]
        _, path_and_weights, file_name_prefixes = get_data_path(
            prefixes_and_weights, data_cache_dir_root, dataset_name)
        cfg["data-path"] = path_and_weights
        for name in SPLIT_NAMES:
            cfg[f"{name}-data-path"] = ""
        return [(dataset_name, file_name_prefixes)]

    data_copy_plan = []
    for name in SPLIT_NAMES:
        split_dataset_name, split_prefixes = per_split[name]
        if not split_dataset_name and not split_prefixes:
            ## megatron logs "blend not provided for <split>" and skips the phase
            cfg[f"{name}-data-path"] = ""
            continue
        if not split_prefixes:
            split_prefixes = [split_dataset_name]
        _, path_and_weights, file_name_prefixes = get_data_path(
            split_prefixes, data_cache_dir_root, split_dataset_name)
        cfg[f"{name}-data-path"] = path_and_weights
        data_copy_plan.append((split_dataset_name, file_name_prefixes))
    cfg["data-path"] = ""
    if cfg.get("split"):
        print(f"**** per-split data paths given, dropping split={cfg['split']!r}", flush=True)
    cfg["split"] = ""
    return data_copy_plan

    
def get_data_path(file_name_prefixes_and_weights, data_local_root, datasets):
    """
    [1]
    args
        file_name_prefixes_and_weights: ['data']
    return
        ## shell=True
        data_path_str_megatron: f"{data_local_root}/{datasets}/data"
        ## shell=False
        file_name_path_and_weights: [f'{data_local_root}/{datasets}/data']
        file_name_prefixes: ['data']
    [2]
    args
        file_name_prefixes_and_weights: [0.3, 'data1', 0.4, 'data2', ...]
    return
        ## shell=True
        data_path_str_megatron: f"0.3 {data_local_root}/{datasets}/data1 0.4 {data_local_root}/{datasets}/data2 ..."
        ## shell=False
        file_name_path_and_weights: ['0.3', f'{data_local_root}/{datasets}/data1', '0.4', f'{data_local_root}/{datasets}/data2', ...]
        file_name_prefixes: ['data1', 'data2', ...]
    """
    file_name_prefixes = []
    file_name_path_and_weights = []
    if len(file_name_prefixes_and_weights) == 1:
        file_name_prefixes = file_name_prefixes_and_weights
        file_name_path_and_weights.append(
            ## f"{data_local_root}/{datasets}/{file_name_prefixes_and_weights[0]}"
            os.path.join(data_local_root, datasets, file_name_prefixes_and_weights[0].strip())
        )
    else:
        if len(file_name_prefixes_and_weights) % 2 != 0:
            raise Exception("data files more than one should be in the form of"
                            " weights and files: ```[0.3, 'data1', 0.4, 'data2', ...]```")
        num_datasets = len(file_name_prefixes_and_weights) // 2
        for i in range(num_datasets):
            _weight = file_name_prefixes_and_weights[2 * i]
            _prefix = file_name_prefixes_and_weights[2 * i + 1].strip()
            if not (isinstance(_weight, int) or isinstance(_weight, float)):
                raise Exception("weight should be int or float")
            file_name_prefixes.append(_prefix)
            file_name_path_and_weights.append(
                str(_weight)
            )  # append weight
            file_name_path_and_weights.append(
                ## f"{data_local_root}/{datasets}/{_prefix}"
                os.path.join(data_local_root, datasets, _prefix)
            )  # append path

    data_path_str_megatron = " ".join(file_name_path_and_weights)
    return data_path_str_megatron, file_name_path_and_weights, file_name_prefixes


def get_sorted_index_mapping(data: List):
    """
    mapping from old index to new index after using python `sorted`
    e.g.
    ```data = [4,2,1,3]```
    ```index_mapping_old_to_new = {0: 3, 1: 1, 2: 0, 3: 2}```
    ```index_mapping_new_to_old = {0: 2, 1: 1, 2: 3, 3: 0}```
    """
    indexed_data = list(enumerate(data))
    sorted_indexed_data = sorted(indexed_data, key=lambda x: x[1])

    index_mapping_old_to_new = {}
    index_mapping_new_to_old = {}
    for new_index, (old_index, value) in enumerate(sorted_indexed_data):
        index_mapping_old_to_new[old_index] = new_index
        index_mapping_new_to_old[new_index] = old_index

    index_mapping_old_to_new = dict(sorted(index_mapping_old_to_new.items()))
    index_mapping_new_to_old = dict(sorted(index_mapping_new_to_old.items()))

    return index_mapping_old_to_new, index_mapping_new_to_old
