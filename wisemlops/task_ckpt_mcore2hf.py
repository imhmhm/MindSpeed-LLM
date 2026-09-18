import os
import argparse
import subprocess

import moxing as mox

## per-model convert_ckpt_v2.py flags beyond the shared mcore->hf base set
CONVERTERS = {
    "ailab_slm_0_5b____v2": ["--model-type-hf", "llama2",],
    "Qwen3-0.6B": ["--model-type-hf", "qwen3",],
}

LOCAL_CKPT_ROOT = "/cache/inputs/ckpt"
CONVERT_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "convert_ckpt_v2.py")


def run_mcore2hf(task_names, iters, model, task_dir_root, use_obs):
    """
    pull ckpt (+ hf ref model) from obs or convert in place
    run convert_ckpt_v2.py per task x iter
    push mg2hf back to obs

    `iters` is one iteration list per task: iters[i] belongs to task_names[i],
    e.g. task_names=["job_a", "job_b"], iters=[[500, 1000], [2000]].

    Called in-process by launch.py
    """
    if not task_names or not iters:
        raise ValueError("task_names and iters must be non-empty")
    if len(iters) != len(task_names):
        raise ValueError(f"iters must hold one iteration list per task: "
                         f"got {len(iters)} lists for {len(task_names)} tasks")
    for _task, _task_iters in zip(task_names, iters):
        if not isinstance(_task_iters, (list, tuple)) or not _task_iters:
            raise ValueError(f"iters for task {_task!r} must be a non-empty list, "
                             f"got {_task_iters!r}")
    if model not in CONVERTERS:
        raise ValueError(f"unknown model {model!r}; add it to CONVERTERS: {sorted(CONVERTERS)}")

    if use_obs:
        hf_cfg_dir = f"{LOCAL_CKPT_ROOT}/hf_model/{model}"
        mox.file.copy_parallel(f"{task_dir_root}/hf_model/{model}", hf_cfg_dir)
        ckpt_root = LOCAL_CKPT_ROOT
    else:
        ckpt_root = f"{task_dir_root}/ckpt"
        hf_cfg_dir = f"{task_dir_root}/hf_model/{model}"

    cmd_env = "source /usr/local/Ascend/cann/set_env.sh"

    for idx, _task in enumerate(task_names):
        load_dir = f"{ckpt_root}/mcore/{_task}"
        save_dir = f"{load_dir}/mg2hf"

        for _iter in iters[idx]:
            iter_pad = f"{int(_iter):07d}"

            if use_obs:
                mox.file.copy_parallel(
                    f"{task_dir_root}/ckpt/mcore/{_task}/iter_{iter_pad}",
                    f"{load_dir}/iter_{iter_pad}",
                )
                ## no tracker file needed: --ckpt-iter selects the source iteration directly

            print(f"**** mcore2hf: task={_task} iter={_iter}", flush=True)
            cmd = (
                f"{cmd_env}"
                f" && python {CONVERT_SCRIPT}"
                f" --load-model-type mg --save-model-type hf"
                f" --load-dir {load_dir}"
                f" --ckpt-iter {_iter}"
                f" --save-dir {save_dir}"
                f" --hf-cfg-dir {hf_cfg_dir}"
                f" --merge-layers-safetensors"
                + "".join(f" {flag}" for flag in CONVERTERS[model])
            )
            subprocess.run(cmd, shell=True, check=True)

            if use_obs:
                mox.file.copy_parallel(
                    f"{save_dir}/iter_{iter_pad}",
                    f"{task_dir_root}/ckpt/mcore/{_task}/mg2hf/iter_{iter_pad}",
                )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_names", nargs="*", default=[],
                        help="mcore ckpt task names (training job names)")
    parser.add_argument("--iters", nargs="*", default=[], type=lambda s: [int(i) for i in s.split(",") if i],
                        help="one comma-separated iteration list per task, e.g. "
                             "--task_names job_a job_b --iters 500,1000 2000")
    parser.add_argument("--model", required=True,
                        help="model name: selects CONVERTERS entry and hf_model dir")
    parser.add_argument("--task_dir_root", required=True,
                        help="obs root (use_obs=1) or mounted data root (use_obs=0)")
    parser.add_argument("--use_obs", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    run_mcore2hf(args.task_names, args.iters, args.model, args.task_dir_root, args.use_obs)


if __name__ == "__main__":
    main()
