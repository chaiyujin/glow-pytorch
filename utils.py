import os
import re
import torch
from shutil import copyfile


def _file_at_step(step):
    return "save_{}k{}.pkg".format(int(step // 1000), int(step % 1000))


def _file_best():
    return "trained.pkg"


def save(global_step, graph, optim, criterion_dict=None, pkg_dir="", is_best=False, max_checkpoints=None):
    if optim is None:
        raise ValueError("cannot save without optimzier")
    state = {
        "global_step": global_step,
        # DataParallel wrap model in attr `module`.
        "graph": graph.module.state_dict() if hasattr(graph, "module") else graph.state_dict(),
        "optim": optim.state_dict(),
        "criterion": {}
    }
    if criterion_dict is not None:
        for k in criterion_dict:
            state["criterion"][k] = criterion_dict[k].state_dict()
    save_path = os.path.join(pkg_dir, _file_at_step(global_step))
    best_path = os.path.join(pkg_dir, _file_best())
    torch.save(state, save_path)
    if is_best:
        copyfile(save_path, best_path)
    if max_checkpoints is not None:
        history = []
        for file_name in os.listdir(pkg_dir):
            if re.search("save_\d*k\d*\.pkg", file_name):
                digits = file_name.replace("save_", "").replace(".pkg", "").split("k")
                number = int(digits[0]) * 1000 + int(digits[1])
                history.append(number)
        history.sort()
        while len(history) > max_checkpoints:
            path = os.path.join(pkg_dir, _file_at_step(history[0]))
            print("[Checkpoint]: remove {} to keep {} checkpoints".format(path, max_checkpoints))
            if os.path.exists(path):
                os.remove(path)
            history.pop(0)


def load(step_or_path, graph, optim=None, criterion_dict=None, pkg_dir="", device=None):
    step = step_or_path
    save_path = None
    if isinstance(step, int):
        save_path = os.path.join(pkg_dir, _file_at_step(step))
    if isinstance(step, str):
        if pkg_dir is not None:
            if step == "best":
                save_path = os.path.join(pkg_dir, _file_best())
            else:
                save_path = os.path.join(pkg_dir, step)
        else:
            save_path = step
    if save_path is not None and not os.path.exists(save_path):
        print("[Checkpoint]: Failed to find {}".format(save_path))
        return
    if save_path is None:
        print("[Checkpoint]: Cannot load the checkpoint with given step or filename or `best`")
        return

    # begin to load
    state = torch.load(save_path, map_location=device)
    global_step = state["global_step"]
    graph.load_state_dict(state["graph"])
    if optim is not None:
        optim.load_state_dict(state["optim"])
    if criterion_dict is not None:
        for k in criterion_dict:
            criterion_dict[k].load_state_dict(state["criterion"][k])

    print("[Checkpoint]: Load {} successfully".format(save_path))
    return global_step