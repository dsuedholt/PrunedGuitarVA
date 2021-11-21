import os


def latest_checkpoint_from_folder(log_dir):
    log_dir = os.path.join(log_dir, "lightning_logs")

    latest_run_dir = os.path.join(log_dir, sorted(os.listdir(log_dir))[-1])

    ckpt_path = os.path.join(latest_run_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_path, sorted(os.listdir(ckpt_path))[-1])

    return ckpt_path
