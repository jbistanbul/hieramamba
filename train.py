import warnings
import re
import argparse
import os
import shutil
import sys

warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*is deprecated.*")

sys.path.append(os.path.join(os.path.dirname(__file__), 'hydra'))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from libs import load_opt
from libs.worker import *
from libs.dist_utils import print0

if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    gpu_capability = torch.cuda.get_device_capability(gpu_id)
    if gpu_capability[0] < 8:
        os.environ["TRITON_F32_DEFAULT"] = "ieee"


def main(rank, opt):
    torch.cuda.set_device(rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print(f"Training process: {rank}")

    if opt['_distributed']:
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(
            backend='nccl', init_method='env://',
            rank=rank, world_size=opt['_world_size']
        )
    trainer_type = opt.get('meta', {}).get('trainer_type', 'TrainerOriginal')
    try:
        trainer_cls = globals()[trainer_type]
    except KeyError:
        raise ValueError(f"Trainer class '{trainer_type}' not found. Make sure it is imported.")

    print0(f"Using trainer: {trainer_cls.__name__}")
    trainer = trainer_cls(opt)
    trainer.run()
    if opt['_distributed']:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help="training options")
    parser.add_argument('--name', type=str, help="job name")
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='use all visible GPUs with torch.distributed instead of the default single-GPU mode',
    )
    args = parser.parse_args()

    os.makedirs('experiments', exist_ok=True)
    root = os.path.join('experiments', args.name)
    os.makedirs(root, exist_ok=True)
    try:
        opt = load_opt(os.path.join(root, 'opt.yaml'), is_training=True)
    except:
        opt_path = os.path.join('opts', args.opt)
        opt = load_opt(opt_path, is_training=True)
        shutil.copyfile(opt_path, os.path.join(root, 'opt.yaml'))
        os.makedirs(os.path.join(root, 'models'), exist_ok=True)
        os.makedirs(os.path.join(root, 'states'), exist_ok=True)
    opt['_root'] = root
    opt['_resume'] = (
        os.path.exists(os.path.join(root, 'models', 'last.pth'))
        and os.path.exists(os.path.join(root, 'states', 'last.pth'))
    )

    visible_gpus = torch.cuda.device_count()
    opt['_distributed'] = args.distributed and visible_gpus > 1
    opt['_world_size'] = visible_gpus if opt['_distributed'] else 1

    if args.distributed and visible_gpus <= 1:
        print0('Distributed training requested, but fewer than 2 visible GPUs were found. Falling back to single-GPU mode.')

    if opt['_distributed']:
        mp.spawn(main, nprocs=visible_gpus, args=(opt, ))
    else:
        main(0, opt)
