import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'hydra'))

import torch
from libs import load_opt
from libs.worker import *

if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    gpu_capability = torch.cuda.get_device_capability(gpu_id)
    if gpu_capability[0] < 8:
        os.environ["TRITON_F32_DEFAULT"] = "ieee"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="job name")
    parser.add_argument('--ckpt', type=str, help="checkpoint name")
    args = parser.parse_args()

    root = os.path.join('experiments', args.name)
    try:
        opt = load_opt(os.path.join(root, 'opt.yaml'), is_training=False)
    except:
        raise ValueError('experiment folder not found')
    assert os.path.exists(os.path.join(root, 'models', f'{args.ckpt}.pth'))
    opt['_root'] = root
    opt['_ckpt'] = args.ckpt

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    evaluator_type = opt.get('meta', {}).get('evaluator_type', 'EvaluatorOriginal')
    try:
        evaluator_cls = globals()[evaluator_type]
    except KeyError:
        raise ValueError(f"Trainer class '{evaluator_type}' not found. Make sure it is imported.")
    evaluator = evaluator_cls(opt)
    evaluator.run()
