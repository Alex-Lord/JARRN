import argparse
import os
from pathlib import Path
from src.src_main import AbsRel_depth
from src.networks import UNet
from src.utils import str2bool, DDPutils

import torch
import multiprocessing
import setproctitle
import numpy as np
import random
import datetime

# turn fast mode on
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        "options for AbsRel_depth  estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rgbd_dataset",
        action="store",
        type=list,
        default=[''],
        help="Path to RGB-depth folder",
        required=False
    )
    parser.add_argument(
        "--model",
        action="store",
        type=str,
        default='unet',
        help="unet-JARRN",
        required=False,
    )
    parser.add_argument(
        "--rgbd_dir",
        action="store",
        type=list,
        default=[],
        help="Path to RGB-depth folder",
        required=False
    )
    parser.add_argument(
        "--gpu_num",
        action="store",
        type=str,
        default="2",
        help="Select GPU",
        required=False
    )

    parser.add_argument(
        "--hole_dir",
        action="store",
        type=list,
        default=['/data1/name/JARRN/Hole_Dataset'],
        help="Path to Hole folder",
        required=False
    )
    parser.add_argument(
        "--save_dir",
        action="store",
        type=str,
        default=r'/data1/name/JARRN/result_JARRN/',
        help="Path to the directory for saving the logs and models",
        required=False
    )
    parser.add_argument(
        "--rezero",
        action="store",
        type=str2bool,
        default=True,
        help="whether to use the ReZero",
        required=False,
    )
    
    parser.add_argument(
        "--sobel",
        action="store",
        type=str2bool,
        default=True,
        help="whether to use the sobel operator",
        required=False,
    )
    parser.add_argument(
        "--msgrad",
        action="store",
        type=str2bool,
        default=True,
        help="whether to use the multi-scale gradient",
        required=False,
    )
    parser.add_argument(
        "--mode",
        action="store",
        type=str,
        default='mar',
        help="standardization methods: ['mean_robust(mar)','mean(ma)','median(md)']",
        required=False,
    )
    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        required=False,
        nargs="+",
        default=60,
        help="epochs numbers",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        required=False,
        nargs="+",
        default=4,
        # default=1,
        help="batch sizes",
    )
    parser.add_argument(
        "--resume_train",
        action="store",
        type=str2bool,
        default=True,
        help="resume train or not",
        required=False,
    )
    parser.add_argument(
        "--model_dir",
        action="store",
        type=lambda x: Path(x),
        default=r'None',
        help="Path to load models",
        required=False
    )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        default=6587,
        help="DDP port",
    )

    args = parser.parse_args()
    
    return args

def generate_random_seed(seed):
    if (seed is not None) and (seed != -1):
        return seed
    seed = np.random.randint(2 ** 31)
    return seed

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def DDP_main(rank, world_size):
    args = parse_arguments()
    seed = generate_random_seed(3407+rank)
    set_random_seed(seed)
    if args.rezero:
        args.save_dir += '_rz'
    else:
        args.save_dir += '_bn'
    if args.msgrad:
        if args.sobel:
            args.save_dir += '_sb'
        else:
            args.save_dir += '_gd'
            
    args.save_dir += '_' + args.mode
    
    args.save_dir = Path(args.save_dir)
    # DDP components
    DDPutils.setup(rank, world_size, args.port)

    
    model_dict = { 'unet':UNet(rezero=args.rezero)}
    network = model_dict[args.model]
    
    if rank == 0:
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print("Started Time:", formatted_time)
        print(f'We use {args.model} model, we have {model_dict}')
        print_model_parm_nums(network)
        
    semigan = AbsRel_depth(
        network,
        rank,
    )
    args.resume_train = False

    if rank == 0:
        print(f"Selected arguments: {args}")
    # resume train
    if args.resume_train:
        if rank == 0:
            print('resume training...')
            # load everything
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.model_dir, map_location=map_location)
    else:
        checkpoint = None

    semigan.train(
        args=args,
        rank=rank,
        learning_rate=0.0002,
        feedback_factor=1000,
        checkpoint_factor=2,
        num_workers=4,
        checkpoint=checkpoint,
    )
    if rank == 0:
        print_model_parm_nums(network)
    DDPutils.cleanup()

if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn')
    args = parse_arguments()


    args.gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    n_gpus = torch.cuda.device_count()
    if torch.cuda.is_available():
        DDPutils.run_demo(DDP_main, n_gpus)  