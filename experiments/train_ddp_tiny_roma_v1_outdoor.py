import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from argparse import ArgumentParser
from pathlib import Path
import math
import numpy as np
import random 
from torch import nn
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import wandb
from PIL import Image
from torchvision.transforms import ToTensor
import datetime
from functools import partial
from romatch.benchmarks import MegadepthDenseBenchmark, ScanNetBenchmark
from romatch.benchmarks import Mega1500PoseLibBenchmark #, ScanNetPoselibBenchmark
from romatch.datasets.megadepth import MegadepthBuilder
from romatch.losses.robust_loss_tiny_roma import RobustLosses
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark, MegadepthDenseBenchmark, HpatchesHomogBenchmark
from romatch.train.train import train_k_steps
from romatch.checkpointing import CheckPoint
from model_tiny2 import TinyRomaV2_1 as  TinyRoma

resolutions = {"low":(448, 448), "medium":(14*8*5, 14*8*5), "high":(14*8*6, 14*8*6), "xfeat": (600,800), "big": (768, 1024)}

#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def train(args):
    dist.init_process_group('nccl')
    
    gpus = int(os.environ['WORLD_SIZE'])
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}")
    print("torch.cuda.device_count()", torch.cuda.device_count())
    device_id = rank % torch.cuda.device_count()
    romatch.LOCAL_RANK = device_id
    torch.cuda.set_device(device_id)
    

    print("romatch.RANK",romatch.RANK)
    resolution = "big"
    # wandb_log = not args.dont_log_wandb
    wandb_log=False
    experiment_name = Path(__file__).stem
    wandb_mode = "online" if wandb_log and rank == 0 else "disabled"
    wandb.init(project="romatch", entity=args.wandb_entity, name=experiment_name, reinit=False, mode = wandb_mode)

    # 获取当前时间的datetime对象
    current_time = datetime.datetime.now()

    # 使用strftime将datetime对象格式化为字符串
    formatted_time = current_time.strftime('%Y-%m-%d_%H:%M:%S')

    if not args.checkpoint_dir:
        checkpoint_dir = f"workspace/checkpoints-{formatted_time}/"
        # checkpoint_dir = "workspace/checkpoints-2024-12-31_17:44:17/"
    else:
        checkpoint_dir = args.checkpoint_dir

    h,w = resolutions[resolution]
    model = TinyRoma(freeze_xfeat = False).to(device_id)
    # Num steps
    global_step = 0
    batch_size = args.gpu_batch_size
    step_size = gpus*batch_size
    romatch.STEP_SIZE = step_size
    
    N = 2_000_000  # 2M pairs
    # checkpoint every
    k = 25000 // romatch.STEP_SIZE

    # Data
    mega = MegadepthBuilder(data_root="data/megadepth", loftr_ignore=True, imc21_ignore = True)
    use_horizontal_flip_aug = True
    normalize = False # don't imgnet normalize
    rot_prob = 0
    depth_interpolation_mode = "bilinear"
    megadepth_train1 = mega.build_scenes(
        split="train_loftr", min_overlap=0.01, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
        ht=h,wt=w, normalize = normalize
    )
    megadepth_train2 = mega.build_scenes(
        split="train_loftr", min_overlap=0.35, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
        ht=h,wt=w, normalize = normalize
    )
    megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)
    # Loss and optimizer
    depth_loss = RobustLosses(
        ce_weight=0.01, 
        local_dist={4:4},
        depth_interpolation_mode=depth_interpolation_mode,
        alpha = {4:0.15, 8:0.15},
        c = 1e-4,
        epe_mask_prob_th = 0.001,
        )

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint,  map_location=model.device)['model'])
        print(f"Load checkpoint {args.checkpoint}")
    if args.train_fine_matcher:
        model.train_fine_matcher()
        parameters = [
        {"params": [p for p in model.parameters() if p.requires_grad ], "lr": romatch.STEP_SIZE * 1e-4 / 8},
    ]
    else:
        parameters = [
            {"params": model.parameters(), "lr": romatch.STEP_SIZE * 1e-4 / 8},
        ]

    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[(9*N/romatch.STEP_SIZE)//10])
    #megadense_benchmark = MegadepthDenseBenchmark("data/megadepth", num_samples = 1000, h=h,w=w)
    mega1500_benchmark = Mega1500PoseLibBenchmark("data/megadepth", num_ransac_iter = 1, test_every = 30)

    checkpointer = CheckPoint(checkpoint_dir, experiment_name)
    model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler, global_step)

    if args.resume:
        model, optimizer, lr_scheduler, global_step = checkpointer.resume(args.resume, model, optimizer, lr_scheduler, global_step)
        
    romatch.GLOBAL_STEP = global_step
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters = True, gradient_as_bucket_view=True)
    grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1_000_000)
    grad_clip_norm = 0.01
    #megadense_benchmark.benchmark(model)

    seed=1234
    for n in range(romatch.GLOBAL_STEP, N, k * romatch.STEP_SIZE):
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples = batch_size * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size = batch_size,
                sampler = mega_sampler,
                num_workers = 8,
                worker_init_fn=partial(
                    worker_init_fn, num_workers=8, rank=rank,
                    seed=seed+n) 
            )
        )
        train_k_steps(
            n, k, mega_dataloader, ddp_model, depth_loss, optimizer, lr_scheduler, grad_scaler, grad_clip_norm = grad_clip_norm,
        )
        checkpointer.save(ddp_model, optimizer, lr_scheduler, romatch.GLOBAL_STEP)
        # wandb.log(mega1500_benchmark.benchmark(model, model_name=experiment_name), step = romatch.GLOBAL_STEP)

def test_mega_8_scenes(model, name):
    mega_8_scenes_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth",
                                                scene_names=['mega_8_scenes_0019_0.1_0.3.npz',
                                                    'mega_8_scenes_0025_0.1_0.3.npz',
                                                    'mega_8_scenes_0021_0.1_0.3.npz',
                                                    'mega_8_scenes_0008_0.1_0.3.npz',
                                                    'mega_8_scenes_0032_0.1_0.3.npz',
                                                    'mega_8_scenes_1589_0.1_0.3.npz',
                                                    'mega_8_scenes_0063_0.1_0.3.npz',
                                                    'mega_8_scenes_0024_0.1_0.3.npz',
                                                    'mega_8_scenes_0019_0.3_0.5.npz',
                                                    'mega_8_scenes_0025_0.3_0.5.npz',
                                                    'mega_8_scenes_0021_0.3_0.5.npz',
                                                    'mega_8_scenes_0008_0.3_0.5.npz',
                                                    'mega_8_scenes_0032_0.3_0.5.npz',
                                                    'mega_8_scenes_1589_0.3_0.5.npz',
                                                    'mega_8_scenes_0063_0.3_0.5.npz',
                                                    'mega_8_scenes_0024_0.3_0.5.npz'])
    mega_8_scenes_results = mega_8_scenes_benchmark.benchmark(model, model_name=name)
    print(mega_8_scenes_results)
    json.dump(mega_8_scenes_results, open(f"results/mega_8_scenes_{name}.json", "w"))

def test_mega1500(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega1500_{name}.json", "w"))

def test_mega1500_poselib(model, name):
    mega1500_benchmark = Mega1500PoseLibBenchmark("data/megadepth", num_ransac_iter = 1, test_every = 1)
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega1500_poselib_{name}.json", "w"))

def test_mega_8_scenes_poselib(model, name):
    mega1500_benchmark = Mega1500PoseLibBenchmark("data/megadepth", num_ransac_iter = 1, test_every = 1,
                                                  scene_names=['mega_8_scenes_0019_0.1_0.3.npz',
                                                    'mega_8_scenes_0025_0.1_0.3.npz',
                                                    'mega_8_scenes_0021_0.1_0.3.npz',
                                                    'mega_8_scenes_0008_0.1_0.3.npz',
                                                    'mega_8_scenes_0032_0.1_0.3.npz',
                                                    'mega_8_scenes_1589_0.1_0.3.npz',
                                                    'mega_8_scenes_0063_0.1_0.3.npz',
                                                    'mega_8_scenes_0024_0.1_0.3.npz',
                                                    'mega_8_scenes_0019_0.3_0.5.npz',
                                                    'mega_8_scenes_0025_0.3_0.5.npz',
                                                    'mega_8_scenes_0021_0.3_0.5.npz',
                                                    'mega_8_scenes_0008_0.3_0.5.npz',
                                                    'mega_8_scenes_0032_0.3_0.5.npz',
                                                    'mega_8_scenes_1589_0.3_0.5.npz',
                                                    'mega_8_scenes_0063_0.3_0.5.npz',
                                                    'mega_8_scenes_0024_0.3_0.5.npz'])
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega_8_scenes_poselib_{name}.json", "w"))

# def test_scannet_poselib(model, name):
#     scannet_benchmark = ScanNetPoselibBenchmark("data/scannet")
#     scannet_results = scannet_benchmark.benchmark(model)
#     json.dump(scannet_results, open(f"results/scannet_{name}.json", "w"))

def test_scannet(model, name):
    scannet_benchmark = ScanNetBenchmark("data/scannet")
    scannet_results = scannet_benchmark.benchmark(model)
    json.dump(scannet_results, open(f"results/scannet_{name}.json", "w"))

if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    import romatch
    parser = ArgumentParser()
    parser.add_argument("--only_test", action='store_true')
    parser.add_argument("--debug_mode", action='store_true')
    parser.add_argument("--dont_log_wandb", action='store_true')
    parser.add_argument("--train_resolution", default='medium')
    parser.add_argument("--gpu_batch_size", default=8, type=int)
    parser.add_argument("--wandb_entity", required = False)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--train_fine_matcher", action='store_true')

    args, _ = parser.parse_known_args()
    romatch.DEBUG_MODE = args.debug_mode
    if not args.only_test:
        train(args)

    # experiment_name = "tiny_roma_v1_outdoor"#Path(__file__).stem
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = TinyRoma(freeze_xfeat=False, exact_softmax=False).to(device)
    # model.load_state_dict(torch.load(f"{experiment_name}.pth"))
    # test_mega1500_poselib(model, experiment_name)
    