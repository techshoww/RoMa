export CUDA_VISIBLE_DEVICES="1"
# nohup torchrun --nproc_per_node=4 --nnodes=4 --rdzv_backend=c10d  experiments/train_ddp_tiny_roma_v1_outdoor.py --gpu_batch_size 1 > train.log &

# nohup python -m torch.distributed.launch --nproc_per_node=4 experiments/train_ddp_tiny_roma_v1_outdoor.py \
#                                         --gpu_batch_size 4 \
#                                         > train.log &


nohup torchrun --nproc_per_node=1 --nnodes=4 --rdzv_backend=c10d  experiments/train_ddp_tiny_roma_v1_outdoor.py --gpu_batch_size 10 > train2.log &