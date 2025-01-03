export CUDA_VISIBLE_DEVICES="1,2,6,7"
# nohup torchrun --nproc_per_node=1 --nnodes=1 --rdzv_backend=c10d  experiments/train_tiny_roma_v1_outdoor.py --gpu_batch_size 1 > train_bn.log &

# nohup python -m torch.distributed.launch --nproc_per_node=4 experiments/train_ddp_tiny_roma_v1_outdoor.py \
#                                         --gpu_batch_size 6 \
#                                         > train_tiny2_1.log &


nohup python -m torch.distributed.launch --nproc_per_node=4 experiments/train_ddp_tiny_roma_v1_outdoor.py \
                                        --gpu_batch_size 6 \
                                        > train_tiny2_2.log &
# nohup torchrun --nproc_per_node=1 --nnodes=4 --rdzv_backend=c10d  experiments/train_ddp_tiny_roma_v1_outdoor.py --gpu_batch_size 10 > train_tiny1.log &