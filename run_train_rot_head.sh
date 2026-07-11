torchrun \
--nnodes=1 \
--nproc_per_node=8 \
train_rot_head.py \
--model SiT-S/2 \
--epochs=200 \
--data-path /home/jiayou.zhang/hom/personal/imagenet_dataset/images/train \
--wandb \
--global-batch-size=1024

# batch_size x 4, lr x 2
