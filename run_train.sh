torchrun \
--nnodes=1 \
--nproc_per_node=8 \
train.py \
--model SiT-S/2 \
--epochs=400 \
--data-path /home/jiayou.zhang/hom/personal/imagenet_dataset/images/train \
--wandb \
--global-batch-size=1024

# batch_size x 4, lr x 2
