torchrun \
--nnodes=1 \
--nproc_per_node=8 \
sample_ddp.py \
ODE \
--model SiT-S/2 \
--num-fid-samples 50000 \
--cfg-scale 4.0 \
--ckpt /home/jiayou.zhang/hom/personal/jinyuan/SiT/results/006-SiT-S-2-rot-head-Linear-velocity-None/checkpoints/0250000.pt