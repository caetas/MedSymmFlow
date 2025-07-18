cd src/medsymmflow

accelerate launch --mixed_precision=bf16 classification.py \
--train \
--dataset bloodmnist \
--snapshot 10 \
--n_epochs 1000 \
--lr 4e-4 \
--warmup 100 \
--decay 0.0 \
--sample_and_save_freq 20 \
--num_workers 16 \
--model_channels 128 \
--num_res_blocks 2 \
--channel_mult 1 2 2 2 \
--num_heads 4 \
--num_head_channels 64 \
--attention_resolutions 2 \
--batch_size 64 \
--solver_lib torchdiffeq \
--solver euler \
--step_size 0.04 \
--beta 4 \
--rgb_mask \
--ema_rate 0.7 \
--image_weight 0.7 \
--n_classes 8 \
--latent \
--size 256