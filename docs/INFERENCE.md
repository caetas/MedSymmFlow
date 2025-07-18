# Inference

## Sampling

You can sample from a pretrained model using:

    python classification.py \
    --sample \
    --num_samples 4 \ 
    --dataset bloodmnist \
    --model_channels 128 \
    --num_res_blocks 2 \
    --channel_mult 1 2 2 2 \
    --num_heads 4 \
    --num_head_channels 64 \
    --attention_resolutions 2 \
    --solver_lib torchdiffeq \
    --solver euler \
    --step_size 0.04 \
    --beta 4 \
    --rgb_mask \
    --n_classes 8 \
    --latent \
    --size 256 \
    --checkpoint ../../models/SymmetricalFlowMatchingClass/RGB_224/LatFM_bloodmnist_beta4.0_rgb.pt

## Classification

If you want to classify images from the test set, you can use:

    python classification.py \
    --classification \
    --batch_size 16 \ 
    --dataset bloodmnist \
    --model_channels 128 \
    --num_res_blocks 2 \
    --channel_mult 1 2 2 2 \
    --num_heads 4 \
    --num_head_channels 64 \
    --attention_resolutions 2 \
    --solver_lib torchdiffeq \
    --solver euler \
    --step_size 0.04 \
    --beta 4 \
    --rgb_mask \
    --n_classes 8 \
    --latent \
    --size 256 \
    --checkpoint ../../models/SymmetricalFlowMatchingClass/RGB_224/LatFM_bloodmnist_beta4.0_rgb.pt

