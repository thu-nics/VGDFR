# conda activate vgdfr
# height=960
# width=540
# length=65
# CUDA_VISIBLE_DEVICES=3 python experiments/ablation.py --exp_name compress_denoise_step2 --compress_denoise_steps 2 --flow-reverse --keep_token_ratio 0.7 --video-size ${height} ${width} --video-length ${length}

# # longer video
# conda activate vgdfr
# height=960
# width=540
# length=129
# CUDA_VISIBLE_DEVICES=0 python experiments/ablation.py --exp_name f129 --compress_denoise_steps 1 --flow-reverse --keep_token_ratio 0.7 --video-size ${height} ${width} --video-length ${length}

# larger video
conda activate vgdfr
height=1280
width=720
length=65
CUDA_VISIBLE_DEVICES=3 python experiments/ablation.py --exp_name 720p --compress_denoise_steps 1 --flow-reverse --keep_token_ratio 0.7 --video-size ${height} ${width} --video-length ${length}