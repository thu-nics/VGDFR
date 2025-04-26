conda activate vgdfr
height=960
width=540
length=65
CUDA_VISIBLE_DEVICES=0 python experiments/compression_grid.py --flow-reverse --keep_token_ratio 1.0 --video-size ${height} ${width} --video-length ${length} &
CUDA_VISIBLE_DEVICES=1 python experiments/compression_grid.py --flow-reverse --keep_token_ratio 0.6 --video-size ${height} ${width} --video-length ${length} &
CUDA_VISIBLE_DEVICES=2 python experiments/compression_grid.py --flow-reverse --keep_token_ratio 0.7 --video-size ${height} ${width} --video-length ${length} &
CUDA_VISIBLE_DEVICES=3 python experiments/compression_grid.py --flow-reverse --keep_token_ratio 0.8 --video-size ${height} ${width} --video-length ${length} &
CUDA_VISIBLE_DEVICES=4 python experiments/compression_grid.py --flow-reverse --keep_token_ratio 0.9 --video-size ${height} ${width} --video-length ${length} 