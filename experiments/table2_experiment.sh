conda activate vgdfr
CUDA_VISIBLE_DEVICES=3 python experiments/table2_experiment.py --flow-reverse --compress_ratio 0.5 &
CUDA_VISIBLE_DEVICES=4 python experiments/table2_experiment.py --flow-reverse --compress_ratio 0.6 &
CUDA_VISIBLE_DEVICES=5 python experiments/table2_experiment.py --flow-reverse --compress_ratio 0.7 &
CUDA_VISIBLE_DEVICES=6 python experiments/table2_experiment.py --flow-reverse --compress_ratio 0.8 &
CUDA_VISIBLE_DEVICES=7 python experiments/table2_experiment.py --flow-reverse --compress_ratio 0.9