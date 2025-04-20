# VGDFR: Diffuison-based Video Generation with Dynamic Frame Rate
This is the official implementation of the paper [VGDFR: Diffuison-based Video Generation with Dynamic Frame Rate](https://arxiv.org/abs/2504.12259).

<!-- ## Result Show -->

## Installation

```bash
# 1. Create conda environment
conda create -n vgdfr python==3.10.9

# 2. Activate the environment
conda activate vgdfr

# 3. Install PyTorch and other dependencies using conda
# For CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# For CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. Install pip dependencies
python -m pip install -r requirements.txt

# 5. Install flash attention for acceleration
conda install cuda-nvcc
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.4

# 6. Install xDiT for parallel inference (It is recommended to use torch 2.4.0 and flash-attn 2.6.3)
python -m pip install xfuser==0.4.3

# 7. Download flownet.pkl to data directory
wget https://github.com/thu-nics/VGDFR/releases/download/v0.1/flownet.pkl -P data/

# 8. Download the pre-trained HunyuanVideo model to ckpts directory (ref to https://github.com/Tencent/HunyuanVideo/tree/main/ckpts)
mkdir -p ckpts
cd ckpts
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
cd ../
python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ckpts/text_encoder

```

## Generation with VGDFR

There is a ipython notebook example `experiments/example.ipynb` in the root directory. You can run it in Jupyter Notebook.

The following is a simple example of how to use VGDFR to generate videos.
```python
from VGDFR.hunyuan_vgdfr import VGDFRHunyuanVideoSampler
hunyuan_video_sampler = VGDFRHunyuanVideoSampler.from_pretrained(models_root_path, args=args)
hunyuan_video_sampler.pipeline.schedule_mode = "compress_ratio"
hunyuan_video_sampler.pipeline.before_compression_steps = 5
hunyuan_video_sampler.pipeline.compress_ratio = 0.75
samples = hunyuan_video_sampler.predict(
      prompt=prompt,
      height=height,
      width=width,
      video_length=video_length,
      seed=seed,
      negative_prompt=args.neg_prompt,
      infer_steps=args.infer_steps,
      guidance_scale=args.cfg_scale,
      num_videos_per_prompt=args.num_videos,
      flow_shift=args.flow_shift,
      batch_size=args.batch_size,
      embedded_guidance_scale=args.embedded_cfg_scale,
)["samples"]
```

We provide two kinds of schedule methods: `compress_ratio` and `similarity_threshold`. The `compress_ratio` method is used to control the token compression ratio of generating, while the `similarity_threshold` method is used to set the minimum similarity threshold to compress adjacent frames.

The `before_compression_steps` parameter is used to set the number of steps before the compression starts.

## Ackowledgement and Citation

This code is built upon the following open-source projects:
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
- [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)

If you find this code useful in your research, please consider citing our paper:
```bibtex
@misc{yuan2025vgdfrdiffusionbasedvideogeneration,
      title={VGDFR: Diffusion-based Video Generation with Dynamic Latent Frame Rate}, 
      author={Zhihang Yuan and Rui Xie and Yuzhang Shang and Hanling Zhang and Siyuan Wang and Shengen Yan and Guohao Dai and Yu Wang},
      year={2025},
      eprint={2504.12259},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.12259}, 
}
```