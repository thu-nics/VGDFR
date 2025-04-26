# VGDFR: Diffuison-based Video Generation with Dynamic Frame Rate
This is the official implementation of the paper [VGDFR: Diffuison-based Video Generation with Dynamic Frame Rate](https://arxiv.org/abs/2504.12259).
We exploit the inherent temporal non-uniformity of real-world videos and observe that videos exhibit dynamic information density, with high-motion segments demanding greater detail preservation than static scenes. We propose VGDFR, a training-free approach for Diffusion-based Video Generation with Dynamic Latent Frame Rate. VGDFR adaptively adjusts the number of elements in latent space based on the motion frequency of the latent space content, using fewer tokens for low-frequency segments while preserving detail in high-frequency segments. Experiments show that VGDFR can achieve a speedup up to 3x for video generation.

## 🎥 Demo Videos

<table>
  <thead>
    <tr>
      <th></th>
      <th>A cute girl with red hair. Realistic, Natural lighting, Casual</th>
      <th>A fluffy dog with a joyful expression bounds through a snowy landscape under a soft blue sky. Snowflakes gently fall as the dog rolls, pounces into snowdrifts, and digs energetically. Occasionally, the dog pauses, wagging its tail and looking back at the camera, inviting you to play. The surroundings feature snow-covered trees, frosted bushes, and a serene winter backdrop. The video is lighthearted, with soft, playful background music enhancing the happy and lively atmosphere. Realistic, Natural lighting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Raw</td>
      <td><video src="https://github.com/user-attachments/assets/7d3ff150-37be-4f05-b6cc-46118f8acf03"></video></td>
      <td><video src="https://github.com/user-attachments/assets/42fcde74-b9be-41f8-ae91-fd4caf84bdcb"></video></td>
    </tr>
    <tr>
      <td>90% tokens</td>
      <td><video width=10 src="https://github.com/user-attachments/assets/76b2ac90-6d7c-4aeb-af8e-37aa61708adb"></video></td>
      <td><video width=10 src="https://github.com/user-attachments/assets/55c8f2e4-aa32-4f82-b6c6-6bf1c1630f37"></video></td>
    </tr>
    <tr>
      <td>80% tokens</td>
      <td><video src="https://github.com/user-attachments/assets/8d090b1a-c1a4-47fc-913c-d17d02295651"></video></td>
      <td><video src="https://github.com/user-attachments/assets/9469bb84-6dd5-45ca-a6a2-8231b500a22d"></video></td>
    </tr>
    <tr>
      <td>70% tokens</td>
      <td><video src="https://github.com/user-attachments/assets/bbfb69e3-7967-4c19-907d-db0dd4c7810a"></video></td>
      <td><video src="https://github.com/user-attachments/assets/20e563bc-b263-42e4-9f72-d207f3e4d083"></video></td>
    </tr>
    <tr>
      <td>60% tokens</td>
      <td><video src="https://github.com/user-attachments/assets/32d83493-f15a-48aa-8bfe-f48fa10ad9fa"></video></td>
      <td><video src="https://github.com/user-attachments/assets/376c3483-d63a-4ae7-878b-4ae135d0214c"></video></td>
    </tr>
  </tbody>
</table>

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
hunyuan_video_sampler.pipeline.schedule_mode = "keep_token_ratio"
hunyuan_video_sampler.pipeline.before_compression_steps = 5
hunyuan_video_sampler.pipeline.keep_token_ratio = 0.8
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

We provide two kinds of schedule methods: `keep_token_ratio` and `similarity_threshold`. The `keep_token_ratio` method is used to control the token compression ratio of generating, while the `similarity_threshold` method is used to set the minimum similarity threshold to compress adjacent frames.

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
