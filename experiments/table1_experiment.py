import os, sys

sys.path.append(".")
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from hyvideo.vae import load_vae
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.inference import HunyuanVideoSampler
from pathlib import Path
from datetime import datetime
import time
from hyvideo.config import *
from VGDFR.hunyuan_vgdfr import VGDFRHunyuanVideoSampler


parser = argparse.ArgumentParser(description="HunyuanVideo inference script")

parser = add_network_args(parser)
parser = add_extra_models_args(parser)
parser = add_denoise_schedule_args(parser)
parser = add_inference_args(parser)
parser = add_parallel_args(parser)
parser.add_argument("--threshold", type=float, default=0.6, help="Threshold theta for VGDFR")

args = parser.parse_args()
print(args)

models_root_path = Path(args.model_base)
hunyuan_video_sampler = VGDFRHunyuanVideoSampler.from_pretrained(models_root_path, args=args)

prompt_path = "data/prompts_per_dimension_longer"
all_prompts = []
for file in os.listdir(prompt_path):
    with open(os.path.join(prompt_path, file)) as f:
        all_prompts.extend(f.readlines()[:2])
# print(all_prompts)
print(len(all_prompts))
with open("data/prompts.txt", "w") as f:
    f.writelines(all_prompts)

seed = 3
args.infer_steps = 50
width, height = 960, 544
video_length = 97

for prompt in all_prompts:
    hunyuan_video_sampler.pipeline.similarity_threshold = args.threshold
    for before_compression_steps in [5, 10, 15]:
        hunyuan_video_sampler.pipeline.before_compression_steps = before_compression_steps

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
        save_path = args.save_path
        # log_dlfr_t = hunyuan_video_sampler.pipeline.log_dlfr_t
        # Save samples
        if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                file_name = f"raw_seed{seed}_{prompt[:100].replace('/','')}"
                save_folder = f"{save_path}/vgdfr/th_{args.threshold}_k{before_compression_steps}"
                raw_save_path = f"{save_folder}/{file_name}.mp4"
                save_videos_grid(sample, raw_save_path, fps=12)
                torch.save(
                    sample,
                    f"{save_folder}/{file_name}.pt",
                )
                denoise_latency = hunyuan_video_sampler.pipeline.denoise_latency
                with open(f"{save_folder}/latency.txt", "a+") as f:
                    f.write(f"{denoise_latency}")
