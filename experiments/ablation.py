import os, sys

sys.path.append(".")
import torch
from hyvideo.utils.file_utils import save_videos_grid
from pathlib import Path
from datetime import datetime
import time
from hyvideo.config import *
from VGDFR.hunyuan_vgdfr import VGDFRHunyuanVideoSampler, HunyuanVideoSampler


parser = argparse.ArgumentParser(description="HunyuanVideo inference script")

parser = add_network_args(parser)
parser = add_extra_models_args(parser)
parser = add_denoise_schedule_args(parser)
parser = add_inference_args(parser)
parser = add_parallel_args(parser)
parser.add_argument("--keep_token_ratio", type=float, default=0.7, help="Token Compress Ratio for VGDFR")
parser.add_argument("--prompts_txt_path", type=str, default="data/prompts_ablation.txt")
parser.add_argument("--compress_denoise_steps", type=int, default=1)
parser.add_argument("--before_compression_steps", type=int, default=10)
parser.add_argument("--ablation_random_fusion", action="store_true", help="Use random fusion")
parser.add_argument("--disable_dyrope", action="store_true", help="Disable DyRoPE")
parser.add_argument("--exp_name", type=str, default="")

args = parser.parse_args()
print(args)

models_root_path = Path(args.model_base)
if args.keep_token_ratio < 1.0:
    hunyuan_video_sampler = VGDFRHunyuanVideoSampler.from_pretrained(models_root_path, args=args)
else:
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

all_prompts = []
with open(args.prompts_txt_path) as f:
    all_prompts.extend(f.readlines())
all_prompts = [prompt.strip() for prompt in all_prompts]
print(all_prompts)
print(len(all_prompts))

seed = 3
args.infer_steps = args.infer_steps
width, height = args.video_size
video_length = args.video_length

for prompt in all_prompts:
    hunyuan_video_sampler.pipeline.keep_token_ratio = args.keep_token_ratio
    hunyuan_video_sampler.pipeline.before_compression_steps = args.before_compression_steps
    hunyuan_video_sampler.pipeline.compress_denoise_steps = args.compress_denoise_steps
    if args.ablation_random_fusion:
        hunyuan_video_sampler.pipeline.ablation_random_fusion = True
    if args.disable_dyrope:
        hunyuan_video_sampler.pipeline.global_freq_layer_ids = []

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
            file_name = f"seed{seed}_{prompt[:100].replace('/','')}"
            save_folder = f"{save_path}/vgdfr_{args.exp_name}/cr_{args.keep_token_ratio}_k{args.before_compression_steps}"
            raw_save_path = f"{save_folder}/{file_name}.mp4"
            save_videos_grid(sample, raw_save_path, fps=12)
            torch.save(
                sample,
                f"{save_folder}/{file_name}.pt",
            )
            denoise_latency = hunyuan_video_sampler.pipeline.denoise_latency
            with open(f"{save_folder}/latency.txt", "a+") as f:
                f.write(f"{denoise_latency}\n")

            torch.save(
                hunyuan_video_sampler.pipeline.compress_similarity_info,
                f"{save_folder}/compress_similarity_info_{file_name}.pt",
            )
