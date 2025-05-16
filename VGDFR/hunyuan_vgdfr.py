from hyvideo.diffusion.pipelines.pipeline_hunyuan_video import *
from hyvideo.inference import *
import torch.nn.functional as F
from tqdm import tqdm
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from hyvideo.vae.unet_causal_3d_blocks import UpsampleCausal3D, DownsampleCausal3D, CausalConv3d
import torch.nn as nn
from VGDFR.metric import calculate_videos_ssim
from pytorch_msssim import ssim as calc_ssim_func
from hyvideo.modules.models import HYVideoDiffusionTransformer, get_cu_seqlens
from VGDFR.frame_interpolate import RIFEInterpolator
from hyvideo.vae import load_vae
from VGDFR.frame_rate_schedule import similarity_threhshold_schedule, keep_token_ratio_schedule


def mod_rope_forward(
    self,
    x: torch.Tensor,
    t: torch.Tensor,  # Should be in range(0, 1000).
    text_states: torch.Tensor = None,
    text_mask: torch.Tensor = None,  # Now we don't use it.
    text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
    # freqs_cos: Optional[torch.Tensor] = None,
    # freqs_sin: Optional[torch.Tensor] = None,
    global_freqs_cis=None,
    local_freqs_cis=None,
    global_freq_layer_ids=[4, 19, 23, 31, 35, 36, 37, 40],
    guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
    return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    out = {}
    img = x
    txt = text_states
    _, _, ot, oh, ow = x.shape
    tt, th, tw = (
        ot // self.patch_size[0],
        oh // self.patch_size[1],
        ow // self.patch_size[2],
    )

    # Prepare modulation vectors.
    vec = self.time_in(t)

    # text modulation
    vec = vec + self.vector_in(text_states_2)

    # guidance modulation
    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")

        # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
        vec = vec + self.guidance_in(guidance)

    # Embed image and text.
    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    # Compute cu_squlens and max_seqlen for flash attention
    cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q

    # --------------------- Pass through DiT blocks ------------------------
    for _, block in enumerate(self.double_blocks):
        if global_freqs_cis is not None and _ in global_freq_layer_ids:
            freqs_cis = global_freqs_cis
        else:
            freqs_cis = local_freqs_cis
        double_block_args = [
            img,
            txt,
            vec,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            freqs_cis,
        ]

        img, txt = block(*double_block_args)

    # Merge txt and img to pass through single stream blocks.
    x = torch.cat((img, txt), 1)
    if len(self.single_blocks) > 0:
        for _, block in enumerate(self.single_blocks):
            if global_freqs_cis is not None and _ + len(self.double_blocks) in global_freq_layer_ids:
                freqs_cos, freqs_sin = global_freqs_cis
            else:
                freqs_cos, freqs_sin = local_freqs_cis

            single_block_args = [
                x,
                vec,
                txt_seq_len,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                (freqs_cos, freqs_sin),
            ]

            x = block(*single_block_args)

    img = x[:, :img_seq_len, ...]

    # ---------------------------- Final layer ------------------------------
    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

    img = self.unpatchify(img, tt, th, tw)
    if return_dict:
        out["x"] = img
        return out
    return img


class VGDFRHunyuanVideoSampler(HunyuanVideoSampler):
    def load_diffusion_pipeline(
        self,
        args,
        vae,
        text_encoder,
        text_encoder_2,
        model,
        scheduler=None,
        device=None,
        progress_bar_config=None,
        data_type="video",
    ):
        """Load the denoising scheduler for inference."""
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift,
                    reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")

        pipeline = VGDFRHunyuanVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            progress_bar_config=progress_bar_config,
            args=args,
        )
        if self.use_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        return pipeline


class VGDFRHunyuanVideoPipeline(HunyuanVideoPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HYVideoDiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        progress_bar_config: Dict[str, Any] = None,
        args=None,
        before_compression_steps: int = 5,
        schedule_mode: str = "keep_token_ratio",
        similarity_threshold=0.75,
        keep_token_ratio=0.75,
        compress_denoise_steps=1,
    ):
        super().__init__(vae, text_encoder, transformer, scheduler, text_encoder_2, progress_bar_config, args)
        self.frame_interpolator = RIFEInterpolator()
        # Init compression module
        self.compression_vae_module = self.init_compression_vae_module()
        self.compression_info = None
        self.before_compression_steps = before_compression_steps
        self.similarity_threshold = similarity_threshold
        self.keep_token_ratio = keep_token_ratio
        self.denoise_latency = None
        self.schedule_mode = schedule_mode
        self.compress_denoise_steps=compress_denoise_steps
        self.ablation_random_fusion = False # Only for ablation study, please do not use it
        self.global_freq_layer_ids = [4, 19, 23, 31, 35, 36, 37, 40]
        self.compress_similarity_info=None

    def init_compression_vae_module(self):
        compression_vae_module, _, s_ratio, t_ratio = load_vae(
            vae_type="884-16c-hy",
            vae_precision="fp16",
            logger=logger,
            vae_path="ckpts/hunyuan-video-t2v-720p/vae",
            device="cuda",
        )
        compression_vae_module = compression_vae_module.eval()
        compression_vae_module.enable_tiling()
        for name, module in compression_vae_module.decoder.named_modules():
            if isinstance(module, UpsampleCausal3D):
                # if module.upsample_factor[1]>1 and "_blocks.2" not in name:
                if module.upsample_factor[1] > 1 and "_blocks.0" in name:
                    module._raw_upsample_factor = module.upsample_factor
                    module.upsample_factor = (module.upsample_factor[0], 1, 1)
                    print(f"UpsampleCausal3D {name}: convert {module._raw_upsample_factor} to {module.upsample_factor}")

        for name, module in compression_vae_module.encoder.named_modules():
            if isinstance(module, CausalConv3d):
                module = module.conv
                # if module.stride[1]==2 and "_blocks.0" not in name:
                if module.stride[1] == 2 and "_blocks.2" in name:
                    module._raw_stride = module.stride
                    module.stride = (module.stride[0], 1, 1)
                    print(f"Downsample Conv3d {name}: convert {module._raw_stride} to {module.stride}")
        # print(compression_module.tile_latent_min_size)
        compression_vae_module.tile_latent_min_size = 64
        return compression_vae_module

    def run_compression_module(self, latents_without_noise, freqs_cis, latent_noise):
        """
        VGDFR Compression Module
        """
        

        print(f"Run VGDFR Compression Module...")
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (vae_dtype != torch.float32) and not self.args.disable_autocast
        T = latents_without_noise.shape[2]
        sigma = self.scheduler.sigmas[self.scheduler.step_index]

        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents_without_noise = (
                latents_without_noise / self.vae.config.scaling_factor + self.vae.config.shift_factor
            )
        else:
            latents_without_noise = latents_without_noise / self.vae.config.scaling_factor
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):

            # Compression Decode
            self.compression_vae_module.enable_tiling()
            image = self.compression_vae_module.decode(latents_without_noise, return_dict=False, generator=None)[0]

            # Dynamic Frame Rate Schedule
            image_reshape = (image[0].transpose(0, 1) / 2 + 0.5).clamp(0, 1)
            print(image_reshape.max(), image_reshape.min())

            ssim_results3 = calc_ssim_func(image_reshape[:-3], image_reshape[3:], size_average=False, data_range=1.0)
            ssim_results2 = calc_ssim_func(image_reshape[:-2], image_reshape[2:], size_average=False, data_range=1.0)
            ssim_results1 = calc_ssim_func(image_reshape[:-1], image_reshape[1:], size_average=False, data_range=1.0)

            self.compress_similarity_info={
                "ssim_results1": ssim_results1,
                "ssim_results2": ssim_results2,
                "ssim_results3": ssim_results3,
            }

            if self.ablation_random_fusion:
                print("\n\n======== WARNING: use ablation_random_fusion ========\n")
                ssim_results1=torch.rand_like(ssim_results1)
                ssim_results2=torch.rand_like(ssim_results2)
                ssim_results3=torch.rand_like(ssim_results3)

            # print(f"ssim_results1={ssim_results1}\nssim_results2={ssim_results2}\nssim_results3={ssim_results3}")

            if self.schedule_mode == "similarity_threshold":
                print(
                    f"Using similarity threshold schedule with similarity_threshold={self.similarity_threshold} and k={self.before_compression_steps}"
                )
                merge_plan, merge2x4_inds, merge4x4_inds, latent_remain_inds, video_remain_inds = (
                    similarity_threhshold_schedule(
                        image_reshape, ssim_results1, ssim_results2, ssim_results3, self.similarity_threshold
                    )
                )
            else:
                print(
                    f"Using compress ratio schedule with keep_token_ratio={self.keep_token_ratio} and k={self.before_compression_steps}"
                )
                merge_plan, merge2x4_inds, merge4x4_inds, latent_remain_inds, video_remain_inds = (
                    keep_token_ratio_schedule(
                        image_reshape, ssim_results1, ssim_results2, ssim_results3, self.keep_token_ratio
                    )
                )
            print(f"merge plan: \nmerge2x4_inds:{merge2x4_inds},\nmerge4x4_inds:{merge4x4_inds}")
            self.compression_info = {
                "merge2x4_inds": merge2x4_inds,
                "merge4x4_inds": merge4x4_inds,
                "video_remain_inds": video_remain_inds,
                "original_T": len(image_reshape),
            }

            # print(image.shape, merge_plan)
            for inds in merge_plan:
                image[:, :, inds[0]] = sum([image[:, :, _] for _ in inds]) / len(inds)
            image = image[:, :, video_remain_inds]

            # Compression Encode
            merged_latents = self.compression_vae_module.encode(image)[0].mode()

        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            merged_latents = (merged_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            merged_latents = merged_latents * self.vae.config.scaling_factor

        # Renoise
        renoised_latents = latent_noise[:, :, latent_remain_inds] * sigma + (1 - sigma) * merged_latents

        # Generate DyRoPE
        _, _, To, H, W = merged_latents.shape
        n_tokens = To * H // 2 * W // 2
        local_freqs_cis = (freqs_cis[0][:n_tokens], freqs_cis[1][:n_tokens])
        global_freqs_cis = (
            freqs_cis[0].view(T, H // 2, W // 2, -1)[latent_remain_inds].flatten(0, 2),
            freqs_cis[1].view(T, H // 2, W // 2, -1)[latent_remain_inds].flatten(0, 2),
        )
        return renoised_latents, local_freqs_cis, global_freqs_cis

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            video_length,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            vae_ver=vae_ver,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = torch.device(f"cuda:{dist.get_rank()}") if dist.is_initialized() else self._execution_device

        # 3. Encode input prompt
        lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
            data_type=data_type,
        )
        if self.text_encoder_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type=data_type,
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            **extra_set_timesteps_kwargs,
        )

        if "884" in vae_ver:
            video_length = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            video_length = (video_length - 1) // 8 + 1
        else:
            video_length = video_length

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            video_length,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": eta},
        )

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (target_dtype != torch.float32) and not self.args.disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (vae_dtype != torch.float32) and not self.args.disable_autocast

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # 7.1 Before compression, run `before_compression_steps` denoise steps
        assert self.before_compression_steps < num_inference_steps
        latent_noise = latents
        torch.cuda.synchronize()
        denoise_st = time.time()
        for i, t in enumerate(timesteps):
            if i >= self.before_compression_steps:
                break
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.contiguous()

            # t_expand = t.repeat(latent_model_input.shape[0])
            t_expand = t.view(1)
            guidance_expand = (
                torch.tensor(
                    [embedded_guidance_scale] * latent_model_input.shape[0],
                    dtype=torch.float32,
                    device=device,
                ).to(target_dtype)
                * 1000.0
                if embedded_guidance_scale is not None
                else None
            )

            # predict the noise residual
            with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                noise_pred = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                    latent_model_input,  # [2, 16, 33, 24, 42]
                    t_expand,  # [2]
                    text_states=prompt_embeds,  # [2, 256, 4096]
                    text_mask=prompt_mask,  # [2, 256]
                    text_states_2=prompt_embeds_2,  # [2, 768]
                    freqs_cos=freqs_cis[0],  # [seqlen, head_dim]
                    freqs_sin=freqs_cis[1],  # [seqlen, head_dim]
                    guidance=guidance_expand,
                    return_dict=True,
                )["x"]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred,
                    noise_pred_text,
                    guidance_rescale=self.guidance_rescale,
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        
        # 7.2 Run VGDFR Compression Module
        assert self.compress_denoise_steps>0
        if self.compress_denoise_steps==1:
            # One-step Denoise
            sigma = self.scheduler.sigmas[self.scheduler.step_index]
            latents_without_noise = latents - noise_pred * sigma
        if self.compress_denoise_steps>1:
            # Steps > 1, run multiple denoise steps for compression module
            dt = self.scheduler.sigmas[self.scheduler.step_index]/self.compress_denoise_steps
            latents_with_noise=latents
            latents_with_noise = latents_with_noise - noise_pred * dt
            left_steps=len(timesteps)-self.before_compression_steps
            compression_denoise_timesteps=[]
            for i in range(1,self.compress_denoise_steps):
                ind=round(self.before_compression_steps+left_steps/self.compress_denoise_steps*i)
                compression_denoise_timesteps.append(timesteps[ind])

            for i, t in enumerate(compression_denoise_timesteps):
                latent_model_input = torch.cat([latents_with_noise] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.contiguous()

                # t_expand = t.repeat(latent_model_input.shape[0])
                t_expand = t.view(1)
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                # predict the noise residual
                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    noise_pred = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                        latent_model_input,  # [2, 16, 33, 24, 42]
                        t_expand,  # [2]
                        text_states=prompt_embeds,  # [2, 256, 4096]
                        text_mask=prompt_mask,  # [2, 256]
                        text_states_2=prompt_embeds_2,  # [2, 768]
                        freqs_cos=freqs_cis[0],  # [seqlen, head_dim]
                        freqs_sin=freqs_cis[1],  # [seqlen, head_dim]
                        guidance=guidance_expand,
                        return_dict=True,
                    )["x"]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_with_noise = latents_with_noise - noise_pred * dt
            latents_without_noise=latents_with_noise


        latents, local_freqs_cis, global_freqs_cis = self.run_compression_module(
            latents_without_noise, freqs_cis, latent_noise
        )

        # 7.3 After compression, run the left denoising steps
        with self.progress_bar(total=num_inference_steps - self.before_compression_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i < self.before_compression_steps:
                    continue
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                # predict the noise residual
                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    noise_pred = mod_rope_forward(
                        self.transformer,
                        latent_model_input,  # [2, 16, 33, 24, 42]
                        t_expand,  # [2]
                        text_states=prompt_embeds,  # [2, 256, 4096]
                        text_mask=prompt_mask,  # [2, 256]
                        text_states_2=prompt_embeds_2,  # [2, 768]
                        global_freqs_cis=global_freqs_cis,  # [seqlen, head_dim]
                        local_freqs_cis=local_freqs_cis,  # [seqlen, head_dim]
                        global_freq_layer_ids=self.global_freq_layer_ids,
                        guidance=guidance_expand,
                        return_dict=True,
                    )["x"]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        torch.cuda.synchronize()
        denoise_ed = time.time()
        self.denoise_latency = denoise_ed - denoise_st

        if not output_type == "latent":
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
                if enable_tiling:
                    self.vae.enable_tiling()
                    image = self.vae.decode(latents, return_dict=False, generator=generator)[0]
                else:
                    image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)

        else:
            image = latents

        image = (image / 2 + 0.5).clamp(0, 1).float()

        # Do Frame Interpolate
        image = self.frame_interpolator.vgdfr_frame_interpolate(
            image,
            self.compression_info["merge2x4_inds"],
            self.compression_info["merge4x4_inds"],
            self.compression_info["video_remain_inds"],
            self.compression_info["original_T"],
        )

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image)
