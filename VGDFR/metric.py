from pytorch_msssim import ms_ssim
from pytorch_msssim import ssim as calc_ssim_func

# import lpips

# from torcheval.metrics import PeakSignalNoiseRatio

# loss_fn_alex = lpips.LPIPS(net='alex').cuda()


def calculate_videos_ssim(videos1, videos2):
    # print("calculate_ssim...")

    # videos [batch_size, timestamps, channel, h, w]

    assert videos1.shape == videos2.shape

    ssim_results = []

    for video_num in range(videos1.shape[0]):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num].mul(255).add_(0.5).clamp_(0, 255)
        video2 = videos2[video_num].mul(255).add_(0.5).clamp_(0, 255)

        value = calc_ssim_func(video1, video2)
        ssim_results.append(value.item())
    return ssim_results
