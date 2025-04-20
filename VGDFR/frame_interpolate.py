import torch
from torch.nn import functional as F
from VGDFR.RIFE_model import Model


class RIFEInterpolator:
    def __init__(self, rife_model_path="data"):
        self.model = Model()
        self.model.load_model(rife_model_path, -1)
        self.model.eval()

    def interpolate(self, img0, img1, n_interpolate=1):
        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        # Perform interpolation
        frame_list = []
        with torch.no_grad():
            for i in range(n_interpolate):
                # Interpolate between img0 and img1
                ratio = (i + 1) / (n_interpolate + 1)
                # print("interpolate ratio", ratio)
                new_frame = self.model.inference(img0, img1, ratio)
                frame_list.append(new_frame)
        new_frames = torch.stack(frame_list, dim=2)
        # Remove padding
        new_frames = new_frames[:, :, :, :h, :w]
        return new_frames

    def vgdfr_frame_interpolate(self, video, merge2_inds, merge4_inds, video_remain_inds, original_T):
        """
        video shape: (1, c, t, h, w)
        """
        assert video.shape[0] == 1, "video shape should be (1, c, t, h, w)"
        B, C, Tc, H, W = video.shape
        new_video = torch.zeros((B, C, original_T, H, W)).to(video.device)
        new_video[:, :, video_remain_inds] = video
        self.model.flownet.to(video.device)
        # merge 8 inds
        for ind in merge2_inds:
            start_frame = new_video[:, :, ind]
            end_frame = new_video[:, :, ind + 2]
            new_frames = self.interpolate(start_frame, end_frame, 1)
            new_video[:, :, ind + 1 : ind + 2] = new_frames
        # merge 16 inds
        for ind in merge4_inds:
            start_frame = new_video[:, :, ind]
            end_frame = new_video[:, :, ind + 4]
            new_frames = self.interpolate(start_frame, end_frame, 3)
            new_video[:, :, ind + 1 : ind + 4] = new_frames
        return new_video
