import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log

from pathlib import Path

def load_image_as_tensor(image_path):
    # OpenCV로 이미지 읽기
    image = cv2.imread(image_path)
    # BGR에서 RGB로 변환 (OpenCV는 이미지를 BGR 형식으로 읽습니다)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 이미지 정규화 (0-255 범위를 0.0-1.0 범위로 변환)
    image = image / 255.0
    # NumPy 배열을 PyTorch 텐서로 변환 (HWC에서 CHW로 변환)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    return image_tensor

def eval_rendering(
    gt_dir,
    img_dir,
    # frames,
    # gaussians,
    # dataset,
    # save_dir,
    # pipe,
    # background,
    # kf_indices,
    iteration="final",
):
    # interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    # end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    gt_images = sorted(list(Path(gt_dir).glob("*.png")))
    pred_images = sorted(list(Path(img_dir).glob("*.png")))

    # print(gt_images[:10])
    # print(pred_images[:10])
    for idx, (gt_image, pred_image) in enumerate(zip(gt_images, pred_images)):

        rendering = load_image_as_tensor(str(pred_image))
        image = torch.clamp(rendering, 0.0, 1.0).to("cuda:0")
        gt_image = load_image_as_tensor(str(gt_image)).to("cuda:0")

        gt = (gt_image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )

        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    # psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    # mkdir_p(psnr_save_dir)

    # json.dump(
    #     output,
    #     open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
    #     indent=4,
    # )
    return output

if __name__ == "__main__":
    gt_image = "/home/jungyu/MonoGS/image/final_gt_image"
    image = "/home/jungyu/MonoGS/image/final_rendered_image"

    output = eval_rendering(gt_dir = gt_image, img_dir = image, )

    print("psnr : " , output["mean_psnr"])
    print("ssim : " , output["mean_ssim"])
    print("lpips : " , output["mean_lpips"])