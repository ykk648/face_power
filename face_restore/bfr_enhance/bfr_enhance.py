"""
from https://github.com/harlanhong/ACTalker/blob/master/src/utils/enhance/model_enhance.py
"""
import cv2
import numpy as np
import os
import torch
from cv2box import CVImage
from enhance import model_enhance


class test_pipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def init_model(self, enhance_pth_path, device):
        self.device = device
        self.G = model_enhance.get_enhance_model(pth_path=enhance_pth_path, is_training=False)
        self.G.to(device)

        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = torch.from_numpy(self.mask).view(1, 1, 512, 512).to(device)

    def enhance_cropface(self, cropface):
        # rgb [-1,1]
        if cropface.dtype == 'uint16':
            crop_face_tensor = torch.from_numpy(cv2.cvtColor(cropface, cv2.COLOR_BGR2RGB).astype('float32') / 32767.5 - 1).permute(2, 0, 1).to(self.device).unsqueeze(0)
        else:
            crop_face_tensor = torch.from_numpy(cv2.cvtColor(cropface, cv2.COLOR_BGR2RGB).astype('float32') / 127.5 - 1).permute(2, 0, 1).to(self.device).unsqueeze(0)

        crop_face_tensor_512 = torch.nn.functional.interpolate(crop_face_tensor, (512, 512), mode="bilinear")
        result_tensor, _ = self.G(crop_face_tensor_512)
        result_tensor = result_tensor.clamp(-1, 1)

        result_merge = result_tensor * self.mask + crop_face_tensor * (1 - self.mask)  # nchw -1,1 rgb

        if cropface.dtype == 'uint16':
            result_crop_bgr = cv2.cvtColor((result_merge[0] * 32767.5 + 32767.5).cpu().numpy().astype('uint16').transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        else:
            result_crop_bgr = cv2.cvtColor((result_merge[0] * 127.5 + 127.5).cpu().numpy().astype('uint8').transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

        # CVImage(result_crop_bgr).save("test.jpg")
        return result_crop_bgr


if __name__ == '__main__':
    os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"

    test_img = ""
    img_bgr = CVImage(test_img).rgb()

    tp = test_pipeline()
    tp.init_model("pretrain_models/face_lib/bfr_enhance/face_restoreYT-enhance-512.pth", "cuda")
    tp.enhance_cropface(img_bgr)
