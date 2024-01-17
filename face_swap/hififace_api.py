# -- coding: utf-8 --
# @Time : 2022/8/25
# @Author : ykk648

import numpy as np
from apstone import ModelBase

# Ref https://johann.wang/HifiFace/

MODEL_ZOO = {
    # input_name:['target', 'vsid'], shape:[[1, 3, 256, 256], [1, 512]]
    # output_name:['output', 'mask'], shape:[[1, 3, 256, 256], [1, 1, 256, 256]]
    '865': {
        'model_path': 'private_models/deep_fake/hififace/pretrain_models/9O_865k_sim_bs1.onnx',
        # 'input_dynamic_shape': [(1, 3, 256, 256), (1, 512)],
    },
    '865K_trt': {
        # trt version must be 8003
        # use self wrote trt wrapper, mmlab version raise error, no time to solve for now
        'model_path': 'private_models/deep_fake/hififace/pretrain_models/9O_865k_sim_bs1_8003_fp16_ws4096_io16.engine',
        'trt_wrapper_self': True,
    },
}


class HifiFace(ModelBase):
    def __init__(self, model_name='865K_dynamic', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)

    def forward(self, src_face_image, dst_face_latent):
        """
        Args:
            src_face_image: RGB
            dst_face_latent: [1, 512]
        Returns:
        """
        img_tensor = ((src_face_image.transpose(2, 0, 1) / 255.0) * 2 - 1)[None]
        blob = [img_tensor.astype(np.float32), dst_face_latent.astype(np.float32)]
        output = self.model.forward(blob)
        if self.model_type == 'trt':
            mask, swap_face = output
        else:
            swap_face, mask = output

        return mask, swap_face
