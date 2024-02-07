# -- coding: utf-8 --
# @Time : 2023/12/9
# @LastEdit : 2024/2/7
# @Author : ykk648

from apstone import ModelBase
from cv2box import CVImage
import numpy as np

"""
input_name:['input.1'], 
shape:[[1, 3, 40, 80]], 
type:['tensor(float)']
output_name:['250'], 
shape:[[1, 2]], 
type:['tensor(float)']
"""
MODEL_ZOO = {
    'openmouth_detect': {
        'model_path': 'private_models/face_lib/mouth_open_detect/openmouth_detect_1213.onnx',
    },
}


class MouthOpen(ModelBase):
    def __init__(self, model_name='openmouth_detect', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)

    def forward(self, image_in_):
        """
        Args:
            image_in_: BGR
        Returns:
        """
        image_in_ = CVImage(image_in_).bgr
        image_in_ = (image_in_ - 127) / 128
        # CVImage(image_in_).show(0)
        image_in_ = image_in_.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        outputs = self.model.forward(image_in_)
        return outputs[0][0]

