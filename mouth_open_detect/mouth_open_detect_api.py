# -- coding: utf-8 --
# @Time : 2023/12/9
# @LastEdit : 2024/2/7
# @Author : ykk648
import cv2
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
        'model_path': 'private_models/face_lib/mouth_open_detect/openmouth_detect.onnx',
    },
}


class MouthOpen(ModelBase):
    def __init__(self, model_name='openmouth_detect', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)

    def forward(self, image_in_, landmark_68):
        """
        Args:
            image_in_: BGR
            landmark_68: from pfpld
        Returns:
        """
        image_in_ = CVImage(image_in_).bgr

        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(landmark_68.astype(np.int32)[49:])
        new_x = int(rect_x - 0.1 * rect_w)
        new_y = int(rect_y - 0.1 * rect_w)
        new_w = int(1.2 * rect_w)
        new_h = int(rect_h + 0.2 * rect_w)
        image_in_ = image_in_[new_y:new_y + new_h, new_x:new_x + new_w, ...]
        image_in_ = cv2.resize(image_in_, (80, 40)).astype(np.float32)

        image_in_ = (image_in_ - 127) / 128
        # CVImage(image_in_).show(0)
        image_in_ = image_in_.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        outputs = self.model.forward(image_in_)
        return 'open' if np.argmax(outputs[0][0]) == 1 else 'close'


