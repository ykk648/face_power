# -- coding: utf-8 --
# @Time : 2022/8/25
# @Author : ykk648

"""
Ref https://github.com/hanson-young/nniefacelib/tree/master/PFPLD/models/onnx
"""
import cv2
import numpy as np
from cv2box import CVImage
from apstone import ModelBase

from .utils import convert98to68

MODEL_ZOO = {
    'pfpld': {
        'model_path': 'pretrain_models/face_lib/face_landmark/pfpld.onnx',
        'model_input_size': (112, 112), },
}


class PFPLD(ModelBase):
    def __init__(self, model_name='pfpld', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)
        self.image_in = None
        self.results = None

    def forward(self, image_in):
        """
        Args:
            image_in: CVImage acceptable class
        Returns: (1,68,2)
        """
        self.image_in = CVImage(image_in).bgr
        input_image_shape = self.image_in.shape
        face_image = CVImage(self.image_in).resize((112, 112)).bgr
        face_image = (face_image / 255).astype(np.float32)
        pred = self.model.forward(face_image, trans=True)
        pred = convert98to68(pred[1])
        self.results = pred.reshape(-1, 68, 2) * input_image_shape[:2][::-1]
        return self.results

    def draw_face(self):
        draw_image = self.image_in.copy()
        for i in range(len(self.results[0])):
            kp = self.results[0][i].astype(int)
            cv2.circle(draw_image, tuple(kp), 1, (0, 0, 255), 1)
        CVImage(draw_image).show()
        return draw_image
