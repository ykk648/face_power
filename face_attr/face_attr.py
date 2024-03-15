# -- coding: utf-8 --
# @Time : 2022/1/7
# @Author : ykk648

import numpy as np
from cv2box import CVImage
from apstone import ModelBase

MODEL_ZOO = {
    'face_attr_mbnv3': {
        'model_path': 'pretrain_models/face_lib/face_attr/face_attr_epoch_12_220318.onnx',
        'input_dynamic_shape': (1, 3, 512, 512),
    },
}


class FaceAttr(ModelBase):
    def __init__(self, model_name='face_attr_mbnv3', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)

    def forward(self, image_p_):
        blob = CVImage(image_p_).blob_innormal(512, input_mean=[132.38155592, 110.99284567, 102.62942472],
                                               input_std=[68.5106407, 61.65929394, 58.61700102])
        result = self.model.forward(blob, trans=False)[0]
        return np.around(result, 3)

    @staticmethod
    def show_label():
        print('female male front side clean occlusion super_hq hq blur nonhuman')
