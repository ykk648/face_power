# -- coding: utf-8 --
# @Time : 2021/11/10
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from cv2box import CVImage, MyFpsCounter
from cv2box.utils.math import Normalize
from apstone import ModelBase


def down_sample(target_, size):
    import torch.nn.functional as F
    return F.interpolate(target_, size=size, mode='bilinear', align_corners=True)


MODEL_ZOO = {
    # https://github.com/HuangYG123/CurricularFace
    'CurricularFace': {
        'model_path': 'pretrain_models/face_lib/face_embedding/CurricularFace.onnx'
    },
    # https://github.com/deepinsight/insightface/tree/master/model_zoo
    # https://github.com/SthPhoenix/InsightFace-REST
    'insightface_mbf': {
        'model_path': 'pretrain_models/face_lib/face_embedding/w600k_mbf.onnx',
    },
    'insightface_r50': {
        'model_path': 'pretrain_models/face_lib/face_embedding/w600k_r50.onnx',
    },
}


class FaceEmbedding(ModelBase):
    def __init__(self, model_type='arcface', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        self.input_std = self.input_mean = 127.5
        self.input_size = (112, 112)

    def forward(self, face_image):
        """
        Args:
            face_image: BGR, 0-255, CVImage
        Returns: (512,), 0-1
        """
        face = CVImage(face_image).blob(self.input_size, self.input_mean, self.input_std, rgb=True)
        # for batch
        # return Normalize(self.model.forward(face)[0]).batch_norm()
        return Normalize(self.model.forward(face)[0].ravel()).np_norm()


