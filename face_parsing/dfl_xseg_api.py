# -- coding: utf-8 --
# @Time : 2022/8/12
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
based on https://github.com/iperov/DeepFaceLab/tree/master/models/Model_XSeg
base model come from https://www.dfldata.cc/forum.php, self trained (private data
"""
from apstone import ModelBase
from cv2box import CVImage
import numpy as np

MODEL_ZOO = {
    'xseg_net': {
        'model_path': 'pretrain_models/face_lib/face_parsing/230611_dfldata_16_17.onnx',
        'input_dynamic_shape': (1, 256, 256, 3),
    },
    'xseg_net_private': {
        'model_path': 'private_models/deep_fake/deepfacelab/xseg/xseg_211104_4790000.onnx',
        'input_dynamic_shape': (1, 256, 256, 3),
    }
}


class XsegNet(ModelBase):
    def __init__(self, model_name='xseg_net_private', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)

    def forward(self, image_in):
        """
        Args:
            image_in: CVImage acceptable class
        Returns:
        """
        image_in = CVImage(image_in)
        image_in_shape = image_in.bgr.shape[:2]
        image_in_ = image_in.resize(self.input_dynamic_shape[1:3]).bgr
        # CVImage(image_in_).show(0)
        image_in_ = image_in_[np.newaxis, :].astype(np.float32) / 255
        outputs = self.model.forward(image_in_)[0][0]
        outputs = CVImage(outputs).resize(image_in_shape).bgr
        return outputs



