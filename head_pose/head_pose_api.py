# -- coding: utf-8 --
# @Time : 2024/1/15
# @Author : ykk648

"""
https://github.com/Ascend-Research/HeadPoseEstimation-WHENet
"""
import numpy as np
from cv2box import CVImage
from apstone import ModelBase

"""
input_name:['input_1'], 
shape:[[1, 3, 224, 224]], 
type:['tensor(float)']
output_name:['tf.identity', 'tf.identity_1', 'tf.identity_2'], 
shape:[[1, 120], [1, 66], [1, 66]], 
type:['tensor(float)', 'tensor(float)', 'tensor(float)']
"""
MODEL_ZOO = {
    'WHENet': {
        'model_path': 'pretrain_models/face_lib/head_pose/WHENet.onnx'
    },
}


class HeadPose(ModelBase):
    def __init__(self, model_name='WHENet', provider='gpu'):
        super().__init__(model_info=MODEL_ZOO[model_name], provider=provider)
        self.parsing_results = None
        self.face_image = None
        self.input_size = 224
        self.input_mean = (0.485, 0.456, 0.406)
        self.input_std = (0.229, 0.224, 0.225)

        self.idx_tensor_yaw = [np.array(idx, dtype=np.float32) for idx in range(120)]
        self.idx_tensor = [np.array(idx, dtype=np.float32) for idx in range(66)]

    def softmax(self, x):
        x -= np.max(x, axis=1, keepdims=True)
        a = np.exp(x)
        b = np.sum(np.exp(x), axis=1, keepdims=True)
        return a / b

    def forward(self, head_image):
        """
        Args:
            head_image: cv2 0-255 (3,h,w)
        Returns: (512,512)
        """
        head_image_in = CVImage(head_image).blob_innormal(self.input_size, self.input_mean, self.input_std,
                                                          rgb=True)
        # head_image_in = head_image_in[None, ...].astype(np.float32)
        yaw, roll, pitch = self.model.forward(head_image_in.astype(np.float32))

        yaw = np.sum(self.softmax(yaw) * self.idx_tensor_yaw, axis=1) * 3 - 180
        pitch = np.sum(self.softmax(pitch) * self.idx_tensor, axis=1) * 3 - 99
        roll = np.sum(self.softmax(roll) * self.idx_tensor, axis=1) * 3 - 99
        yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
        return pitch, roll, yaw
