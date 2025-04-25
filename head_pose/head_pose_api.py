# -- coding: utf-8 --
# @Time : 2024/1/15
# @LastEdit : 2025/4/25
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

    def extract_face_angle(self, frame, box):
        """
        Extracts the face angle (pitch, roll, yaw) from a given image frame and bounding box.

        Parameters:
        frame (numpy.ndarray): The input image frame.
        box (list): The bounding box coordinates [x1, y1, x2, y2] for the face.

        Returns:
        list: A list containing the pitch, roll, and yaw angles of the face.
            Returns [100, 100, 100] if the face region is invalid.
        """
        [y1, y2, x1, x2] = box
        h = y2 - y1
        w = x2 - x1

        # enlarge the bbox to include more background margin, similar to process_detection
        new_h1 = max(0, int(y1 - h * 0.1))  # expand 10% up
        new_h2 = min(int(y2 + h * 0.1), frame.shape[0])  # expand 10% down
        new_w1 = max(0, int(x1 - w * 0.2))  # expand 20% left
        new_w2 = min(int(x2 + w * 0.2), frame.shape[1])  # expand 20% right

        face_first = frame[new_h1:new_h2, new_w1:new_w2]
        if face_first.shape[0] == 0 or face_first.shape[1] == 0:
            return [100, 100, 100]
        else:
            pitch, roll, yaw = self.forward(face_first)
            return [round(pitch), round(roll), round(yaw)]
