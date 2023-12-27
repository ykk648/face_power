# -- coding: utf-8 --
# @Time : 2023/12/22
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from cv2box import CVImage
from .mtcnn import MTCNN

# https://github.com/taotaonice/FaceShifter/blob/master/face_modules/mtcnn.py
# & https://github.com/TropComplique/mtcnn-pytorch
MTCNN_MODEL_PATH = 'pretrain_models/face_lib/face_detect/mtcnn_weights/'


class MtcnnAPI:
    def __init__(self, ):
        self.kpss = None
        self.bboxes = None
        self.model = MTCNN(model_dir=MTCNN_MODEL_PATH)

    def forward(self, image_in, min_bbox_size=1):
        pil_image = CVImage(image_in).pillow()
        self.bboxes, self.kpss = self.model.detect_faces(pil_image, min_face_size=min_bbox_size,
                                                         thresholds=[0.6, 0.7, 0.8],
                                                         nms_thresholds=[0.7, 0.7, 0.7])
        return self.bboxes, self.kpss
