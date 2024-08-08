# -- coding: utf-8 --
# @Time : 2024/8/5
# @Author : ykk648
import numpy as np
from tqdm import tqdm
from cv2box import CVFile, CVImage

from face_parsing.face_parsing_ibug.ibug.face_detection import RetinaFacePredictor
from face_parsing.face_parsing_ibug.ibug.face_parsing import FaceParser as RTNetPredictor
from face_parsing.face_parsing_ibug.ibug.face_parsing.utils import label_colormap

"""
0 : background
1 : skin (including face and scalp)
2 : left_eyebrow
3 : right_eyebrow
4 : left_eye
5 : right_eye
6 : nose
7 : upper_lip
8 : inner_mouth
9 : lower_lip
10 : hair
11 : left_ear
12 : right_ear
13 : glasses
"""


class IbugFaceParser:
    def __init__(self, device=0):
        self.face_detector = RetinaFacePredictor(
            threshold=0.9,
            device=f'cuda:{device}',
            model=(RetinaFacePredictor.get_model('mobilenet0.25')))
        self.face_parser = RTNetPredictor(
            device=f'cuda:{device}',
            ckpt=
            'pretrain_models/face_lib/face_parsing/ibug/rtnet50-fcn-14.torch',
            encoder='rtnet50',
            decoder='fcn',
            num_classes=14)

    def forward(self, img):
        img = CVImage(img).bgr
        faces = self.face_detector(img, rgb=False)

        if len(faces) == 0:
            print('no face detected')
            return None

        faces = faces[:1]
        masks = self.face_parser.predict_img(img, faces, rgb=False)

        # colormap = label_colormap(14)
        # res = colormap[masks[0]]
        # CVImage(res).show()

        return masks

    def mouth_inner_exist(self, img):
        masks = self.forward(img)
        mask = masks[0]
        selected_pixels = np.isin(mask, [8])

        # CVImage(mask.astype(np.uint8)).show()

        if np.count_nonzero(selected_pixels) <= 10:
            return False
        return True
