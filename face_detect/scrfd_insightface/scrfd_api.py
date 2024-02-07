# -- coding: utf-8 --
# @Time : 2023/12/22
# @Author : ykk648

import numpy as np
import cv2
from cv2box import CVImage
from cv2box.utils.math import Normalize

from .scrfd import SCRFD

# https://github.com/deepinsight/insightface/tree/master/detection/scrfd
MODEL_ZOO = {
    'scrfd_500m_bnkps_shape640x640': {
        'model_path': 'pretrain_models/face_lib/face_detect/scrfd_onnx/scrfd_500m_bnkps_shape640x640.onnx',
        'input_dynamic_shape': (640, 640),
    },
    'scrfd_10g_bnkps': {
        'model_path': 'pretrain_models/face_lib/face_detect/scrfd_onnx/scrfd_10g_bnkps.onnx',
        'input_dynamic_shape': (640, 640),
    },

}


class ScrfdAPI:
    def __init__(self, model_name='scrfd_500m_bnkps_shape640x640', provider='gpu', dynamic_shape=None, tracking=False,
                 tracking_ratio=0.5):
        self.model = SCRFD(model_file=MODEL_ZOO[model_name]['model_path'])
        ctx_id = 0 if provider == 'gpu' else -1
        self.model.prepare(ctx_id=ctx_id,
                           input_size=dynamic_shape if dynamic_shape else MODEL_ZOO[model_name]['input_dynamic_shape'])

        self.image = None
        self.bboxes = None
        self.kpss = None

        # tracking one face
        self.tracking = tracking
        self.tracking_ratio = tracking_ratio
        self.dis_list = []
        self.last_bboxes_ = None

    def forward(self, image_in, nms_thresh=0.5, max_num=0):
        """
        Args:
            image_in: CVImage acceptable class, BGR
            nms_thresh: default 0.5
            max_num: 0 means detect all faces
        Returns:
            bboxes: (N,5) if max_num=0
            kpss: (N,5,2) if max_num=0
        """
        self.image = CVImage(image_in).rgb()

        if self.tracking:
            if self.last_bboxes_ is None:
                self.bboxes, self.kpss = self.model.detect_faces(self.image, thresh=nms_thresh, max_num=1,
                                                                 metric='default')
                self.last_bboxes_ = self.bboxes
            else:
                self.bboxes, self.kpss = self.model.detect_faces(self.image, thresh=nms_thresh, max_num=0,
                                                                 metric='default')
                self.bboxes, self.kpss = self.tracking_filter()
        else:
            self.bboxes, self.kpss = self.model.detect_faces(self.image, thresh=nms_thresh, max_num=max_num,
                                                             metric='default')

        return self.bboxes, self.kpss

    def tracking_filter(self):
        if len(self.bboxes) == 0 or len(self.last_bboxes_) == 0:
            return None, None
        for i in range(len(self.bboxes)):
            self.dis_list.append(
                np.linalg.norm(Normalize(self.bboxes[i]).np_norm() - Normalize(self.last_bboxes_[0]).np_norm()))
        if not self.dis_list:
            return None, None
        best_index = np.argmin(np.array(self.dis_list))
        # print(self.dis_list[best_index])
        if self.dis_list[best_index] > self.tracking_ratio:
            return None, None
        self.dis_list = []
        self.last_bboxes_ = np.array([self.bboxes[best_index]])
        return self.last_bboxes_, np.array([self.kpss[best_index]])

    def draw_face(self):
        draw_image = self.image.copy()
        for i_ in range(self.bboxes.shape[0]):
            bbox = self.bboxes[i_]
            x1, y1, x2, y2, score = bbox.astype(int)
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if self.kpss is not None:
                kps = self.kpss[i_]
                for kp in kps:
                    kp = kp.astype(int)
                    cv2.circle(draw_image, tuple(kp), 1, (0, 0, 255), 2)
        CVImage(CVImage(draw_image).rgb(), image_format='cv2').show()
        return draw_image
