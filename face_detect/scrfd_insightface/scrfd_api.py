# -- coding: utf-8 --
# @Time : 2023/12/22
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
import cv2
from cv2box import CVImage

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
    def __init__(self, model_name='scrfd_500m_bnkps_shape640x640', provider='gpu', dynamic_shape=None):
        self.model = SCRFD(model_file=MODEL_ZOO[model_name]['model_path'])
        ctx_id = 0 if provider == 'gpu' else -1
        self.model.prepare(ctx_id=ctx_id, input_size=dynamic_shape if dynamic_shape else MODEL_ZOO[model_name]['input_dynamic_shape'])

        self.image = None
        self.bboxes = None
        self.kpss = None

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
        self.bboxes, self.kpss = self.model.detect_faces(self.image, thresh=nms_thresh, max_num=max_num,
                                                         metric='default')
        return self.bboxes, self.kpss

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



