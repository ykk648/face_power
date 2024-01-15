# -- coding: utf-8 --
# @Time : 2024/1/15
# @Author : ykk648

from face_detect import ScrfdAPI
from head_pose import HeadPose
from cv2box import CVImage

image_in = 'resources/multi_face.jpg'

scrfd = ScrfdAPI(model_name='scrfd_500m_bnkps_shape640x640', provider='gpu', dynamic_shape=None)
bboxs_, kps_ = scrfd.forward(image_in)
scrfd.draw_face()
print(bboxs_, kps_)

hp = HeadPose()

for i in range(len(bboxs_)):
    x1, y1, x2, y2, _ = bboxs_[i]
    w, h = x2 - x1, y2 - y1
    w_c, h_c = (x2 + x1) / 2, (y2 + y1) / 2
    face_first = CVImage(image_in).bgr[int(h_c - h * 0.9):int(h_c + h * 0.6), int(w_c - w * 1):int(w_c + w * 1)]
    # CVImage(image_in).show()
    # face_first = CVImage(image_in).bgr[int(bboxs_[0][1]):int(bboxs_[0][3]),int(bboxs_[0][0]):int(bboxs_[0][2])]
    CVImage(face_first).show()
    print(hp.forward(face_first))
