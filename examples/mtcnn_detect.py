# -- coding: utf-8 --
# @Time : 2023/12/27
# @Author : ykk648
# @Project : https://github.com/ykk648/face_lib
from cv2box import CVImage

from face_detect import MtcnnAPI

image_p = 'resources/multi_face.jpg'
image_in = CVImage(image_p).resize(256).bgr

mtcnn = MtcnnAPI()
print(mtcnn.forward(image_in))

