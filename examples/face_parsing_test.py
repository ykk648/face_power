# -- coding: utf-8 --
# @Time : 2023/12/27
# @Author : ykk648
# @Project : https://github.com/ykk648/face_lib
from cv2box import CVImage

from face_parsing import FaceParsing

test_img = 'resources/cropped_face/512.jpg'
fp = FaceParsing(model_name='face_parse_onnx', provider='gpu')

parsing = fp.forward(test_img)
fp.show()
mask = fp.get_face_mask((512, 512))
CVImage(mask).show(0)
