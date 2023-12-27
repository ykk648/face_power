# -- coding: utf-8 --
# @Time : 2023/12/27
# @Author : ykk648
# @Project : https://github.com/ykk648/face_lib
from face_attr import FaceAttr

image_p = 'resources/cropped_face/512.jpg'
fa = FaceAttr(model_name='face_attr_mbnv3', provider='gpu')

print(fa.forward(image_p))
