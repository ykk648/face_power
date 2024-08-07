# -- coding: utf-8 --
# @Time : 2023/12/27
# @LastEdit : 2024/2/28
# @Author : ykk648

from face_attr import FaceAttr

image_p = 'resources/cropped_face/512.jpg'
fa = FaceAttr(model_name='face_attr_mbnv3', provider='gpu')

print(fa.forward(image_p))
print(fa.show_label())
