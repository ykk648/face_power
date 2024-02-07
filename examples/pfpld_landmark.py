# -- coding: utf-8 --
# @Time : 2023/12/27
# @Author : ykk648

from face_landmark import PFPLD

pfpld = PFPLD(model_name='pfpld', provider='gpu')

image_p = 'resources/cropped_face/112.png'

print(pfpld.forward(image_p))
pfpld.draw_face()
