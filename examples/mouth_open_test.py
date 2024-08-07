# -- coding: utf-8 --
# @Time : 2024/8/7
# @Author : ykk648

from mouth_open_detect import MouthOpen
from face_landmark import PFPLD

image_p = 'resources/cropped_face/112.png'
fa = MouthOpen(model_name='openmouth_detect', provider='gpu')

pfpld = PFPLD(model_name='pfpld', provider='gpu')
landmark_68 = pfpld.forward(image_p)[0]

print(fa.forward(image_p, landmark_68))
