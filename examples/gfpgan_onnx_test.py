# -- coding: utf-8 --
# @Time : 2023/12/27
# @Author : ykk648
# @Project : https://github.com/ykk648/face_lib
from cv2box import CVImage

from face_restore_test import GFPGAN

face_img_p = 'resources/cropped_face/512.jpg'
fa = GFPGAN(model_type='GFPGANv1.4', provider='gpu')

face = fa.forward(face_img_p)
# CVImage(face, image_format='cv2').save('./gfpgan.jpg')
CVImage(face, image_format='cv2').show()

# # cuda forward test
# import onnxruntime
# import torch
# face_image_tensor_ = CVImage(face_img_p).blob(fa.input_size, fa.input_mean, fa.input_std, rgb=True)
# # face_image_tensor_ = onnxruntime.OrtValue.ortvalue_from_numpy(face_image_tensor_, 'cuda', 0)
# face_image_tensor_ = torch.tensor(face_image_tensor_).cuda()
# output_face = fa.cuda_forward(face_image_tensor_)
# CVImage(output_face, image_format='cv2').show()
