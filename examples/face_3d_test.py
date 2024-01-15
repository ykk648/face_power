# -- coding: utf-8 --
# @Time : 2024/1/15
# @Author : ykk648

from face_3d import Face3dCoeffs

f3c = Face3dCoeffs(model_type='facerecon_230425', provider='gpu')
coeffs_ = f3c.forward('resources/cropped_face/112.png')[0]
print(coeffs_.shape)
print(coeffs_)

face_vertex_, face_texture_, face_color_, landmark_ = f3c.get_3d_params(coeffs_)
# torch.Size([1, 35709, 3]) torch.Size([1, 35709, 3]) torch.Size([1, 35709, 3]) torch.Size([1, 68, 2])
print(face_vertex_.shape, face_texture_.shape, face_color_.shape, landmark_.shape)