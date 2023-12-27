# -- coding: utf-8 --
# @Time : 2023/12/27
# @Author : ykk648
# @Project : https://github.com/ykk648/face_lib
from face_embedding import FaceEmbedding

# CurricularFace
fe = FaceEmbedding(model_type='CurricularFace', provider='gpu')
latent = fe.forward('resources/cropped_face/112.png')
print(latent.shape)
print(min(latent), max(latent))


# insightface MBF
fe = FaceEmbedding(model_type='insightface_mbf')
latent_mbf_1 = fe.forward('resources/cropped_face/112.png')
latent_mbf_2 = fe.forward('resources/cropped_face/112.png')
print(latent_mbf_1.shape)
print(latent_mbf_1)
from cv2box.utils.math import CalDistance
print(CalDistance().sim(latent_mbf_1, latent_mbf_2))

