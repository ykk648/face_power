# -- coding: utf-8 --
# @Time : 2023/12/27
# @Author : ykk648

from face_detect import S3FD

s3 = S3FD(model_type='s3fd', provider='gpu')
detect_results = s3.forward('resources/multi_face.jpg')
print(detect_results)
