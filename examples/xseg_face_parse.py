# -- coding: utf-8 --
# @Time : 2023/12/27
# @LastEdit : 2024/2/28
# @Author : ykk648

from cv2box import CVImage
import numpy as np
import cv2

# xseg test
from face_parsing import XsegNet
import onnxruntime as ort

ort.set_default_logger_severity(3)

image_p = 'resources/cropped_face/112.png'
image_in = CVImage(image_p).bgr

xseg = XsegNet(model_name='xseg_net')

output = xseg.forward(image_in)
print(output.shape)
print(output)
CVImage(output).show(0)

# masked face
mask = np.uint8(output * 255)
image_mask = cv2.bitwise_and(image_in, image_in, mask=mask)
CVImage(image_mask).show(0)

