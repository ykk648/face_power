# -- coding: utf-8 --
# @Time : 2024/2/29
# @Author : ykk648

from face_3d import DECA
from cv2box import CVImage
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

deca = DECA(device='cuda')
deca.eval()  # !!!!
# (B, 3, 224, 224), val: 0-1
input_tensor = CVImage('resources/cropped_face/112.png').resize(224).tensor().to('cuda')
src_codedict = deca.encode(input_tensor, use_detail=False)
src_outputs = deca.decode(src_codedict, rendering=False, vis_lmk=False, return_vis=False, use_detail=False)
print(src_outputs)
