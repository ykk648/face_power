# -- coding: utf-8 --
# @Time : 2024/8/5
# @Author : ykk648
from tqdm import tqdm
from face_parsing.face_parsing_ibug.face_parsing_ibug_api import IbugFaceParser
from cv2box import CVVideoLoader, get_path_by_ext

ifp = IbugFaceParser()

# img_path = r''
# print(ifp.mouth_inner_exist(img_path))

# video_p = r''
# with CVVideoLoader(video_p) as cvvl:
#     for i in tqdm(range(len(cvvl))):
#         _, frame = cvvl.get()
#         if not ifp.mouth_inner_exist(frame):
#             print(i + 1)

image_dir = r''
for image_p in get_path_by_ext(image_dir, ['.jpg'], sorted_by_stem=True):
    if not ifp.mouth_inner_exist(str(image_p)):
        print(image_p)
