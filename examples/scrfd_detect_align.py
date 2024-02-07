# -- coding: utf-8 --
# @Time : 2023/12/26
# @LastEdit : 2024/2/7
# @Author : ykk648

from cv2box import CVImage

from face_detect import ScrfdAPI
from face_alignment import FaceAlignmentAPI

if __name__ == '__main__':
    image_p = 'resources/multi_face.jpg'
    image_in = CVImage(image_p).bgr

    # # border image
    # borderpad = int(np.max([np.max(image_in.shape[:2]) * 0.1, 25]))
    # image_in = np.pad(image_in, ((borderpad, borderpad), (borderpad, borderpad), (0, 0)), 'constant',
    #                   constant_values=(0, 0))

    scrfd = ScrfdAPI(model_name='scrfd_500m_bnkps_shape640x640', provider='gpu', dynamic_shape=None)
    bboxs_, kps_ = scrfd.forward(image_in)
    scrfd.draw_face()
    print(bboxs_, kps_)

    fa = FaceAlignmentAPI(crop_size=256, mode='mtcnn_512')
    align_img, mat_rev, roi_box = fa.align_single_face(image_in, bboxs_, kps_, apply_roi=True, pad_ratio=0.2)
    CVImage(align_img).show()
