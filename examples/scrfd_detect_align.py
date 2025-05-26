# -- coding: utf-8 --
# @Time : 2023/12/26
# @LastEdit : 2025/5/26
# @Author : ykk648

from cv2box import CVImage, CVVideoLoader

from face_detect import ScrfdAPI
from face_alignment import FaceAlignmentAPI

if __name__ == '__main__':
    scrfd = ScrfdAPI(model_name='scrfd_500m_bnkps_shape640x640', provider='gpu', dynamic_shape=None)

    image_p = 'resources/multi_face.jpg'
    image_in = CVImage(image_p).bgr

    # # border image
    # borderpad = int(np.max([np.max(image_in.shape[:2]) * 0.1, 25]))
    # image_in = np.pad(image_in, ((borderpad, borderpad), (borderpad, borderpad), (0, 0)), 'constant', constant_values=(0, 0))

    bboxs_, kps_ = scrfd.forward(image_in)
    scrfd.draw_face()
    print(bboxs_, kps_)

    # fa = FaceAlignmentAPI(crop_size=512, mode='multi_src_map_th')
    # align_img, mat_rev, roi_box = fa.align_single_face(image_in, bboxs_, kps_, apply_roi=True, pad_ratio=0.2)
    # CVImage(align_img).show()
    # # CVImage(align_img).save('./align_img.png')

    # # for video face debug
    # video_p = r''
    # with CVVideoLoader(video_p) as cvvl:
    #     fps = cvvl.fps
    #     size = cvvl.size
    #     for i in range(len(cvvl)):
    #         success, frame = cvvl.get()
    #         if not success:
    #             break
    #         bboxs_, kps_ = scrfd.forward(frame)
    #         scrfd.draw_face()
