# -- coding: utf-8 --
# @Time : 2022/8/25
# @LastEdit : 2024/2/28
# @Author : ykk648

from cv2box import CVImage
from cv2box.utils.util import common_face_mask
from face_restore import GFPGAN, FaceRestore
from face_detect import ScrfdAPI
from face_alignment import FaceAlignmentAPI


class FaceRestorePipe:
    def __init__(self, mode='RestoreFormer', show=False):
        self.show = show

        self.fr = FaceRestore(mode=mode)
        self.scrfd = ScrfdAPI(model_name='scrfd_500m_bnkps_shape640x640')
        self.fa = FaceAlignmentAPI(crop_size=512, mode='mtcnn_512')
        self.fm = common_face_mask((512, 512))

    def face_detect_and_align(self, image_in):
        image_in = CVImage(image_in).bgr
        bboxes_scrfd, kpss_scrfd = self.scrfd.forward(image_in)
        face_image_, mat_rev_, roi_box_ = self.fa.align_single_face(image_in, bboxes_scrfd, kpss_scrfd,
                                                                    apply_roi=True, pad_ratio=0.1)
        if self.show:
            CVImage(face_image_).show(0)
        return face_image_, mat_rev_, roi_box_

    def face_restore(self, src_face_image_):
        return self.fr.forward(src_face_image_, output_size=512)

    def forward(self, image_in):
        image_in = CVImage(image_in).bgr
        face_image, mat_rev, roi_box = self.face_detect_and_align(image_in)
        if face_image is None:
            return image_in
        face_restore_out = self.face_restore(face_image)
        # face_restore_out = CVImage(face_restore_out).rgb()
        restore_roi = CVImage(None).recover_from_reverse_matrix(face_restore_out / 255,
                                                                image_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]],
                                                                mat_rev, img_fg_mask=self.fm)
        image_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = restore_roi
        return image_in


if __name__ == '__main__':

    # === for image ===
    src_img = 'resources/multi_face.jpg'
    fsp = FaceRestorePipe(show=True)
    img_result = fsp.forward(src_img)
    CVImage(img_result).show(0)

    # # === for video ===
    # from cv2box import CVVideoLoader
    # import cv2
    # from tqdm import tqdm
    #
    # frp = FaceRestorePipe(mode='gfpganv4')
    # video_p = ''
    # video_out_p = video_p.replace('.mp4', '_gfpganv4.mp4')
    #
    # with CVVideoLoader(video_p) as cvv:
    #     fps = cvv.fps
    #     size = cvv.size
    #
    # video_writer = cv2.VideoWriter(video_out_p, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    # with CVVideoLoader(video_p) as cvv:
    #     for _ in tqdm(range(len(cvv))):
    #         success, frame = cvv.get()
    #         frame_out = frp.forward(frame)
    #         # CVImage(frame_out).show()
    #         video_writer.write(frame)
