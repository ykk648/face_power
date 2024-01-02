# -- coding: utf-8 --
# @Time : 2023/12/26
# @Author : ykk648
# @Project : https://github.com/ykk648/face_lib
from cv2box import CVImage
import cv2
import numpy as np
from .utils import area_center_filter, estimate_norm, apply_roi_func


class FaceAlignmentAPI:
    def __init__(self, crop_size=256, mode='mtcnn_512', smooth_m=False, smooth_window=5):
        """
        :param crop_size: align face size
        :param mode: norm estimate mode
        :param smooth_m: weather to smooth matrix
        :param smooth_window: smooth window
        """
        self.mode = mode
        self.crop_size = crop_size
        assert self.mode in ['default', 'mtcnn_512', 'mtcnn_256', 'arcface_512', 'arcface', 'default_95', 'multi',
                             'multi_src_map_th', 'multi_src_map_th_close', 'multi_src_map_remove_eye']

        self.smooth = smooth_m
        self.smooth_window = smooth_window
        if self.smooth:
            self.mat_list = []

    def norm_crop(self, img, landmark, crop_size=112, mode='arcface'):
        mat, pose_index, lmk_after = estimate_norm(landmark, crop_size, mode)

        if self.smooth:
            if len(self.mat_list) < self.smooth_window:
                self.mat_list.append(mat)
            else:
                self.mat_list.pop(0)
                self.mat_list.append(mat)
            mat = np.mean(np.array(self.mat_list), axis=0)

        # in some face copy&paste scene, border replicate will remove black line around bbox
        # warped = cv2.warpAffine(img, mat, (crop_size, crop_size), borderValue=0.0)
        warped = cv2.warpAffine(img, mat, (crop_size, crop_size), borderMode=cv2.BORDER_REPLICATE)
        mat_rev = cv2.invertAffineTransform(mat)
        return warped, mat_rev, lmk_after

    def align_single_face(self, image_in, bboxes, kpss, apply_roi=False, pad_ratio=0):
        """
        Args:
            image_in: CVImage acceptable class, BGR
            bboxes:
            kpss:
            apply_roi: roi around bbox
            pad_ratio: 0 for speed, 0.2 for best results
        Returns: align_img, mat_rev, roi_box
        """

        image = CVImage(image_in).rgb()

        if bboxes.shape[0] == 0:
            return None, None, None

        if len(bboxes.shape) > 1:
            best_bbox, best_index = area_center_filter(image.shape, bboxes)
            bbox = best_bbox
            kp = kpss[best_index]
        else:
            bbox = bboxes
            kp = kpss

        # # for talking head
        # if only_roi:
        #     roi, roi_box, roi_kpss = apply_roi_func(image, bbox, kp, pad_ratio=pad_ratio)
        #     roi = CVImage(roi).resize_keep_ratio(self.crop_size)[0]
        #     roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        #     return roi, roi_box, None

        if apply_roi:
            roi, roi_box, roi_kpss = apply_roi_func(image, bbox, kp, pad_ratio=pad_ratio)
            align_img, mat_rev, kpss_new = self.norm_crop(roi, roi_kpss, self.crop_size, mode=self.mode)
            align_img = CVImage(align_img).rgb()
            return align_img, mat_rev, roi_box
        else:
            align_img, mat_rev, kpss_new = self.norm_crop(image, kp, self.crop_size, mode=self.mode)
            align_img = CVImage(align_img).rgb()
            return align_img, mat_rev, None

    def align_multi_face(self, image_in, bboxes, kpss, apply_roi=False, pad_ratio=0):
        """
        Args:
            crop_size:
            apply_roi:
            pad_ratio:
        Returns:
        """
        if bboxes.shape[0] == 0:
            return None, None, None
        align_img_list = []
        mat_rev_list = []
        roi_box_list = []
        for i in range(bboxes.shape[0]):
            if kpss is not None:
                if apply_roi:
                    roi, roi_box, roi_kpss = apply_roi_func(image_in, bboxes[i], kpss[i], pad_ratio=pad_ratio)
                    align_img, mat_rev, _ = self.norm_crop(roi, roi_kpss, self.crop_size, mode=self.mode)
                else:
                    align_img, mat_rev, _ = self.norm_crop(image_in, kpss[i], self.crop_size, mode=self.mode)
                align_img = cv2.cvtColor(align_img, cv2.COLOR_RGB2BGR)
                align_img_list.append(align_img)
                mat_rev_list.append(mat_rev)
                if apply_roi:
                    roi_box_list.append(roi_box)
        return align_img_list, mat_rev_list, roi_box_list
