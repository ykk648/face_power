# -- coding: utf-8 --
# @Time : 2023/12/26
# @LastEdit : 2025/4/25
# @Author : ykk648

import cv2
import numpy as np
from skimage import transform as trans
from cv2box import CVBbox

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
# <--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

# ---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

# -->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

# -->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

multi_src = np.array([src1, src2, src3, src4, src5])
multi_src_map = {112: multi_src, 224: multi_src * 2, 512: multi_src * (512 / 112)}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

mtcnn_512 = [[187.20187, 239.27705],
             [324.1236, 238.51973],
             [256.09793, 317.14795],
             [199.84871, 397.30597],
             [313.2362, 396.6788]]

mtcnn_256 = np.array(mtcnn_512) * 0.5

arcface_src_512 = arcface_src * np.array([512 / 112, 512 / 112])
arcface_src = np.expand_dims(arcface_src, axis=0)

arcface_src_224 = arcface_src * 2

# for talking head
multi_src_th = np.array([[
    [190, 102.4],
    [342, 102.4],
    [256, 180],
    [190, 274],
    [322, 274]],  # front
])
multi_src_th_close = np.array([[
    [152, 20],
    [360, 20],
    [256, 144],
    [174, 280],
    [338, 280]],  # front
])
multi_src_remove_eye = np.array([[
    [63, -18],
    [193, -18],
    [128, 35],
    [128, 100],
    [128, 100]],  # front
])
multi_src_map_th = {256: multi_src_th / 2, 512: multi_src_th}
multi_src_map_th_close = {256: multi_src_th_close / 2, 512: multi_src_th_close}
multi_src_map_remove_eye = {256: multi_src_remove_eye, 512: multi_src_remove_eye * 2}


def get_src_modify(srcs, arcface_src):
    srcs += ((arcface_src[2] - srcs[2][2]) * np.array([1, 1.8]))[None]
    return srcs


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        # assert image_size == 112
        src = arcface_src
        src_map = {112: src.copy(), 128: src.copy() * 128 / 112}
        src = src_map[image_size]
    elif mode == 'arcface_224':
        # for Deep3DFaceRecon
        assert image_size == 224
        src = arcface_src_224
    elif mode == 'arcface_512':
        src = np.expand_dims(arcface_src_512, axis=0)
    elif mode == 'mtcnn_512':
        src = np.expand_dims(mtcnn_512, axis=0)
    elif mode == 'mtcnn_256':
        src = np.expand_dims(mtcnn_256, axis=0)
    elif mode == 'default_95':
        src = get_src_modify(multi_src, arcface_src[0])
        src_map = {112: src.copy(), 224: src.copy() * 2, 256: src.copy() * 256 / 112 * 0.95,
                   512: src.copy() * (512 / 112) * 0.95}
        src = src_map[image_size]
    elif mode == 'multi_src_map_th':
        src = multi_src_map_th[image_size]
    elif mode == 'multi_src_map_th_close':
        src = multi_src_map_th_close[image_size]
    elif mode == 'multi_src_map_remove_eye':
        src = multi_src_map_remove_eye[image_size]
        lmk[3:, 0] = sum(lmk[3:, 0]) / 2
        lmk[3:, 1] = sum(lmk[3:, 1]) / 2
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    else:
        src = multi_src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index, results


def norm_crop(img, landmark, crop_size=112, mode='arcface'):
    mat, pose_index, lmk_after = estimate_norm(landmark, crop_size, mode)
    # in some face copy&paste scene, border replicate will remove black line around bbox
    # warped = cv2.warpAffine(img, mat, (crop_size, crop_size), borderValue=0.0)
    warped = cv2.warpAffine(img, mat, (crop_size, crop_size), borderMode=cv2.BORDER_REPLICATE)
    mat_rev = cv2.invertAffineTransform(mat)
    return warped, mat_rev, lmk_after


def apply_roi_func(img, box, facial5points, pad_ratio=0):
    """
    Args:
        img:
        box:
        facial5points:
        pad_ratio: set to 0 to speedup
    Returns:
    """
    box = np.round(np.array(box)).astype(int)[:4]

    roi_pad = int(pad_ratio * max([box[2] - box[0], box[3] - box[1]]))
    roi_box = np.array([
        max(0, box[0] - roi_pad),
        max(0, box[1] - roi_pad),
        min(img.shape[1], box[2] + roi_pad),
        min(img.shape[0], box[3] + roi_pad)
    ])

    roi = img[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]].copy()

    mrow1 = roi_box[1]
    mcol1 = roi_box[0]

    roi_facial5points = facial5points.copy()

    roi_facial5points[:, 0] -= mcol1
    roi_facial5points[:, 1] -= mrow1

    return roi, roi_box, roi_facial5points


def area_center_filter(image_shape, bboxes):
    # area center mode
    bbox_filtered_result = CVBbox(bboxes).area_center_filter(image_shape, max_num=1)
    best_bbox = bbox_filtered_result[0][0]
    best_index = bbox_filtered_result[0][1]

    return best_bbox, best_index
