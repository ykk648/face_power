import cv2
import numpy as np
from typing import Optional, Any, Callable


__all__ = ['make_square_rois',
           'roi_tanh_warp', 'roi_tanh_restore',
           'roi_tanh_polar_warp', 'roi_tanh_polar_restore',
           'roi_tanh_circular_warp', 'roi_tanh_circular_restore',
           'roi_tanh_polar_to_roi_tanh', 'roi_tanh_to_roi_tanh_polar',
           'get_warp_func', 'get_restore_func']


def make_square_rois(rois: np.ndarray, opt: int = 0) -> np.ndarray:
    roi_sizes = (rois[..., 2:4] - rois[..., :2])
    if opt < 0:
        target_sizes = roi_sizes.min(axis=-1)
    elif opt > 0:
        target_sizes = roi_sizes.max(axis=-1)
    else:
        target_sizes = (roi_sizes[..., 0] * roi_sizes[..., 1]) ** 0.5
    deltas = (roi_sizes - target_sizes) / 2.0
    deltas = np.hstack((deltas, -deltas))
    return rois + deltas


def roi_tanh_warp(image: np.ndarray, roi: np.ndarray, target_width: int, target_height: int,
                  angular_offset: float = 0.0, interpolation: Optional[int] = cv2.INTER_LINEAR,
                  border_mode: Optional[int] = cv2.BORDER_CONSTANT, border_value: Any = 0) -> np.ndarray:
    roi_center = (roi[2:4] + roi[:2]) / 2.0
    half_roi_size = (roi[2:4] - roi[:2]) / 2.0
    cos_offset, sin_offset = np.cos(angular_offset), np.sin(angular_offset)

    normalised_dest_x_indices = np.arange(-1.0, 1.0, 2.0 / target_width) + 1.0 / target_width
    normalised_dest_y_indices = np.arange(-1.0, 1.0, 2.0 / target_height) + 1.0 / target_height

    src_x_indices = np.tile(half_roi_size[0] * np.arctanh(normalised_dest_x_indices), (target_height, 1))
    src_y_indices = np.tile(half_roi_size[1] * np.arctanh(normalised_dest_y_indices), (target_width, 1)).transpose()
    src_x_indices, src_y_indices = (roi_center[0] + cos_offset * src_x_indices - sin_offset * src_y_indices,
                                    roi_center[1] + cos_offset * src_y_indices + sin_offset * src_x_indices)

    return cv2.remap(image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_restore(warped_image: np.ndarray, roi: np.ndarray, image_width: int, image_height: int,
                     angular_offset: float = 0.0, interpolation: Optional[int] = cv2.INTER_LINEAR,
                     border_mode: Optional[int] = cv2.BORDER_CONSTANT, border_value: Any = 0) -> np.ndarray:
    roi_center = (roi[2:4] + roi[:2]) / 2.0
    half_roi_size = (roi[2:4] - roi[:2]) / 2.0
    cos_offset, sin_offset = np.cos(angular_offset), np.sin(angular_offset)

    dest_indices = np.stack(np.meshgrid(np.arange(image_width), np.arange(image_height)), axis=-1).astype(float)
    src_indices = (np.tanh(np.matmul(dest_indices - roi_center, np.array([[cos_offset, -sin_offset],
                                                                          [sin_offset, cos_offset]])) /
                           half_roi_size) + 1.0) / 2.0 * warped_image.shape[1::-1] - 0.5

    return cv2.remap(warped_image, src_indices[..., 0].astype(np.float32), src_indices[..., 1].astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_polar_warp(image: np.ndarray, roi: np.ndarray, target_width: int, target_height: int,
                        angular_offset: float = 0.0, interpolation: Optional[int] = cv2.INTER_LINEAR,
                        border_mode: Optional[int] = cv2.BORDER_CONSTANT, border_value: Any = 0,
                        keep_aspect_ratio: bool = False) -> np.ndarray:
    roi_center = (roi[2:4] + roi[:2]) / 2.0
    roi_radii = (roi[2:4] - roi[:2]) / np.pi ** 0.5
    cos_offset, sin_offset = np.cos(angular_offset), np.sin(angular_offset)

    normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / target_width),
                                                   np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / target_height)),
                                       axis=-1)
    radii = normalised_dest_indices[..., 0]
    orientation_x = np.cos(normalised_dest_indices[..., 1])
    orientation_y = np.sin(normalised_dest_indices[..., 1])

    if keep_aspect_ratio:
        src_radii = np.arctanh(radii) * (roi_radii[0] * roi_radii[1] / np.sqrt(
            roi_radii[1] ** 2 * orientation_x ** 2 + roi_radii[0] ** 2 * orientation_y ** 2))
        src_x_indices = src_radii * orientation_x
        src_y_indices = src_radii * orientation_y
    else:
        src_radii = np.arctanh(radii)
        src_x_indices = roi_radii[0] * src_radii * orientation_x
        src_y_indices = roi_radii[1] * src_radii * orientation_y
    src_x_indices, src_y_indices = (roi_center[0] + cos_offset * src_x_indices - sin_offset * src_y_indices,
                                    roi_center[1] + cos_offset * src_y_indices + sin_offset * src_x_indices)

    return cv2.remap(image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_polar_restore(warped_image: np.ndarray, roi: np.ndarray, image_width: int, image_height: int,
                           angular_offset: float = 0.0, interpolation: Optional[int] = cv2.INTER_LINEAR,
                           border_mode: Optional[int] = cv2.BORDER_CONSTANT, border_value: Any = 0,
                           keep_aspect_ratio: bool = False) -> np.ndarray:
    warped_height, warped_width = warped_image.shape[:2]
    roi_center = (roi[2:4] + roi[:2]) / 2.0
    roi_radii = (roi[2:4] - roi[:2]) / np.pi ** 0.5
    cos_offset, sin_offset = np.cos(angular_offset), np.sin(angular_offset)

    dest_indices = np.stack(np.meshgrid(np.arange(image_width), np.arange(image_height)), axis=-1).astype(float)
    normalised_dest_indices = np.matmul(dest_indices - roi_center, np.array([[cos_offset, -sin_offset],
                                                                             [sin_offset, cos_offset]]))
    if keep_aspect_ratio:
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)
        normalised_dest_indices[..., 0] /= np.clip(radii, 1e-9, None)
        normalised_dest_indices[..., 1] /= np.clip(radii, 1e-9, None)
        radii *= np.sqrt(roi_radii[1] ** 2 * normalised_dest_indices[..., 0] ** 2 +
                         roi_radii[0] ** 2 * normalised_dest_indices[..., 1] ** 2) / roi_radii[0] / roi_radii[1]
    else:
        normalised_dest_indices /= roi_radii
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)

    src_radii = np.tanh(radii)
    warped_image = np.pad(np.pad(warped_image, [(1, 1), (0, 0)] + [(0, 0)] * (warped_image.ndim - 2), mode='wrap'),
                          [(0, 0), (1, 0)] + [(0, 0)] * (warped_image.ndim - 2), mode='edge')
    src_x_indices = src_radii * warped_width + 1.0
    src_y_indices = np.mod((np.arctan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) /
                            2.0 / np.pi) * warped_height, warped_height) + 1.0

    return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_circular_warp(image: np.ndarray, roi: np.ndarray, target_width: int, target_height: int,
                           angular_offset: float = 0.0, interpolation: Optional[int] = cv2.INTER_LINEAR,
                           border_mode: Optional[int] = cv2.BORDER_CONSTANT, border_value: Any = 0,
                           keep_aspect_ratio: bool = False) -> np.ndarray:
    roi_center = (roi[2:4] + roi[:2]) / 2.0
    roi_radii = (roi[2:4] - roi[:2]) / np.pi ** 0.5
    cos_offset, sin_offset = np.cos(angular_offset), np.sin(angular_offset)

    normalised_dest_x_indices = np.arange(-1.0, 1.0, 2.0 / target_width) + 1.0 / target_width
    normalised_dest_y_indices = np.arange(-1.0, 1.0, 2.0 / target_height) + 1.0 / target_height
    normalised_dest_indices = np.stack(np.meshgrid(normalised_dest_x_indices, normalised_dest_y_indices), axis=-1)
    radii = np.linalg.norm(normalised_dest_indices, axis=-1)
    orientation_x = normalised_dest_indices[..., 0] / np.clip(radii, 1e-9, None)
    orientation_y = normalised_dest_indices[..., 1] / np.clip(radii, 1e-9, None)

    if keep_aspect_ratio:
        src_radii = np.arctanh(np.clip(radii, None, 1.0 - 1e-9)) * (roi_radii[0] * roi_radii[1] / np.sqrt(
            roi_radii[1] ** 2 * orientation_x ** 2 + roi_radii[0] ** 2 * orientation_y ** 2))
        src_x_indices = src_radii * orientation_x
        src_y_indices = src_radii * orientation_y
    else:
        src_radii = np.arctanh(np.clip(radii, None, 1.0 - 1e-9))
        src_x_indices = roi_radii[0] * src_radii * orientation_x
        src_y_indices = roi_radii[1] * src_radii * orientation_y
    src_x_indices, src_y_indices = (roi_center[0] + cos_offset * src_x_indices - sin_offset * src_y_indices,
                                    roi_center[1] + cos_offset * src_y_indices + sin_offset * src_x_indices)

    return cv2.remap(image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_circular_restore(warped_image: np.ndarray, roi: np.ndarray, image_width: int, image_height: int,
                              angular_offset: float = 0.0, interpolation: Optional[int] = cv2.INTER_LINEAR,
                              border_mode: Optional[int] = cv2.BORDER_CONSTANT, border_value: Any = 0,
                              keep_aspect_ratio: bool = False) -> np.ndarray:
    warped_height, warped_width = warped_image.shape[:2]
    roi_center = (roi[2:4] + roi[:2]) / 2.0
    roi_radii = (roi[2:4] - roi[:2]) / np.pi ** 0.5
    cos_offset, sin_offset = np.cos(angular_offset), np.sin(angular_offset)

    dest_indices = np.stack(np.meshgrid(np.arange(image_width), np.arange(image_height)), axis=-1).astype(float)
    normalised_dest_indices = np.matmul(dest_indices - roi_center, np.array([[cos_offset, -sin_offset],
                                                                             [sin_offset, cos_offset]]))
    if keep_aspect_ratio:
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)
        orientation_x = normalised_dest_indices[..., 0] / np.clip(radii, 1e-9, None)
        orientation_y = normalised_dest_indices[..., 1] / np.clip(radii, 1e-9, None)
        radii *= np.sqrt(roi_radii[1] ** 2 * orientation_x ** 2 +
                         roi_radii[0] ** 2 * orientation_y ** 2) / roi_radii[0] / roi_radii[1]
    else:
        normalised_dest_indices /= roi_radii
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)
        orientation_x = normalised_dest_indices[..., 0] / np.clip(radii, 1e-9, None)
        orientation_y = normalised_dest_indices[..., 1] / np.clip(radii, 1e-9, None)

    src_radii = np.tanh(radii)
    src_x_indices = (orientation_x * src_radii + 1.0) * warped_width / 2.0 - 0.5
    src_y_indices = (orientation_y * src_radii + 1.0) * warped_height / 2.0 - 0.5

    return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_polar_to_roi_tanh(warped_image: np.ndarray, roi: np.ndarray,
                               target_width: int = 0, target_height: int = 0,
                               interpolation: Optional[int] = cv2.INTER_LINEAR,
                               border_mode: Optional[int] = cv2.BORDER_CONSTANT,
                               border_value: Any = 0, keep_aspect_ratio: bool = False) -> np.ndarray:
    target_width = warped_image.shape[1] if target_width <= 0 else target_width
    target_height = warped_image.shape[0] if target_height <= 0 else target_height
    warped_height, warped_width = warped_image.shape[:2]
    half_roi_size = (roi[2:4] - roi[:2]) / 2.0
    roi_radii = half_roi_size * 2.0 / np.pi ** 0.5

    normalised_dest_x_indices = np.arange(-1.0, 1.0, 2.0 / target_width) + 1.0 / target_width
    normalised_dest_y_indices = np.arange(-1.0, 1.0, 2.0 / target_height) + 1.0 / target_height
    normalised_dest_indices = np.stack(np.meshgrid(half_roi_size[0] * np.arctanh(normalised_dest_x_indices),
                                                   half_roi_size[1] * np.arctanh(normalised_dest_y_indices)), axis=-1)

    if keep_aspect_ratio:
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)
        normalised_dest_indices[..., 0] /= np.clip(radii, 1e-9, None)
        normalised_dest_indices[..., 1] /= np.clip(radii, 1e-9, None)
        radii *= np.sqrt(roi_radii[1] ** 2 * normalised_dest_indices[..., 0] ** 2 +
                         roi_radii[0] ** 2 * normalised_dest_indices[..., 1] ** 2) / roi_radii[0] / roi_radii[1]
    else:
        normalised_dest_indices /= roi_radii
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)

    src_radii = np.tanh(radii)
    warped_image = np.pad(np.pad(warped_image, [(1, 1), (0, 0)] + [(0, 0)] * (warped_image.ndim - 2), mode='wrap'),
                          [(0, 0), (1, 0)] + [(0, 0)] * (warped_image.ndim - 2), mode='edge')
    src_x_indices = src_radii * warped_width + 1.0
    src_y_indices = np.mod((np.arctan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) /
                            2.0 / np.pi) * warped_height, warped_height) + 1.0

    return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_to_roi_tanh_polar(warped_image: np.ndarray, roi: np.ndarray,
                               target_width: int = 0, target_height: int = 0,
                               interpolation: Optional[int] = cv2.INTER_LINEAR,
                               border_mode: Optional[int] = cv2.BORDER_CONSTANT,
                               border_value: Any = 0, keep_aspect_ratio: bool = False) -> np.ndarray:
    target_width = warped_image.shape[1] if target_width <= 0 else target_width
    target_height = warped_image.shape[0] if target_height <= 0 else target_height
    warped_height, warped_width = warped_image.shape[:2]
    half_roi_size = (roi[2:4] - roi[:2]) / 2.0
    roi_radii = half_roi_size * 2.0 / np.pi ** 0.5

    normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / target_width),
                                                   np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / target_height)),
                                       axis=-1)
    radii = normalised_dest_indices[..., 0]
    orientation_x = np.cos(normalised_dest_indices[..., 1])
    orientation_y = np.sin(normalised_dest_indices[..., 1])

    if keep_aspect_ratio:
        src_radii = np.arctanh(radii) * (roi_radii[0] * roi_radii[1] / np.sqrt(
            roi_radii[1] ** 2 * orientation_x ** 2 + roi_radii[0] ** 2 * orientation_y ** 2))
        src_x_indices = src_radii * orientation_x
        src_y_indices = src_radii * orientation_y
    else:
        src_radii = np.arctanh(radii)
        src_x_indices = roi_radii[0] * src_radii * orientation_x
        src_y_indices = roi_radii[1] * src_radii * orientation_y

    src_x_indices = (np.tanh(src_x_indices / half_roi_size[0]) + 1.0) / 2.0 * warped_width - 0.5
    src_y_indices = (np.tanh(src_y_indices / half_roi_size[1]) + 1.0) / 2.0 * warped_height - 0.5

    return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def get_warp_func(variant: str = 'cartesian') -> Callable[..., np.ndarray]:
    variant = variant.lower().strip()
    if variant == 'cartesian':
        return roi_tanh_warp
    elif variant == 'polar':
        return roi_tanh_polar_warp
    elif variant == 'circular':
        return roi_tanh_circular_warp
    else:
        raise ValueError('variant must be set to either cartesian, polar, or circular')


def get_restore_func(variant: str = 'cartesian') -> Callable[..., np.ndarray]:
    variant = variant.lower().strip()
    if variant == 'cartesian':
        return roi_tanh_restore
    elif variant == 'polar':
        return roi_tanh_polar_restore
    elif variant == 'circular':
        return roi_tanh_circular_restore
    else:
        raise ValueError('variant must be set to either cartesian, polar, or circular')
