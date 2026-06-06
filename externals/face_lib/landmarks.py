"""Minimal 2D landmark alignment (from DeepFaceLab facelib)."""

from __future__ import annotations

import math

import numpy as np
import numpy.linalg as npla

from externals.face_lib.face_type import FaceType
from externals.face_lib.umeyama import umeyama

landmarks_2D_new = np.array(
    [
        [0.000213256, 0.106454],
        [0.0752622, 0.038915],
        [0.18113, 0.0187482],
        [0.29077, 0.0344891],
        [0.393397, 0.0773906],
        [0.586856, 0.0773906],
        [0.689483, 0.0344891],
        [0.799124, 0.0187482],
        [0.904991, 0.038915],
        [0.98004, 0.106454],
        [0.490127, 0.203352],
        [0.490127, 0.307009],
        [0.490127, 0.409805],
        [0.490127, 0.515625],
        [0.36688, 0.587326],
        [0.426036, 0.609345],
        [0.490127, 0.628106],
        [0.554217, 0.609345],
        [0.613373, 0.587326],
        [0.121737, 0.216423],
        [0.187122, 0.178758],
        [0.265825, 0.179852],
        [0.334606, 0.231733],
        [0.260918, 0.245099],
        [0.182743, 0.244077],
        [0.645647, 0.231733],
        [0.714428, 0.179852],
        [0.793132, 0.178758],
        [0.858516, 0.216423],
        [0.79751, 0.244077],
        [0.719335, 0.245099],
        [0.254149, 0.780233],
        [0.726104, 0.780233],
    ],
    dtype=np.float32,
)

FaceType_to_padding_remove_align = {
    FaceType.HALF: (0.0, False),
    FaceType.MID_FULL: (0.0675, False),
    FaceType.FULL: (0.2109375, False),
    FaceType.FULL_NO_ALIGN: (0.2109375, True),
    FaceType.WHOLE_FACE: (0.40, False),
    FaceType.HEAD: (0.70, False),
    FaceType.HEAD_NO_ALIGN: (0.70, True),
}


def transform_points(points, mat, invert=False):
    import cv2

    if invert:
        mat = cv2.invertAffineTransform(mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    return np.squeeze(points)


def get_transform_mat(image_landmarks, output_size, face_type, scale=1.0):
    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array(image_landmarks)

    mat = umeyama(
        np.concatenate([image_landmarks[17:49], image_landmarks[54:55]]),
        landmarks_2D_new,
        True,
    )[0:2]

    g_p = transform_points(
        np.float32([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]),
        mat,
        True,
    )
    g_c = g_p[4]

    tb_diag_vec = (g_p[2] - g_p[0]).astype(np.float32)
    tb_diag_vec /= npla.norm(tb_diag_vec)
    bt_diag_vec = (g_p[1] - g_p[3]).astype(np.float32)
    bt_diag_vec /= npla.norm(bt_diag_vec)

    padding, remove_align = FaceType_to_padding_remove_align.get(face_type, (0.0, False))
    mod = (1.0 / scale) * (npla.norm(g_p[0] - g_p[2]) * (padding * math.sqrt(2.0) + 0.5))

    if face_type == FaceType.WHOLE_FACE:
        vec = (g_p[0] - g_p[3]).astype(np.float32)
        vec_len = npla.norm(vec)
        vec /= vec_len
        g_c += vec * vec_len * 0.07

    if not remove_align:
        l_t = np.array(
            [
                g_c - tb_diag_vec * mod,
                g_c + bt_diag_vec * mod,
                g_c + tb_diag_vec * mod,
            ]
        )
    else:
        l_t = np.array(
            [
                g_c - tb_diag_vec * mod,
                g_c + bt_diag_vec * mod,
                g_c + tb_diag_vec * mod,
                g_c - bt_diag_vec * mod,
            ]
        )
        area = 0.5 * abs(
            np.dot(l_t[:, 0], np.roll(l_t[:, 1], 1))
            - np.dot(l_t[:, 1], np.roll(l_t[:, 0], 1))
        )
        side = np.float32(math.sqrt(area) / 2)
        l_t = np.array(
            [
                g_c + [-side, -side],
                g_c + [side, -side],
                g_c + [side, side],
            ]
        )

    pts2 = np.float32(((0, 0), (output_size, 0), (output_size, output_size)))
    import cv2

    return cv2.getAffineTransform(l_t, pts2)
