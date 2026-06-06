from enum import IntEnum


class FaceType(IntEnum):
    HALF = 0
    MID_FULL = 1
    FULL = 2
    FULL_NO_ALIGN = 3
    WHOLE_FACE = 4
    HEAD = 10
    HEAD_NO_ALIGN = 20
    MARK_ONLY = 100


_to_string = {
    FaceType.HALF: "half_face",
    FaceType.MID_FULL: "midfull_face",
    FaceType.FULL: "full_face",
    FaceType.FULL_NO_ALIGN: "full_face_no_align",
    FaceType.WHOLE_FACE: "whole_face",
    FaceType.HEAD: "head",
    FaceType.HEAD_NO_ALIGN: "head_no_align",
    FaceType.MARK_ONLY: "mark_only",
}

_from_string = {v: k for k, v in _to_string.items()}


def face_type_from_string(value: str) -> FaceType:
    key = value.strip().lower()
    if key not in _from_string:
        raise ValueError(
            f"unknown face_type={value!r} "
            f"(expected one of: {', '.join(sorted(_from_string))})"
        )
    return _from_string[key]
