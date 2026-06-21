"""Resolve target vs face-labeled images for $image_face_swap."""

from __future__ import annotations

from dataclasses import dataclass

from ahlib.ah_runtime import ArrayBundle
from ahlib.label_utils import entries_for_name, normalize_label_entry

FACE_LABEL = "face"

DEFAULT_PROMPT = (
    "Referring to Images 1 and 2, remove the face and replace the person's head "
    "and face in Image 1 with the one from Image 2, while keeping the natural "
    "lighting and the face skin color of the person in Image 1."
)

DEFAULT_WORKFLOW = "Flux 2 Klein Precise Face_Head Swap Final V2.json"


def _labeled_image_links(bundle: ArrayBundle, label: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for _name, elements, _meta in entries_for_name(bundle.labels, label):
        for kind, path in elements:
            if kind != "images" or not path or path in seen:
                continue
            seen.add(path)
            links.append(path)
    return links


def face_image_links(bundle: ArrayBundle) -> list[str]:
    return _labeled_image_links(bundle, FACE_LABEL)


def target_image_links(bundle: ArrayBundle) -> list[str]:
    face_set = set(face_image_links(bundle))
    out: list[str] = []
    seen: set[str] = set()
    for path in bundle.images:
        if path in face_set or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


@dataclass(frozen=True)
class FaceSwapJob:
    target: str
    face: str


def face_swap_jobs(bundle: ArrayBundle) -> list[FaceSwapJob]:
    targets = target_image_links(bundle)
    faces = face_image_links(bundle)
    if not targets:
        raise RuntimeError(
            "$image_face_swap: need at least one target image in images[] "
            "(not labeled 'face')"
        )
    if not faces:
        raise RuntimeError(
            "$image_face_swap: need at least one image labeled 'face' "
            "(use $add_label('face') on the donor face)"
        )
    count = max(len(targets), len(faces))
    jobs: list[FaceSwapJob] = []
    for index in range(count):
        jobs.append(
            FaceSwapJob(
                target=targets[min(index, len(targets) - 1)],
                face=faces[min(index, len(faces) - 1)],
            )
        )
    return jobs


def passthrough_labels(bundle: ArrayBundle) -> list:
    """Keep labels except the face donor tag on output."""
    out: list = []
    for raw in bundle.labels:
        parsed = normalize_label_entry(raw)
        if parsed is None or parsed[0] == FACE_LABEL:
            continue
        out.append(raw)
    return out
