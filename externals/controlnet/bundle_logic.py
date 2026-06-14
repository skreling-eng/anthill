"""Split source images vs labeled control maps and build index-aligned combos."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ahlib.ah_runtime import ArrayBundle
from ahlib.label_utils import entries_for_name, normalize_label_entry

CONTROL_LABELS = ("pose", "depth", "canny")
SOURCE_LABELS = ("source", "img2img")

# InstantX Union ControlNet type names (comfy.cldm.control_types.UNION_CONTROLNET_TYPES).
LABEL_TO_UNION_TYPE = {
    "pose": "openpose",
    "depth": "depth",
    "canny": "canny",
}


@dataclass(frozen=True)
class ControlCombo:
    """One aligned tuple of control maps (any field may be absent)."""

    pose: str | None = None
    depth: str | None = None
    canny: str | None = None

    def items(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for label in CONTROL_LABELS:
            link = getattr(self, label)
            if link:
                out.append((LABEL_TO_UNION_TYPE[label], link))
        return out

    def __bool__(self) -> bool:
        return bool(self.pose or self.depth or self.canny)


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


def _control_labeled_links(bundle: ArrayBundle) -> set[str]:
    tagged: set[str] = set()
    for label in CONTROL_LABELS:
        for link in _labeled_image_links(bundle, label):
            tagged.add(link)
    return tagged


def _legacy_unlabeled_sources(bundle: ArrayBundle) -> list[str]:
    """Old behavior: any images[] not tagged pose/depth/canny."""
    tagged = _control_labeled_links(bundle)
    out: list[str] = []
    seen: set[str] = set()
    for path in bundle.images:
        if path in tagged or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def source_image_links(bundle: ArrayBundle) -> list[str]:
    """
    Optional reference / img2img sources.

    By default only images labeled ``source`` or ``img2img`` count. Unlabeled
    images in the bundle (e.g. @photo kept alongside control maps) are treated
    as the control reference, not an img2img source — matching Comfy's InstantX
    workflow (preprocess → controls → txt2img).

    Set AH_CONTROLNET_LEGACY_SOURCE=1 to restore unlabeled auto-detection.
    """
    labeled: list[str] = []
    seen: set[str] = set()
    for label in SOURCE_LABELS:
        for path in _labeled_image_links(bundle, label):
            if path in seen:
                continue
            seen.add(path)
            labeled.append(path)
    if labeled:
        return labeled

    if os.environ.get("AH_CONTROLNET_LEGACY_SOURCE", "").lower() in ("1", "true", "yes"):
        return _legacy_unlabeled_sources(bundle)
    return []


def control_combos(bundle: ArrayBundle) -> list[ControlCombo]:
    """Zip pose[i], depth[i], canny[i] by index; missing slots are allowed."""
    by_label = {label: _labeled_image_links(bundle, label) for label in CONTROL_LABELS}
    count = max((len(v) for v in by_label.values()), default=0)
    if count == 0:
        return []
    combos: list[ControlCombo] = []
    for index in range(count):
        combo = ControlCombo(
            pose=by_label["pose"][index] if index < len(by_label["pose"]) else None,
            depth=by_label["depth"][index] if index < len(by_label["depth"]) else None,
            canny=by_label["canny"][index] if index < len(by_label["canny"]) else None,
        )
        if combo:
            combos.append(combo)
    return combos


def passthrough_labels(bundle: ArrayBundle) -> list:
    """Keep non-control labels on output bundle."""
    out: list = []
    for raw in bundle.labels:
        parsed = normalize_label_entry(raw)
        if parsed is None or parsed[0] in CONTROL_LABELS or parsed[0] in SOURCE_LABELS:
            continue
        out.append(raw)
    return out


def validate_bundle(bundle: ArrayBundle) -> tuple[list[str], list[ControlCombo]]:
    sources = source_image_links(bundle)
    combos = control_combos(bundle)
    if not combos:
        raise RuntimeError(
            "$controlnet: need at least one control map labeled "
            f"{', '.join(CONTROL_LABELS)!r}"
        )
    return sources, combos
