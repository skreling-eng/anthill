"""Helpers for structured labels: [name, [(array, path), ...], meta?].

The optional third element is a dict (hashmap) for action-specific metadata.
"""

from __future__ import annotations

from ahlib.ah_runtime import ArrayBundle

LABEL_ELEMENT_TYPES = ("prompts", "texts", "images", "sounds", "videos", "files")


def label_meta(entry: object) -> dict:
    """Return the optional metadata dict from a label entry (empty if absent)."""
    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        third = entry[2]
        if isinstance(third, dict):
            return dict(third)
    return {}


def make_label_entry(
    name: str,
    elements: list[tuple[str, str]],
    meta: dict | None = None,
) -> list:
    """Build a label entry; omit meta when empty."""
    entry: list = [name, list(elements)]
    if meta:
        entry.append(dict(meta))
    return entry


def merge_label_meta(entry: object, updates: dict) -> list:
    """Return a label entry with merged metadata (updates override existing keys)."""
    parsed = normalize_label_entry(entry)
    if parsed is None:
        raise ValueError(f"invalid label entry: {entry!r}")
    name, elements, meta = parsed
    merged = {**meta, **updates}
    return make_label_entry(name, elements, merged or None)


def normalize_element_ref(item: object) -> tuple[str, str] | None:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        kind = str(item[0]).strip()
        path = str(item[1]).strip()
        if kind in LABEL_ELEMENT_TYPES and path:
            return kind, path
    return None


def normalize_label_entry(
    entry: object,
) -> tuple[str, list[tuple[str, str]], dict] | None:
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None
    name = str(entry[0]).strip()
    raw_elements = entry[1]
    if not name or not isinstance(raw_elements, (list, tuple)):
        return None
    elements: list[tuple[str, str]] = []
    for item in raw_elements:
        ref = normalize_element_ref(item)
        if ref is not None:
            elements.append(ref)
    return name, elements, label_meta(entry)


def entries_for_name(
    labels: list, name: str
) -> list[tuple[str, list[tuple[str, str]], dict]]:
    out: list[tuple[str, list[tuple[str, str]], dict]] = []
    for raw in labels:
        parsed = normalize_label_entry(raw)
        if parsed is not None and parsed[0] == name:
            out.append(parsed)
    return out


def bundle_from_elements(elements: list[tuple[str, str]]) -> ArrayBundle:
    out = ArrayBundle()
    for kind, path in elements:
        getattr(out, kind).append(path)
    return out


def bundle_link_set(bundle: ArrayBundle) -> set[tuple[str, str]]:
    links: set[tuple[str, str]] = set()
    for kind in LABEL_ELEMENT_TYPES:
        for path in getattr(bundle, kind):
            links.add((kind, path))
    return links


def label_entry_key(entry: object) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    parsed = normalize_label_entry(entry)
    if parsed is None:
        return None
    name, elements, _meta = parsed
    return name, tuple(elements)


def filter_labels_for_bundle(labels: list, bundle: ArrayBundle) -> list:
    """Keep label entries whose element links appear in bundle arrays."""
    bundle_links = bundle_link_set(bundle)
    out: list = []
    for raw in labels:
        parsed = normalize_label_entry(raw)
        if parsed is None:
            continue
        name, elements, meta = parsed
        kept = [(kind, path) for kind, path in elements if (kind, path) in bundle_links]
        if kept:
            out.append(make_label_entry(name, kept, meta or None))
    return out


def propagate_labels(input_bundle: ArrayBundle, output: ArrayBundle) -> ArrayBundle:
    """Copy label entries from input when their links still appear in output arrays."""
    if not input_bundle.labels:
        return output
    out = output.copy()
    out_links = bundle_link_set(out)
    existing = {
        key
        for raw in out.labels
        if (key := label_entry_key(raw)) is not None
    }
    for raw in input_bundle.labels:
        parsed = normalize_label_entry(raw)
        if parsed is None:
            continue
        name, elements, meta = parsed
        kept = [(kind, path) for kind, path in elements if (kind, path) in out_links]
        if not kept:
            continue
        entry = make_label_entry(name, kept, meta or None)
        key = label_entry_key(entry)
        if key is not None and key not in existing:
            out.labels.append(entry)
            existing.add(key)
    return out


def filter_by_label_name(bundle: ArrayBundle, name: str) -> ArrayBundle:
    """Keep only elements referenced by labels with the given name."""
    out = ArrayBundle()
    for raw in bundle.labels:
        parsed = normalize_label_entry(raw)
        if parsed is None or parsed[0] != name:
            continue
        out.labels.append(make_label_entry(parsed[0], parsed[1], parsed[2] or None))
        for kind, path in parsed[1]:
            getattr(out, kind).append(path)
    return out


def exclude_by_label_name(bundle: ArrayBundle, name: str) -> ArrayBundle:
    """Keep elements not tagged with name, including array elements with no labels."""
    excluded: set[tuple[str, str]] = set()
    for raw in bundle.labels:
        parsed = normalize_label_entry(raw)
        if parsed is not None and parsed[0] == name:
            excluded.update(parsed[1])

    out = ArrayBundle()
    for raw in bundle.labels:
        parsed = normalize_label_entry(raw)
        if parsed is None or parsed[0] == name:
            continue
        out.labels.append(make_label_entry(parsed[0], parsed[1], parsed[2] or None))

    seen: set[tuple[str, str]] = set()
    for kind in LABEL_ELEMENT_TYPES:
        for path in getattr(bundle, kind):
            ref = (kind, path)
            if ref in excluded or ref in seen:
                continue
            seen.add(ref)
            getattr(out, kind).append(path)

    for raw in bundle.labels:
        parsed = normalize_label_entry(raw)
        if parsed is None or parsed[0] == name:
            continue
        for kind, path in parsed[1]:
            ref = (kind, path)
            if ref in excluded or ref in seen:
                continue
            seen.add(ref)
            getattr(out, kind).append(path)

    return out


def add_label_for_elements(
    bundle: ArrayBundle,
    name: str,
    *,
    meta: dict | None = None,
) -> ArrayBundle:
    """Pass input through and append one label entry per element (except labels/changes)."""
    out = bundle.copy()
    for kind in LABEL_ELEMENT_TYPES:
        for path in getattr(bundle, kind):
            out.labels.append(make_label_entry(name, [(kind, path)], meta))
    return out
