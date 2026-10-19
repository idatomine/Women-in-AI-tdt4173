"""Feature groups and their cache.

Each group is a module in this package with a `build(data) -> {name: array}` function.
Arrays must broadcast to (case, unit, hour):
- (case, 1, hour) for hourly values shared by all units,
- (case, unit, 1) for one value per unit-week,
- (1, unit, 1) for unit statics.

A group is computed once for all 3647 cases and cached in features/ as float32. The cache
is rebuilt automatically when the group's module or src/data.py changes.

To add a group: create src/features/<name>.py with a `build` function, then list <name>
under `features:` in an experiment config.
"""

import hashlib
import importlib
import json

import numpy as np

from src.data import HOURS, ROOT, Data

CACHE_DIR = ROOT / "features"
PACKAGE_DIR = ROOT / "src" / "features"


def _code_hash(group: str) -> str:
    code = (PACKAGE_DIR / f"{group}.py").read_bytes() + (ROOT / "src" / "data.py").read_bytes()
    return hashlib.md5(code).hexdigest()[:10]


def load_group(group: str, data: Data) -> tuple[np.ndarray, list[str]]:
    """Features of one group, shape (case, unit, hour, feature), from the cache if possible."""
    stem = f"{group}__{_code_hash(group)}"
    array_path, names_path = CACHE_DIR / f"{stem}.npy", CACHE_DIR / f"{stem}.json"
    if array_path.exists() and names_path.exists():
        return np.load(array_path, mmap_mode="r"), json.loads(names_path.read_text())

    print(f"  building feature group '{group}' (cached afterwards)")
    features = importlib.import_module(f"src.features.{group}").build(data)
    shape = (data.n_cases, len(data.units), HOURS)
    array = np.empty(shape + (len(features),), dtype=np.float32)
    for j, (name, values) in enumerate(features.items()):
        array[..., j] = np.broadcast_to(values, shape)

    CACHE_DIR.mkdir(exist_ok=True)
    for old in CACHE_DIR.glob(f"{group}__*"):  # outdated versions of this group
        old.unlink()
    np.save(array_path, array)
    names_path.write_text(json.dumps(list(features)))
    return array, list(features)


def build_matrix(groups: list[str], data: Data, drop: list[str] = ()) -> tuple[np.ndarray, list[str]]:
    """All selected groups side by side, shape (case, unit, hour, feature).

    `drop` removes single features by name (the config's `drop_features`).
    """
    arrays, names, dropped = [], [], set()
    for group in groups:
        array, group_names = load_group(group, data)
        keep = [j for j, name in enumerate(group_names) if name not in drop]
        dropped |= set(group_names) - {group_names[j] for j in keep}
        arrays.append(array[..., keep])
        names += [group_names[j] for j in keep]
    unknown = set(drop) - dropped
    if unknown:
        raise ValueError(f"drop_features not found in the selected groups: {sorted(unknown)}")
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"feature names used by more than one group: {sorted(duplicates)}")
    return np.concatenate(arrays, axis=-1), names
