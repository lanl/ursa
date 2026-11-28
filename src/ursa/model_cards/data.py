"""Inspect a user's data file and describe it in model-card terms.

A model card says what a model expects: the kind of data, the variables with
their shapes and units, the grid, and the time step. This module reads a
user's file and states the same facts about the data, so the two can be
compared field by field. Each fact names the file and the reader that found
it, which keeps answers traceable.

Readers never load a large array whole. Shapes and attributes come from file
metadata, and value ranges come from a bounded sample. Optional libraries
(``h5py``, ``pandas``, ``scipy``, ``Pillow``) are imported only when a file
needs them; a missing one becomes a note instead of an error.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

DataKind = Literal[
    "field",
    "tensor",
    "time_series",
    "table",
    "molecule",
    "sequence",
    "structure",
    "image",
    "other",
]

MAX_SAMPLE = 1_000_000
MAX_TEXT_BYTES = 50_000_000
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_NETCDF3_MAGIC = (b"CDF\x01", b"CDF\x02", b"CDF\x05")


def _count(n: int, noun: str) -> str:
    """``1 sample`` or ``10 samples``."""
    return f"{n:,} {noun}" + ("" if n == 1 else "s")


class VariableProfile(BaseModel):
    """One variable, array, or column found in the data."""

    name: str
    shape: list[int] = Field(default_factory=list)
    dims: list[str] = Field(default_factory=list)
    dtype: str | None = None
    units: str | None = None
    long_name: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    levels: list[float] = Field(
        default_factory=list,
        description="Vertical levels, when the variable has a level axis.",
    )


class DataProfile(BaseModel):
    """What a data file holds, stated in the terms model cards use."""

    path: str
    format: str
    reader: str = Field(description="Library or parser that read the file.")
    kind: DataKind = "other"
    representation: str | None = Field(
        default=None,
        description="Concrete encoding, such as 'SMILES' or 'The Well HDF5'.",
    )
    variables: list[VariableProfile] = Field(default_factory=list)
    n_samples: int | None = Field(
        default=None, description="Rows, molecules, sequences, or frames."
    )
    spatial_dims: int | None = None
    grid: list[int] = Field(
        default_factory=list, description="Spatial grid sizes, in order."
    )
    resolution: str | None = None
    time_steps: int | None = None
    time_step: str | None = None
    labels: list[str] = Field(
        default_factory=list,
        description="Columns or arrays that look like targets for training.",
    )
    examples: list[str] = Field(default_factory=list)
    column: str | None = Field(
        default=None,
        description="Table column that holds the molecules or sequences.",
    )
    item_sizes: dict[str, list[int]] = Field(
        default_factory=dict,
        description=(
            "Size of each item by measure, such as 'heavy atoms' per molecule "
            "or 'residues' per sequence, for the first MAX_ITEMS items."
        ),
    )
    notes: list[str] = Field(default_factory=list)

    @property
    def name(self) -> str:
        return Path(self.path).name

    def summary(self) -> str:
        """One line that describes the data."""
        parts = [f"{self.format}, {self.kind.replace('_', ' ')}"]
        if self.representation:
            parts.append(self.representation)
        if self.n_samples is not None:
            parts.append(_count(self.n_samples, "sample"))
        if self.spatial_dims:
            grid = " x ".join(str(n) for n in self.grid) if self.grid else ""
            parts.append(
                f"{self.spatial_dims}D grid" + (f" {grid}" if grid else "")
            )
        if self.resolution:
            parts.append(f"resolution {self.resolution}")
        if self.time_steps:
            step = f" every {self.time_step}" if self.time_step else ""
            parts.append(_count(self.time_steps, "time step") + step)
        return "; ".join(parts)


# Coordinate and label names --------------------------------------------------

_LAT = {"lat", "latitude", "nav_lat", "y_lat"}
_LON = {"lon", "longitude", "nav_lon", "x_lon"}
_SPACE = {"x", "y", "z", "nx", "ny", "nz", "i", "j", "k"}
_TIME = {"time", "t", "times", "valid_time", "step", "lead_time"}
_LEVEL = {
    "level",
    "lev",
    "plev",
    "pressure",
    "pressure_level",
    "isobaricinhpa",
    "height",
    "depth",
}
_SAMPLE = {"sample", "samples", "batch", "trajectory", "trajectories", "member"}
_LABEL_NAMES = re.compile(
    r"^(?:y|label|labels|target|targets|class|classes|property|value)$", re.I
)
# Columns that name or number rows, not targets. "index" alone is an ID,
# while "refractive_index" is a quantity.
_ID_NAMES = re.compile(
    r"^(?:index|row|key|sequence|seq)$|"
    r"(?:^|_)(?:id|idx|name|names|smiles|smi)(?:$|_)",
    re.I,
)
# A SMILES string is a run of these tokens and nothing else: bracket atoms,
# organic-subset atoms, bonds, branches, ring closures, and dots. English
# words fail ("ethanol" has e, t, h, a, and l), even though n and o are atoms.
_SMILES = re.compile(
    r"(?:\[[^\[\]]+\]|Br|Cl|[BCNOPSFI]|[bcnops]|[-=#$:/\\.()*]|%\d{2}|\d)+"
)
_SMILES_ATOMS = re.compile(r"Cl|Br|[BCNOPSFI]|[cnops]|\[")
_PROTEIN = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
_NUCLEIC = set("ACGTUN")


# One atom of a SMILES string: a bracket atom, or an organic-subset symbol.
_SMILES_ATOM = re.compile(
    r"\[(?:\d+)?(?P<element>[A-Z][a-z]?|[a-z]{1,2})[^\]]*\]|Br|Cl|[BCNOSPFI]|[bcnops]"
)
MAX_ITEMS = 10_000


def heavy_atoms(smiles: str) -> int:
    """Non-hydrogen atoms in a SMILES string, counted without RDKit."""
    return sum(
        1
        for match in _SMILES_ATOM.finditer(smiles)
        if (match.group("element") or "").lower() != "h"
    )


def is_smiles(value: str) -> bool:
    """Whether ``value`` reads as one SMILES string (a syntax check, not RDKit)."""
    value = value.strip()
    return (
        0 < len(value) <= 400
        and bool(_SMILES.fullmatch(value))
        and bool(_SMILES_ATOMS.search(value))
        and value.count("(") == value.count(")")
    )


def _sequence_alphabet(value: str) -> str | None:
    letters = set(value.strip().upper())
    if len(value.strip()) < 10 or not letters <= _PROTEIN:
        return None
    return "nucleic acid" if letters <= _NUCLEIC else "protein"


def _stats(array: Any) -> tuple[float | None, float | None]:
    """Minimum and maximum of a bounded sample of a numeric array."""
    try:
        flat = np.asarray(array).reshape(-1)
    except Exception:  # noqa: BLE001 - object or ragged arrays
        return None, None
    if flat.size == 0 or not np.issubdtype(flat.dtype, np.number):
        return None, None
    sample = flat[:: max(1, flat.size // MAX_SAMPLE)][:MAX_SAMPLE]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return None, None
    return float(sample.min()), float(sample.max())


def _decode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _decode(value.reshape(-1)[0])
    text = str(value).strip()
    return text or None


def _spacing(values: Any) -> float | None:
    """Median step of a 1D coordinate, or None when it is not monotonic."""
    try:
        coord = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if coord.size < 2:
        return None
    steps = np.diff(coord[:MAX_SAMPLE])
    if not (np.all(steps > 0) or np.all(steps < 0)):
        return None
    return float(np.median(np.abs(steps)))


def _fmt(value: float) -> str:
    return f"{value:.6g}"


_TIME_UNIT = re.compile(r"^\s*(seconds?|minutes?|hours?|days?)\s+since\b", re.I)


def _time_step(values: Any, units: str | None) -> str | None:
    step = _spacing(values)
    if step is None:
        return None
    match = _TIME_UNIT.match(units or "")
    unit = match.group(1).lower().rstrip("s") + "s" if match else None
    if unit == "minutes" and step % 60 == 0:
        step, unit = step / 60, "hours"
    if unit == "seconds" and step % 3600 == 0:
        step, unit = step / 3600, "hours"
    if unit == "hours" and step % 24 == 0 and step >= 24:
        step, unit = step / 24, "days"
    if unit and step == 1:
        unit = unit.rstrip("s")
    return f"{_fmt(step)} {unit}" if unit else _fmt(step)


def _finish_gridded(profile: DataProfile, coords: dict[str, Any]) -> None:
    """Derive grid, resolution, and time spacing from named coordinates."""
    names = {name.lower(): name for name in coords}
    lat = next((names[n] for n in names if n in _LAT), None)
    lon = next((names[n] for n in names if n in _LON), None)
    if lat and lon:
        lat_step = _spacing(coords[lat]["values"])
        lon_step = _spacing(coords[lon]["values"])
        profile.grid = [coords[lat]["size"], coords[lon]["size"]]
        if lat_step and lon_step and abs(lat_step - lon_step) < 1e-6:
            profile.resolution = f"{_fmt(lat_step)} deg"
        elif lat_step and lon_step:
            profile.resolution = f"{_fmt(lat_step)} x {_fmt(lon_step)} deg"
    time = next((names[n] for n in names if n in _TIME), None)
    if time:
        profile.time_steps = coords[time]["size"]
        profile.time_step = _time_step(
            coords[time]["values"], coords[time].get("units")
        )
    spatial = set()
    for variable in profile.variables:
        dims = [d.lower() for d in variable.dims]
        spatial.update(d for d in dims if d in _LAT | _LON | _SPACE)
        if any(d in _LEVEL for d in dims) and len(variable.shape) >= 3:
            level = next(d for d in variable.dims if d.lower() in _LEVEL)
            if level in coords:
                variable.levels = [
                    float(v)
                    for v in np.asarray(coords[level]["values"]).reshape(-1)[
                        :200
                    ]
                ]
    if spatial:
        profile.spatial_dims = len({
            ("lat" if d in _LAT | {"y"} else "lon" if d in _LON | {"x"} else d)
            for d in spatial
        })
        profile.kind = "field"
        if not profile.grid:
            sizes = {
                d.lower(): s
                for variable in profile.variables
                for d, s in zip(variable.dims, variable.shape)
            }
            profile.grid = [sizes[d] for d in ("y", "x") if d in sizes] or [
                sizes[d] for d in sorted(spatial) if d in sizes
            ]


# HDF5 and NetCDF ---------------------------------------------------------------


def _read_hdf5(path: Path) -> DataProfile:
    import h5py

    profile = DataProfile(
        path=str(path),
        format="NetCDF4" if path.suffix.lower() in {".nc", ".nc4"} else "HDF5",
        reader="h5py",
    )
    with h5py.File(path, "r") as handle:
        attrs = {k: _decode(v) for k, v in handle.attrs.items()}
        if "t0_fields" in handle or {"dataset_name", "n_spatial_dims"} <= set(
            attrs
        ):
            return _read_well(handle, attrs, profile)
        coords: dict[str, Any] = {}
        datasets: list[tuple[str, Any]] = []

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, obj))

        handle.visititems(visit)
        for name, dataset in datasets:
            base = name.split("/")[-1]
            units = _decode(dataset.attrs.get("units"))
            is_scale = _decode(dataset.attrs.get("CLASS")) == "DIMENSION_SCALE"
            if dataset.ndim == 1 and (
                is_scale
                or base.lower() in _LAT | _LON | _TIME | _LEVEL | _SPACE
            ):
                coords[base] = {
                    "size": int(dataset.shape[0]),
                    "values": dataset[: min(dataset.shape[0], MAX_SAMPLE)],
                    "units": units,
                }
                continue
            dims = []
            for axis in range(dataset.ndim):
                try:
                    scales = [
                        s.name.split("/")[-1]
                        for s in dataset.dims[axis].values()
                    ]
                except Exception:  # noqa: BLE001 - files without dimension scales
                    scales = []
                dims.append(scales[0] if scales else f"dim_{axis}")
            low, high = _stats(_bounded(dataset))
            profile.variables.append(
                VariableProfile(
                    name=base,
                    shape=[int(n) for n in dataset.shape],
                    dims=dims,
                    dtype=str(dataset.dtype),
                    units=units,
                    long_name=_decode(dataset.attrs.get("long_name")),
                    minimum=low,
                    maximum=high,
                )
            )
    profile.kind = "tensor" if profile.variables else "other"
    _finish_gridded(profile, coords)
    _label_arrays(profile)
    return profile


def _bounded(dataset: Any) -> Any:
    """Read at most about MAX_SAMPLE values from an array-like dataset."""
    if dataset.size <= MAX_SAMPLE:
        return dataset[()]
    step = max(
        1,
        dataset.shape[0]
        // max(1, MAX_SAMPLE // max(1, dataset.size // dataset.shape[0])),
    )
    return dataset[::step]


def _read_well(
    handle: Any, attrs: dict[str, Any], profile: DataProfile
) -> DataProfile:
    """Read a file in The Well HDF5 layout, which NOMAD's WellFormat loads."""
    profile.representation = "The Well HDF5"
    profile.kind = "field"
    if attrs.get("n_spatial_dims"):
        profile.spatial_dims = int(float(attrs["n_spatial_dims"]))
    if attrs.get("dataset_name"):
        profile.notes.append(f"Dataset name: {attrs['dataset_name']}.")
    if attrs.get("grid_type"):
        profile.notes.append(f"Grid type: {attrs['grid_type']}.")
    for group in ("t0_fields", "t1_fields", "t2_fields"):
        if group not in handle:
            continue
        for name, dataset in handle[group].items():
            if not hasattr(dataset, "shape"):
                continue
            low, high = _stats(_bounded(dataset))
            profile.variables.append(
                VariableProfile(
                    name=name,
                    shape=[int(n) for n in dataset.shape],
                    dtype=str(dataset.dtype),
                    units=_decode(dataset.attrs.get("units")),
                    minimum=low,
                    maximum=high,
                )
            )
    dims = handle.get("dimensions")
    if dims is not None:
        sizes = {
            name: int(ds.shape[0])
            for name, ds in dims.items()
            if hasattr(ds, "shape") and len(ds.shape) == 1
        }
        if "time" in sizes:
            profile.time_steps = sizes["time"]
            profile.time_step = _time_step(dims["time"][()], None)
        profile.grid = [sizes[n] for n in ("x", "y", "z") if n in sizes]
    shape = profile.variables[0].shape if profile.variables else []
    if shape:
        # The Well stores each field as (trajectory, time, *space).
        profile.n_samples = shape[0]
        n = profile.spatial_dims or max(0, len(shape) - 2)
        if not profile.grid and len(shape) >= 2 + n:
            profile.grid = shape[2 : 2 + n]
        if profile.time_steps is None and len(shape) >= 2:
            profile.time_steps = shape[1]
    return profile


def _read_netcdf3(path: Path) -> DataProfile:
    from scipy.io import netcdf_file

    profile = DataProfile(path=str(path), format="NetCDF3", reader="scipy")
    coords: dict[str, Any] = {}
    with netcdf_file(str(path), "r", mmap=False) as handle:
        for name, variable in handle.variables.items():
            units = _decode(getattr(variable, "units", None))
            dims = list(variable.dimensions)
            if len(dims) == 1 and dims[0] == name:
                coords[name] = {
                    "size": int(variable.shape[0]),
                    "values": np.array(variable.data[:MAX_SAMPLE]),
                    "units": units,
                }
                continue
            low, high = _stats(np.array(variable.data))
            profile.variables.append(
                VariableProfile(
                    name=name,
                    shape=[int(n) for n in variable.shape],
                    dims=dims,
                    dtype=str(variable.data.dtype),
                    units=units,
                    long_name=_decode(getattr(variable, "long_name", None)),
                    minimum=low,
                    maximum=high,
                )
            )
    profile.kind = "tensor" if profile.variables else "other"
    _finish_gridded(profile, coords)
    return profile


# NumPy and MATLAB ----------------------------------------------------------------


def _array_variable(name: str, array: Any) -> VariableProfile:
    low, high = _stats(array)
    return VariableProfile(
        name=name,
        shape=[int(n) for n in np.shape(array)],
        dtype=str(getattr(array, "dtype", "")) or None,
        minimum=low,
        maximum=high,
    )


def _label_arrays(profile: DataProfile) -> None:
    profile.labels = [
        v.name for v in profile.variables if _LABEL_NAMES.match(v.name)
    ]


def _read_numpy(path: Path) -> DataProfile:
    profile = DataProfile(path=str(path), format="NumPy", reader="numpy")
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        profile.variables.append(_array_variable(path.stem, array))
    else:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                profile.variables.append(_array_variable(name, archive[name]))
    _shape_kind(profile)
    _label_arrays(profile)
    return profile


def _read_matlab(path: Path) -> DataProfile:
    from scipy.io import whosmat

    profile = DataProfile(path=str(path), format="MATLAB", reader="scipy")
    for name, shape, dtype in whosmat(str(path)):
        if not name.startswith("__"):
            profile.variables.append(
                VariableProfile(name=name, shape=list(shape), dtype=dtype)
            )
    _shape_kind(profile)
    _label_arrays(profile)
    return profile


def _shape_kind(profile: DataProfile) -> None:
    """Guess the data kind from array shapes when nothing names the axes."""
    ndims = [len(v.shape) for v in profile.variables if v.shape]
    if not ndims:
        return
    largest = max(ndims)
    profile.kind = "time_series" if largest == 1 else "tensor"
    if largest >= 2:
        profile.notes.append(
            "Axis names are not stored in this format; the spatial grid is not known."
        )


# Tables ---------------------------------------------------------------------------


def _read_table(path: Path) -> DataProfile:
    import pandas as pd

    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    frame = pd.read_csv(path, sep=sep, nrows=5000)
    with path.open("rb") as handle:
        rows = max(0, sum(1 for _ in handle) - 1)
    profile = DataProfile(
        path=str(path),
        format="TSV" if sep == "\t" else "CSV",
        reader="pandas",
        kind="table",
        n_samples=rows,
    )
    smiles_col = sequence_col = None
    for column in frame.columns:
        series = frame[column].dropna()
        numeric = pd.api.types.is_numeric_dtype(series)
        low, high = _stats(series.to_numpy()) if numeric else (None, None)
        profile.variables.append(
            VariableProfile(
                name=str(column),
                shape=[rows],
                dtype=str(frame[column].dtype),
                minimum=low,
                maximum=high,
            )
        )
        if numeric or series.empty:
            continue
        values = [str(v) for v in series.head(200)]
        if smiles_col is None and (
            str(column).lower()
            in {"smiles", "smi", "canonical_smiles", "isomeric_smiles"}
            or sum(is_smiles(v) for v in values) >= 0.8 * len(values)
        ):
            smiles_col = str(column)
        elif sequence_col is None and sum(
            _sequence_alphabet(v) == "protein" for v in values
        ) >= 0.8 * len(values):
            sequence_col = str(column)
    if smiles_col:
        profile.kind, profile.representation = "molecule", "SMILES"
        profile.examples = [str(v) for v in frame[smiles_col].dropna().head(3)]
        profile.item_sizes["heavy atoms"] = [
            heavy_atoms(str(v))
            for v in frame[smiles_col].dropna().head(MAX_ITEMS)
        ]
        profile.notes.append(f"Molecules are in column '{smiles_col}'.")
    elif sequence_col:
        profile.kind, profile.representation = "sequence", "protein sequence"
        profile.notes.append(f"Sequences are in column '{sequence_col}'.")
    key = smiles_col or sequence_col
    profile.column = key
    profile.labels = [
        v.name
        for v in profile.variables
        if v.name != key
        and v.minimum is not None
        and not _ID_NAMES.search(v.name)
        and frame[v.name].nunique() > 1
        and not _counts_rows(frame[v.name])
    ]
    return profile


def _counts_rows(series: Any) -> bool:
    """Whether a column numbers the rows (0, 1, 2, ... or 1, 2, 3, ...)."""
    values = series.dropna().to_numpy()
    if len(values) < 2 or not np.all(np.mod(values, 1) == 0):
        return False
    steps = np.diff(values)
    return bool(np.all(steps == 1) and values[0] in (0, 1))


# Molecules, sequences, and structures -------------------------------------------


def _text_lines(path: Path) -> list[str]:
    if path.stat().st_size > MAX_TEXT_BYTES:
        with path.open("r", errors="replace", encoding="utf-8") as handle:
            return [next(handle) for _ in range(100_000)]
    return path.read_text(errors="replace", encoding="utf-8").splitlines()


def _read_smiles(path: Path) -> DataProfile:
    lines = [line.split()[0] for line in _text_lines(path) if line.strip()]
    if lines and lines[0].lower() in {"smiles", "smi"}:
        lines = lines[1:]
    valid = [line for line in lines if is_smiles(line)]
    profile = DataProfile(
        path=str(path),
        format="SMILES list",
        reader="text",
        kind="molecule" if valid else "other",
        representation="SMILES" if valid else None,
        n_samples=len(lines),
        examples=valid[:3],
    )
    if valid:
        profile.item_sizes["heavy atoms"] = [
            heavy_atoms(v) for v in valid[:MAX_ITEMS]
        ]
    if len(valid) < len(lines):
        profile.notes.append(
            f"{len(lines) - len(valid)} of {len(lines)} lines do not look like SMILES."
        )
    return profile


def _read_fasta(path: Path) -> DataProfile:
    sequences: list[str] = []
    current: list[str] = []
    for line in _text_lines(path):
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
            current = []
        elif line.strip():
            current.append(line.strip())
    if current:
        sequences.append("".join(current))
    alphabets = {_sequence_alphabet(s) for s in sequences[:500]}
    kind = "protein" if "protein" in alphabets else "nucleic acid"
    lengths = [len(s) for s in sequences]
    profile = DataProfile(
        path=str(path),
        format="FASTA",
        reader="text",
        kind="sequence",
        representation=f"{kind} sequence",
        n_samples=len(sequences),
        examples=[s[:40] + ("…" if len(s) > 40 else "") for s in sequences[:2]],
    )
    if lengths:
        profile.notes.append(
            f"Sequence lengths range from {min(lengths)} to {max(lengths)}."
        )
        profile.item_sizes["residues"] = lengths[:MAX_ITEMS]
    return profile


def _read_xyz(path: Path) -> DataProfile:
    lines = _text_lines(path)
    frames, elements, index, periodic = 0, set(), 0, False
    sizes = []
    while index < len(lines):
        try:
            count = int(lines[index].split()[0])
        except (ValueError, IndexError):
            break
        comment = lines[index + 1] if index + 1 < len(lines) else ""
        periodic = periodic or "Lattice=" in comment
        for row in lines[index + 2 : index + 2 + count]:
            if row.split():
                elements.add(row.split()[0])
        frames += 1
        sizes.append(count)
        index += count + 2
    profile = DataProfile(
        path=str(path),
        format="extended XYZ" if periodic else "XYZ",
        reader="text",
        kind="structure",
        representation="atomic structure",
        n_samples=frames,
    )
    if elements:
        profile.notes.append(f"Elements: {', '.join(sorted(elements))}.")
    if sizes:
        profile.notes.append(
            f"{min(sizes)} to {max(sizes)} atoms per structure"
            + ("; periodic cells." if periodic else ".")
        )
    return profile


def _read_cif(path: Path) -> DataProfile:
    text = "\n".join(_text_lines(path))
    blocks = len(re.findall(r"^data_", text, re.M))
    formula = re.search(r"_chemical_formula_sum\s+'?([^'\n]+)'?", text)
    group = re.search(r"_symmetry_space_group_name_H-M\s+'?([^'\n]+)'?", text)
    profile = DataProfile(
        path=str(path),
        format="CIF",
        reader="text",
        kind="structure",
        representation="crystal structure",
        n_samples=blocks or 1,
    )
    if formula:
        profile.notes.append(f"Formula: {formula.group(1).strip()}.")
    if group:
        profile.notes.append(f"Space group: {group.group(1).strip()}.")
    return profile


def _read_pdb(path: Path) -> DataProfile:
    atoms = residues = 0
    chains: set[str] = set()
    for line in _text_lines(path):
        if line.startswith(("ATOM", "HETATM")):
            atoms += 1
            if line[12:16].strip() == "CA":
                residues += 1
            if len(line) > 21 and line[21].strip():
                chains.add(line[21])
    return DataProfile(
        path=str(path),
        format="PDB",
        reader="text",
        kind="structure",
        representation="protein structure"
        if residues
        else "molecular structure",
        n_samples=1,
        notes=[f"{atoms} atoms, {residues} residues, {len(chains)} chain(s)."],
    )


# Images and JSON -------------------------------------------------------------------


def _read_image(path: Path) -> DataProfile:
    from PIL import Image

    with Image.open(path) as image:
        frames = getattr(image, "n_frames", 1)
        profile = DataProfile(
            path=str(path),
            format=image.format or path.suffix.lstrip(".").upper(),
            reader="Pillow",
            kind="image",
            spatial_dims=2,
            grid=[image.height, image.width],
            n_samples=frames,
        )
        profile.variables.append(
            VariableProfile(
                name="bands",
                shape=[image.height, image.width, len(image.getbands())],
                dtype=image.mode,
            )
        )
    return profile


def _read_json(path: Path) -> DataProfile:
    if path.stat().st_size > MAX_TEXT_BYTES:
        return DataProfile(
            path=str(path),
            format="JSON",
            reader="json",
            notes=["The file is too large to inspect as JSON."],
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = DataProfile(path=str(path), format="JSON", reader="json")
    if isinstance(data, dict) and "t0_fields" in data:
        profile.kind, profile.representation = "field", "WellFormat JSON"
        profile.spatial_dims = data.get("n_spatial_dims")
        for group in ("t0_fields", "t1_fields", "t2_fields"):
            for name in data.get(group) or {}:
                profile.variables.append(VariableProfile(name=name))
    elif (
        isinstance(data, list)
        and data
        and all(isinstance(v, str) for v in data)
    ):
        valid = [v for v in data if is_smiles(v)]
        if len(valid) >= 0.8 * len(data):
            profile.kind, profile.representation = "molecule", "SMILES"
            profile.examples = valid[:3]
            profile.item_sizes["heavy atoms"] = [
                heavy_atoms(v) for v in valid[:MAX_ITEMS]
            ]
        profile.n_samples = len(data)
    elif isinstance(data, dict):
        for name, value in data.items():
            try:
                array = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                continue
            profile.variables.append(_array_variable(name, array))
        _shape_kind(profile)
        _label_arrays(profile)
    return profile


# Dispatch -----------------------------------------------------------------------------

_BY_SUFFIX = {
    ".h5": _read_hdf5,
    ".hdf5": _read_hdf5,
    ".he5": _read_hdf5,
    ".nc": None,  # NetCDF3 or NetCDF4: decided by magic bytes
    ".nc4": _read_hdf5,
    ".npy": _read_numpy,
    ".npz": _read_numpy,
    ".mat": _read_matlab,
    ".csv": _read_table,
    ".tsv": _read_table,
    ".smi": _read_smiles,
    ".smiles": _read_smiles,
    ".fasta": _read_fasta,
    ".fa": _read_fasta,
    ".faa": _read_fasta,
    ".xyz": _read_xyz,
    ".extxyz": _read_xyz,
    ".cif": _read_cif,
    ".pdb": _read_pdb,
    ".png": _read_image,
    ".jpg": _read_image,
    ".jpeg": _read_image,
    ".tif": _read_image,
    ".tiff": _read_image,
    ".json": _read_json,
}
_NEEDS = {
    _read_hdf5: "h5py",
    _read_table: "pandas",
    _read_netcdf3: "scipy",
    _read_matlab: "scipy",
    _read_image: "Pillow",
}
SUPPORTED_SUFFIXES = tuple(sorted(_BY_SUFFIX))


def inspect_data(path: str | Path) -> DataProfile:
    """Describe the data in ``path``; unreadable files return a profile with notes."""
    path = Path(path).expanduser()
    if not path.exists():
        return DataProfile(
            path=str(path),
            format="missing",
            reader="none",
            notes=["The file does not exist."],
        )
    if path.is_dir():
        return DataProfile(
            path=str(path),
            format="directory",
            reader="none",
            notes=[
                "Directories (such as Zarr stores) are not inspected yet; name a file."
            ],
        )
    suffix = path.suffix.lower()
    reader = _BY_SUFFIX.get(suffix, _read_unknown)
    if suffix == ".nc" or reader is None:
        with path.open("rb") as handle:
            magic = handle.read(8)
        reader = _read_hdf5 if magic.startswith(_HDF5_MAGIC) else _read_netcdf3
    try:
        return reader(path)
    except ImportError as exc:
        return DataProfile(
            path=str(path),
            format=suffix.lstrip(".").upper() or "unknown",
            reader="none",
            notes=[
                f"Install {_NEEDS.get(reader, exc.name or 'the reader library')} "
                f"to inspect this file ({exc})."
            ],
        )
    except Exception as exc:  # noqa: BLE001 - report unreadable files, never crash
        return DataProfile(
            path=str(path),
            format=suffix.lstrip(".").upper() or "unknown",
            reader="none",
            notes=[f"The file could not be read: {type(exc).__name__}: {exc}"],
        )


def _read_unknown(path: Path) -> DataProfile:
    with path.open("rb") as handle:
        magic = handle.read(8)
    if magic.startswith(_HDF5_MAGIC):
        return _read_hdf5(path)
    if magic[:4] in _NETCDF3_MAGIC:
        return _read_netcdf3(path)
    return DataProfile(
        path=str(path),
        format=path.suffix.lstrip(".").upper() or "unknown",
        reader="none",
        notes=[
            "This format is not supported yet. Supported: "
            + ", ".join(SUPPORTED_SUFFIXES)
            + "."
        ],
    )


def read_items(
    profile: DataProfile, limit: int = 1000
) -> list[tuple[str, str | None]]:
    """The molecules or sequences in a profiled file, in file order.

    Each item is ``(value, name)``: a SMILES string or a sequence, with the
    name the file gives it ("CCO ethanol" or a FASTA header), when it gives
    one. At most ``limit`` items are read. Other kinds of data return an
    empty list, since a model takes them as whole arrays, not item by item.
    """
    path = Path(profile.path)
    items: list[tuple[str, str | None]] = []
    if profile.kind == "molecule" and profile.format == "SMILES list":
        for line in _text_lines(path):
            parts = line.split(maxsplit=1)
            if (
                parts
                and is_smiles(parts[0])
                and parts[0].lower() not in {"smiles", "smi"}
            ):
                items.append((
                    parts[0],
                    parts[1].strip() if len(parts) > 1 else None,
                ))
    elif profile.kind == "sequence" and profile.format == "FASTA":
        name, current = None, []
        for line in _text_lines(path) + [">"]:
            if line.startswith(">"):
                if current:
                    items.append(("".join(current), name))
                name, current = line[1:].strip() or None, []
            elif line.strip():
                current.append(line.strip())
    elif profile.column and profile.format in {"CSV", "TSV"}:
        import pandas as pd

        sep = "\t" if profile.format == "TSV" else ","
        frame = pd.read_csv(path, sep=sep, nrows=limit)
        names = next(
            (
                c
                for c in frame.columns
                if str(c).lower() in {"name", "names", "molecule", "compound"}
            ),
            None,
        )
        for _, row in frame.iterrows():
            value = row[profile.column]
            if isinstance(value, str) and value.strip():
                label = row[names] if names is not None else None
                items.append((
                    value.strip(),
                    str(label) if label is not None else None,
                ))
    elif profile.kind == "molecule" and profile.format == "JSON":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [(v, None) for v in data if isinstance(v, str) and is_smiles(v)]
    return items[:limit]


def variable_names(profile: DataProfile) -> list[str]:
    """Variables a model input could map to, in file order."""
    return [v.name for v in profile.variables]


def profiles_text(profiles: Iterable[DataProfile]) -> str:
    """Flatten profiles into words for lexical matching."""
    parts: list[str] = []
    for profile in profiles:
        parts += [profile.kind, profile.representation or "", profile.format]
        parts += [v.name for v in profile.variables]
        parts += [v.long_name or "" for v in profile.variables]
    return " ".join(p for p in parts if p)
