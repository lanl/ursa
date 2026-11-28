"""Regular expressions and small parsers for model-card facts.

Everything here is a pure function of text, so each pattern can be unit
tested on its own. Patterns favor precision: a missed fact surfaces as a gap
that the LLM pass or the user can fill, while a wrong fact misleads the
recommendation.
"""

from __future__ import annotations

import re

NUMBER = r"\d+(?:[.,]\d+)*"
_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
}

PARAM_TEXT = (
    re.compile(
        rf"(?<![\w.])(?P<num>{NUMBER})\s*(?P<unit>[KMBT]|k|thousand|million|"
        r"billion|trillion)?\s*[- ]?\s*param(?:eter)?s?\b",
        re.I,
    ),
    re.compile(
        r"\bparam(?:eter)?s?(?:\s+count)?\s*[:=]\s*`?\s*"
        rf"(?P<num>{NUMBER})\s*(?P<unit>[KMBT]|million|billion)?\b",
        re.I,
    ),
)
PARAM_NAME = re.compile(
    r"(?<![A-Za-z0-9.])(?P<num>\d+(?:\.\d+)?)(?P<unit>[MBK])(?![A-Za-z0-9])"
)
TUPLE_SHAPE = re.compile(
    r"\(\s*(?P<dims>[A-Za-z0-9_][A-Za-z0-9_ ]*"
    r"(?:\s*[,×]\s*[A-Za-z0-9_][A-Za-z0-9_ ]*){1,7})\s*\)"
)
GRID = re.compile(
    r"(?<![\w.])(\d{1,5})\s*[x×]\s*(\d{1,5})(?:\s*[x×]\s*(\d{1,5}))?(?![\w.])"
)
DEGREES = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°|deg(?:ree)?s?\b)", re.I)
KM_RESOLUTION = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*km\s+(?:resolution|grid|spacing|horizontal)",
    re.I,
)
DT = re.compile(r"\b(?:dt|Δt|delta[ _]t)\s*[=:]\s*([0-9.eE+-]+\s*\w*)", re.I)
SAMPLING = re.compile(r"(\d+(?:\.\d+)?)\s*(Hz|kHz|MHz|samples per second)\b")
DTYPE = re.compile(
    r"\b(float16|float32|float64|bfloat16|bf16|fp16|fp32|fp64|int8|int16|"
    r"int32|uint8|half precision|single precision|double precision)\b",
    re.I,
)
GPU = re.compile(
    r"\b(A100|H100|H200|B200|GH200|V100|P100|A6000|A40|A10G?|L40S?|L4|T4|"
    r"RTX\s?\d{4}(?:\s?Ti)?|MI250X?|MI300X?|TPU\s?v\d\w*)\b"
)
GIGABYTES = re.compile(r"(\d+(?:\.\d+)?)\s*(?:GB|GiB)\b")
CPU_OK = re.compile(
    r"can (?:also )?(?:be )?run on (?:a |the )?cpu|cpu can run|"
    r"runs? on (?:a |the )?cpu|cpu inference is supported|"
    r"supports? cpu|on (?:a )?cpu or (?:a )?(?:single )?gpu|"
    r"(?:cpu|gpu) or (?:a )?(?:cpu|gpu)\b|without a gpu|no gpu (?:is )?required",
    re.I,
)
CPU_NO = re.compile(
    r"requires? (?:an? )?(?:nvidia |cuda[- ]capable )?gpu|gpu is required|"
    r"cuda is required|does not (?:support|run on) (?:a )?cpu|"
    r"not supported on cpu",
    re.I,
)
COMPARATIVE = re.compile(
    r"compared (?:to|with)|faster than|than (?:a )?cpu", re.I
)
MULTI_GPU = re.compile(
    r"requires? (?:multiple|several|\d+) gpus|model[- ]parallel|"
    r"multi-gpu (?:inference )?(?:is )?required",
    re.I,
)
RESTRICTION = re.compile(
    r"research (?:use|purposes) only|for research purposes only|"
    r"non-?commercial|no commercial use|not for commercial use|"
    r"commercial use (?:is )?(?:not permitted|prohibited|requires)|"
    r"academic use only|no redistribution|do not redistribute|"
    r"export[- ]controlled",
    re.I,
)
NONCOMMERCIAL = re.compile(
    r"research (?:use|purposes) only|non-?commercial|no commercial use|"
    r"not for commercial use|commercial use (?:is )?(?:not permitted|"
    r"prohibited|requires)|academic use only",
    re.I,
)
# "ready for commercial/non-commercial use" grants both uses.
BOTH_USES = re.compile(
    r"commercial\s*(?:/|and|or)\s*non-?commercial|"
    r"non-?commercial\s*(?:/|and|or)\s*commercial|ready for commercial",
    re.I,
)
INSTALL = re.compile(
    r"(?:^|\s|`)((?:pip3?|uv pip|uv|conda|mamba|micromamba)\s+"
    r"(?:install|add)\s+[^`\n]+|git clone\s+[^`\n]+)",
    re.M,
)
LOAD_CALL = re.compile(r"\b([A-Z]\w*)\.from_pretrained\(")
IMPORT_FROM = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+(.+)$", re.M)
CLI_CALL = re.compile(
    r"(?:^|`)\s*(?:\$\s*)?(python3?\s+(?:-m\s+[\w.]+|[\w./-]+\.py))\b", re.M
)
# Dimensionality of the physical problem, not of an architecture component
# such as a "3D patch embedding".
SPATIAL_DIMS = re.compile(
    r"\b(?P<n>[123])-?d\b\s+(?:\w+[\s-]+){0,3}?(?:fields?|domains?|grids?|"
    r"data(?:sets?)?|problems?|systems?|simulations?|flows?|pdes?|equations?|"
    r"turbulence|geometr(?:y|ies)|volumes?|images?|slices?|spaces?|meshes|"
    r"dynamics|fluids?|cfd|surrogates?)\b|"
    r"\b(?P<word>one|two|three)-dimensional\s+(?:\w+\s+){0,2}?(?:fields?|"
    r"domains?|grids?|problems?|systems?|simulations?|flows?|spaces?|"
    r"pdes?|equations?|dynamics|turbulence)\b",
    re.I,
)
NOT_CLI = re.compile(r"python3?\s+-m\s+(?:venv|pip|ensurepip|http\.server)\b")
IO_DTYPE = re.compile(
    r"\b(?:inputs?|outputs?)\b[^.;]{0,60}?\b(float16|float32|float64|"
    r"bfloat16|bf16|fp16|fp32|fp64|int8|uint8)\b",
    re.I,
)
HARDWARE_REQUIREMENT = re.compile(
    r"\binference\b|\brequires?\b|\brecommended\b|\bminimum\b|"
    r"\bat least\b|\bmemory\b|\bvram\b|\bfits? (?:on|in)\b|"
    r"\bruns? on\b|\bcan run\b|\bdeploy",
    re.I,
)
GAUSSIAN_GRID = re.compile(r"\b([NO]\d{2,4})\b")
# "30 m resolution", "~2.5 km resolution", "high spatial resolution (30m)".
LENGTH_RESOLUTION = re.compile(
    r"(?P<v1>\d+(?:\.\d+)?)\s*-?\s*(?P<u1>m|km)\b(?=\s*(?:spatial\s+)?"
    r"(?:resolution|gsd|pixels?|grid spacing))|"
    r"(?:resolution|gsd)\s*(?:of\s*)?\(?\s*~?(?P<v2>\d+(?:\.\d+)?)\s*"
    r"(?P<u2>m|km)\b",
    re.I,
)
BACKTICK = re.compile(r"`([^`\n]+)`")
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,40}$")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
VERSION_IN_NAME = re.compile(r"(?:^|[-_ ])v?(\d+\.\d+(?:\.\d+)?)(?=$|[-_ ])")
UNIT_WORDS = (
    "K|°C|degC|Pa|hPa|kPa|MPa|GPa|m/s|m s-1|km/s|m/s2|kg/m3|kg m-3|g/cm3|"
    "eV|meV|eV/Å|eV/A|hartree|Hartree|Ha|debye|Debye|bohr|Bohr|Å|angstrom|"
    "kcal/mol|kJ/mol|J|W/m2|W m-2|mm|cm|km|Hz|kHz|ppm|psu|dB|nm|μm|um|"
    "Mpc|keV|MeV|GeV|TeV|Gauss|mm/day|kg/kg"
)
UNIT_CONTEXT = re.compile(
    rf"(?:\bunits?\s*(?:of|in|:|=)\s*|\bmeasured in\s+|\bin units of\s+)"
    rf"(?P<unit>{UNIT_WORDS})(?![A-Za-z])"
)
# Spelled-out units and their symbols. Cards and tool schemas often write
# "temperature in kelvin" or "length in meters" instead of a symbol.
UNIT_NAME_SYMBOLS = {
    "degrees celsius": "°C",
    "degree celsius": "°C",
    "celsius": "°C",
    "degrees fahrenheit": "°F",
    "fahrenheit": "°F",
    "kelvin": "K",
    "kelvins": "K",
    "meters": "m",
    "meter": "m",
    "metres": "m",
    "metre": "m",
    "kilometers": "km",
    "kilometres": "km",
    "centimeters": "cm",
    "centimetres": "cm",
    "millimeters": "mm",
    "millimetres": "mm",
    "micrometers": "μm",
    "microns": "μm",
    "nanometers": "nm",
    "angstroms": "Å",
    "angstrom": "Å",
    "seconds": "s",
    "milliseconds": "ms",
    "minutes": "min",
    "hours": "h",
    "days": "day",
    "years": "yr",
    "pascals": "Pa",
    "pascal": "Pa",
    "hectopascals": "hPa",
    "kilopascals": "kPa",
    "megapascals": "MPa",
    "gigapascals": "GPa",
    "millibars": "mbar",
    "bars": "bar",
    "atmospheres": "atm",
    "joules": "J",
    "kilojoules": "kJ",
    "electronvolts": "eV",
    "electron volts": "eV",
    "hartrees": "Ha",
    "kilograms": "kg",
    "grams": "g",
    "newtons": "N",
    "watts": "W",
    "volts": "V",
    "amperes": "A",
    "teslas": "T",
    "tesla": "T",
    "hertz": "Hz",
    "moles": "mol",
    "radians": "rad",
    "degrees": "deg",
}
_UNIT_NAMES_IN = re.compile(
    r"\bin\s+(?:units\s+of\s+)?(?P<unit>"
    + "|".join(
        n.replace(" ", r"[\s-]+")
        for n in sorted(UNIT_NAME_SYMBOLS, key=len, reverse=True)
    )
    + r")\b",
    re.I,
)
# Compound symbols such as m^2/s, kg/m3, mm/day, or m s-1. Every factor must
# be a known unit, so "x/y" or "train/test" never reads as a unit.
_UNIT_ATOM = (
    r"(?:kg|mg|km|cm|mm|μm|um|nm|ms|min|hr|day|yr|°C|degC|hPa|kPa|MPa|GPa|"
    r"Pa|bar|atm|kJ|kcal|meV|keV|MeV|GeV|eV|Ha|kW|W|N|V|T|Hz|kHz|MHz|mol|"
    r"rad|sr|deg|ppm|ppb|psu|Mpc|kpc|Å|g|m|s|h|K|J|L|A|C)(?:\^?-?\d)?"
)
COMPOUND_UNIT = (
    rf"{_UNIT_ATOM}(?:\s*[/·]\s*{_UNIT_ATOM}|\s+{_UNIT_ATOM}(?<=-\d))+"
)
_UNIT_SYMBOLS_IN = re.compile(
    rf"\bin\s+(?:units\s+of\s+)?(?P<unit>{COMPOUND_UNIT}|{UNIT_WORDS})"
    r"(?![\w/^-])"
)
# "boiling point (°C)" or "pressure [hPa]": a bracket that holds only a unit.
_UNIT_BRACKETED = re.compile(
    rf"(?<=[A-Za-z]\s)[(\[]\s*(?P<unit>{COMPOUND_UNIT}|{UNIT_WORDS})\s*[)\]]"
)


def find_units(text: str) -> str | None:
    """Return the single unit a phrase states, such as 'K' for 'in kelvin'.

    A phrase that states several different units ("u in m/s and T in K")
    describes several quantities, so it yields no unit rather than a wrong one.
    """
    explicit = UNIT_CONTEXT.search(text)
    if explicit:
        return explicit.group("unit")
    found = [
        UNIT_NAME_SYMBOLS[re.sub(r"[\s-]+", " ", m.group("unit").lower())]
        for m in _UNIT_NAMES_IN.finditer(text)
    ]
    found += [m.group("unit") for m in _UNIT_SYMBOLS_IN.finditer(text)]
    found += [m.group("unit") for m in _UNIT_BRACKETED.finditer(text)]
    distinct = list(dict.fromkeys(found))
    return distinct[0] if len(distinct) == 1 else None


DIMENSIONLESS = re.compile(
    r"\bunitless\b|\bdimensionless\b|\bnormali[sz]ed units\b|"
    r"\barbitrary units\b",
    re.I,
)
NORMALIZATION = re.compile(
    r"normali[sz]|standardi[sz]|min-?max|z-?score|zero mean|unit variance|"
    r"\brms\b",
    re.I,
)
METRIC_NAMES = re.compile(
    r"^(?:v?rmse|nrmse|mse|mae|rmae|acc(?:uracy)?|f1(?:[- ]score)?|r2|r\^2|"
    r"psnr|ssim|crps|auc|roc[- ]auc|precision|recall|iou|miou|"
    r"relative l2|rel\.? ?l2|l2 error|validation loss|test loss|"
    r"best validation loss|loss|skill score|energy mae|force mae|"
    r"top-?1|top-?5)\b",
    re.I,
)
KNOWN_DATASETS = re.compile(
    r"\b(ERA5|MERRA-2|IFS HRES|CMIP6|HLS|Sentinel-2|Landsat|QM9|ZINC|"
    r"PubChem|MoleculeNet|Enamine REAL(?: Space)?|OMat24|OC20|OC22|MPtrj|"
    r"Materials Project|Alexandria|The Well|PDEBench|PDEArena|OpenFWI|"
    r"STEAD|INSTANCE|FlowBench|ImageNet|SDSS|DESI)\b"
)

# (pattern, representation, kind)
REPRESENTATIONS = (
    (re.compile(r"\bSMILES\b"), "SMILES", "molecule"),
    (re.compile(r"\bSELFIES\b"), "SELFIES", "molecule"),
    (re.compile(r"\bInChI\b"), "InChI", "molecule"),
    (
        re.compile(r"\bWellFormat\b|\bthe well format\b", re.I),
        "WellFormat",
        "field",
    ),
    (
        re.compile(r"\bNumPy array\b|\bnp\.ndarray\b|\.npz\b|\.npy\b", re.I),
        "NumPy array",
        "tensor",
    ),
    (
        re.compile(r"\b(?:PyTorch|torch) tensors?\b", re.I),
        "PyTorch tensor",
        "tensor",
    ),
    (re.compile(r"\bxarray\b", re.I), "xarray Dataset", "field"),
    (re.compile(r"\bNetCDF\b|\.nc\b", re.I), "NetCDF", "field"),
    (re.compile(r"\bGRIB2?\b", re.I), "GRIB", "field"),
    (re.compile(r"\bGeoTIFF\b|\.tiff?\b", re.I), "GeoTIFF", "image"),
    (
        re.compile(r"\bCIF\b|\bPOSCAR\b|\bextxyz\b|\bASE Atoms\b"),
        "atomic structure",
        "structure",
    ),
    (re.compile(r"\bpoint clouds?\b", re.I), "point cloud", "point_cloud"),
    (re.compile(r"\bmesh(?:es)?\b|\bgraphs?\b", re.I), "graph", "graph"),
    (
        re.compile(r"\btime[- ]series\b|\bwaveforms?\b|\bseismograms?\b", re.I),
        "time series",
        "time_series",
    ),
    (re.compile(r"\bimages?\b", re.I), "image", "image"),
    (
        re.compile(r"\b(?:protein|DNA|RNA|amino acid) sequences?\b", re.I),
        "sequence",
        "sequence",
    ),
)
FILE_FORMATS = (
    (re.compile(r"\bNetCDF\b|\.nc\b", re.I), "NetCDF"),
    (re.compile(r"\bGRIB2?\b", re.I), "GRIB"),
    (re.compile(r"\bHDF5\b|\.h5\b|\.hdf5\b", re.I), "HDF5"),
    (re.compile(r"\bzarr\b", re.I), "Zarr"),
    (re.compile(r"\.npz\b", re.I), "npz"),
    (re.compile(r"\.npy\b", re.I), "npy"),
    (re.compile(r"\bGeoTIFF\b|\.tiff?\b", re.I), "GeoTIFF"),
    (re.compile(r"\bSEG-?Y\b|\.sgy\b|\.segy\b", re.I), "SEG-Y"),
    (re.compile(r"\bmini-?SEED\b|\.mseed\b", re.I), "miniSEED"),
    (re.compile(r"\bCSV\b|\.csv\b", re.I), "CSV"),
    (re.compile(r"\bParquet\b", re.I), "Parquet"),
    (re.compile(r"\bCIF\b|\.cif\b"), "CIF"),
    (re.compile(r"\bextxyz\b|\.xyz\b", re.I), "XYZ"),
    (re.compile(r"\bPDB\b|\.pdb\b"), "PDB"),
)
# "time delta" and "input delta" describe sampling, not output semantics.
SEMANTICS = (
    (re.compile(r"\blogits?\b", re.I), "logits"),
    (re.compile(r"\bprobabilit(?:y|ies)\b", re.I), "probabilities"),
    (re.compile(r"\bclass labels?\b|\bclassification\b", re.I), "class labels"),
    (
        re.compile(r"\bembeddings?\b(?! dimension)|\bfeature vectors?\b", re.I),
        "embedding",
    ),
    (re.compile(r"\bregression\b", re.I), "regression"),
    (
        re.compile(
            r"\bresiduals?\b|\bincrements?\b|\btendenc(?:y|ies)\b|Δu|"
            r"\bpredicts? (?:the )?(?:delta|difference|change)\b",
            re.I,
        ),
        "delta",
    ),
    (
        re.compile(r"\bensemble mean\b|\bmean and standard deviation\b", re.I),
        "ensemble mean and spread",
    ),
)
NOT_NAMES = frozenset({
    "true",
    "false",
    "none",
    "null",
    "input",
    "inputs",
    "output",
    "outputs",
    "model",
    "data",
    "tensor",
    "array",
    "field",
    "fields",
})


def parse_count(number: str, unit: str | None = None) -> int | None:
    """Parse '710,867,670', '1.3' + 'B', or '26.9' + 'M' into an integer."""
    text = number.strip().replace(" ", "")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", text):
        text = text.replace(",", "")
    else:
        text = text.replace(",", ".")
    try:
        value = float(text)
    except ValueError:
        return None
    scale = {
        "k": 1e3,
        "thousand": 1e3,
        "m": 1e6,
        "million": 1e6,
        "b": 1e9,
        "billion": 1e9,
        "t": 1e12,
        "trillion": 1e12,
    }.get((unit or "").lower(), 1.0)
    count = int(round(value * scale))
    return count if 1_000 <= count <= 10**13 else None


def format_count(count: int | None) -> str | None:
    if not count:
        return None
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if count >= scale:
            return f"{count / scale:.3g}{suffix}"
    return str(count)


def parse_shape(text: str) -> list[str]:
    """Turn '(H, W)', '128 x 128', or '(batch, time, var)' into dims."""
    match = TUPLE_SHAPE.search(text or "")
    if match:
        return [d.strip() for d in re.split(r"[,×]", match.group("dims"))]
    grid = GRID.search(text or "")
    if grid:
        return [g for g in grid.groups() if g]
    return []


def word_number(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _WORD_NUMBERS.get(token)


# Time units from nanoseconds (plasma, molecular dynamics) to years (climate).
_TIME_UNIT = (
    r"(?:ns|nanoseconds?|μs|us|microseconds?|ms|milliseconds?|s|secs?|seconds?|"
    r"min(?:ute)?s?|h|hrs?|hours?|days?|weeks?|months?|years?)"
)
_UNIT_NAMES = {
    "ns": "nanoseconds",
    "nanosecond": "nanoseconds",
    "nanoseconds": "nanoseconds",
    "μs": "microseconds",
    "us": "microseconds",
    "microsecond": "microseconds",
    "microseconds": "microseconds",
    "ms": "milliseconds",
    "millisecond": "milliseconds",
    "milliseconds": "milliseconds",
    "s": "seconds",
    "sec": "seconds",
    "secs": "seconds",
    "second": "seconds",
    "seconds": "seconds",
    "min": "minutes",
    "mins": "minutes",
    "minute": "minutes",
    "minutes": "minutes",
    "h": "hours",
    "hr": "hours",
    "hrs": "hours",
    "hour": "hours",
    "hours": "hours",
    "hourly": "hours",
    "day": "days",
    "days": "days",
    "daily": "days",
    "week": "weeks",
    "weeks": "weeks",
    "month": "months",
    "months": "months",
    "year": "years",
    "years": "years",
}
_STEP_PATTERNS = (
    re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*(hourly|daily)\b", re.I),
    re.compile(
        r"\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
        rf"twelve)[- ]?({_TIME_UNIT})[- ]ahead\b",
        re.I,
    ),
    re.compile(
        r"time[- ]?step(?:\s+size)?\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*"
        rf"({_TIME_UNIT})\b",
        re.I,
    ),
    re.compile(
        rf"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*({_TIME_UNIT})\s+"
        r"(?:per\s+(?:rollout\s+)?step|(?:time[- ]?)?steps?|increments?)\b",
        re.I,
    ),
    re.compile(
        r"trained to (?:produce|predict|make) (\d+(?:\.\d+)?)\s*-?\s*"
        rf"({_TIME_UNIT})\s+(?:forecasts?|predictions?|steps?)",
        re.I,
    ),
    re.compile(
        rf"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*({_TIME_UNIT})\s+"
        r"(?:temporal )?resolution",
        re.I,
    ),
)
_HORIZON_PATTERNS = (
    re.compile(
        rf"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*({_TIME_UNIT})\s+"
        r"(?:global\s+|ahead\s+)?(?:forecasts?|rollouts?|lead times?|"
        r"horizons?|predictions?)",
        re.I,
    ),
    re.compile(
        r"lead times?\s+(?:was chosen from|of|up to|between|in|from)\s*"
        rf"(\[[^\]]{{1,40}}\]|[\d.,\s–-]{{1,20}})\s*({_TIME_UNIT})\b",
        re.I,
    ),
    re.compile(rf"up to (\d+(?:\.\d+)?)\s*({_TIME_UNIT})\b", re.I),
)
_CONTEXT_PATTERNS = (
    re.compile(
        r"context (?:length|window)[^.]{0,30}?(?:set to|of|=|:)\s*(\d+)", re.I
    ),
    re.compile(
        r"history of (\w+) (?:snapshots|frames|steps|time ?steps|states)",
        re.I,
    ),
    re.compile(
        r"\b(\w+)\s+(?:input\s+)?(?:timestamps|time ?steps|snapshots|frames|"
        r"states)\s+as\s+inputs?",
        re.I,
    ),
    re.compile(
        r"(?:data|states?) from (\w+) (?:timestamps|time ?steps|times)\b",
        re.I,
    ),
)


def _unit(token: str) -> str:
    return _UNIT_NAMES.get(token.lower(), token)


def find_time_step(sentence: str) -> tuple[str, str] | None:
    """Return (value, matched text) for a model time step."""
    dt = DT.search(sentence)
    if dt:
        return dt.group(1).strip(), dt.group(0)
    for pattern in _STEP_PATTERNS:
        match = pattern.search(sentence)
        if match:
            value = match.group(1)
            if not value[0].isdigit():
                value = str(word_number(value))
            return f"{value} {_unit(match.group(2))}", match.group(0)
    return None


def find_horizon(
    sentence: str, exclude: str | None = None
) -> tuple[str, str] | None:
    """Return (value, matched text) for a forecast or rollout horizon.

    ``exclude`` skips a candidate equal to it, such as the time step in
    "trained to produce 5-day forecasts and supports rollouts up to a year".
    """
    for pattern in _HORIZON_PATTERNS:
        for match in pattern.finditer(sentence):
            value = f"{match.group(1).strip()} {_unit(match.group(2))}"
            if value != exclude:
                return value, match.group(0)
    return None


def find_context_steps(sentence: str) -> tuple[int, str] | None:
    for pattern in _CONTEXT_PATTERNS:
        match = pattern.search(sentence)
        if match:
            value = word_number(match.group(1))
            if value and 1 <= value <= 1000:
                return value, match.group(0)
    return None


SPDX_NAMES = {
    "apache-2.0": ("apache-2.0", "apache 2.0", "apache v2.0", "apache2"),
    "mit": ("mit", "mit license"),
    "bsd-3-clause": ("bsd-3-clause", "bsd 3-clause", "bsd-3", "bsd"),
    "bsd-2-clause": ("bsd-2-clause", "bsd 2-clause"),
    "cc-by-4.0": (
        "cc-by-4.0",
        "cc by 4.0",
        "creative commons attribution 4.0",
    ),
    "cc-by-sa-4.0": ("cc-by-sa-4.0",),
    "cc-by-nc-4.0": ("cc-by-nc-4.0", "cc by-nc 4.0"),
    "cc-by-nc-sa-4.0": ("cc-by-nc-sa-4.0",),
    "cc-by-nc-nd-4.0": ("cc-by-nc-nd-4.0",),
    "cc0-1.0": ("cc0-1.0", "cc0"),
    "cdla-permissive-2.0": ("cdla-permissive-2.0",),
    "gpl-3.0": ("gpl-3.0", "gplv3", "gpl"),
    "lgpl-3.0": ("lgpl-3.0",),
    "agpl-3.0": ("agpl-3.0",),
    "mpl-2.0": ("mpl-2.0",),
    "openrail": ("openrail",),
    "other": ("other",),
}
COMMERCIAL_OK = frozenset({
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "cdla-permissive-2.0",
    "gpl-3.0",
    "lgpl-3.0",
    "agpl-3.0",
    "mpl-2.0",
    "openrail",
})


def normalize_license(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    key = value.strip().lower()
    for spdx, names in SPDX_NAMES.items():
        if key == spdx or key in names:
            return spdx
    for spdx, names in SPDX_NAMES.items():
        if spdx != "other" and any(
            re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", key)
            for n in names
            if len(n) > 3
        ):
            return spdx
    return key if re.fullmatch(r"[a-z0-9.+-]{2,40}", key) else None


def commercial_use_for(spdx: str | None) -> bool | None:
    if spdx in COMMERCIAL_OK:
        return True
    if spdx and "-nc" in spdx:
        return False
    return None


def split_camel(text: str) -> str:
    """'SwinForPDE' -> 'Swin For PDE'; 'pre_earthquake' -> 'pre earthquake'."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return text.replace("_", " ")
