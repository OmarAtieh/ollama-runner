"""GGUF metadata reader and model directory scanner."""

from __future__ import annotations

import logging
import re
import struct
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

GGUF_MAGIC = 0x46554747  # "GGUF" as little-endian uint32

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# Map file_type int to quantization name
FILE_TYPE_MAP = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "IQ3_XXS",
    22: "IQ1_S",
    23: "IQ4_NL",
    24: "IQ3_S",
    25: "IQ2_S",
    26: "IQ4_XS",
    27: "IQ1_M",
    28: "BF16",
    29: "Q4_0_4_4",
    30: "Q4_0_4_8",
    31: "Q4_0_8_8",
    32: "Q1_0",
}

# Known quant patterns for filename fallback (order matters: longest first)
_QUANT_PATTERNS = [
    "Q1_0_g128", "Q1_0",
    "IQ2_XXS", "IQ3_XXS", "IQ1_S", "IQ1_M", "IQ2_XS", "IQ2_S",
    "IQ3_S", "IQ4_NL", "IQ4_XS",
    "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_K_S", "Q4_K_M",
    "Q5_K_S", "Q5_K_M",
    "Q2_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1",
    "Q6_K", "Q8_0", "Q8_1",
    "F16", "F32", "BF16",
]


# Capability detection patterns: (capability_name, patterns_to_match)
_CAPABILITY_PATTERNS = [
    ("thinking", ["think", "reasoning", "cot", "-r1-", "-r1.", "r1-"]),
    ("vision", ["vision", "-vl-", "-vl.", "4.6v", "multimodal", "llava",
                "minicpm-v", "cogvlm", "internvl", "pixtral"]),
    ("code", ["code", "coder", "codestral", "starcoder", "codellama",
              "deepseek-v2.5", "qwen2.5-coder", "qwen3.5-coder"]),
    ("math", ["math", "mathstral", "deepseek-math"]),
    ("chat", ["chat", "instruct", "-it-", "-it."]),
    ("tools", ["tool-use", "function", "hermes", "functionary"]),
    ("creative", ["creative", "writing", "story"]),
    ("multilingual", ["multilingual", "aya", "qwen"]),
    ("embedding", ["embed", "e5-", "bge-"]),
    ("roleplay", ["roleplay", "rp-", "mythomax", "lumimaid"]),
    ("moe", ["-moe-", "-a2b", "-a3b", "-a4b", "mixture"]),
    ("1bit", ["q1_0", "1-bit", "1bit", "bonsai"]),
]


def infer_capabilities(name: str, metadata_tags: list[str] | None = None) -> list[str]:
    """Infer model capabilities from name, path, and metadata tags."""
    lower = name.lower()
    caps = []
    for cap, patterns in _CAPABILITY_PATTERNS:
        for pat in patterns:
            if pat in lower:
                caps.append(cap)
                break
    # Also check metadata tags if available (e.g. from general.tags)
    if metadata_tags:
        tag_str = " ".join(t.lower() for t in metadata_tags)
        for cap, patterns in _CAPABILITY_PATTERNS:
            if cap not in caps:
                for pat in patterns:
                    if pat in tag_str:
                        caps.append(cap)
                        break
    return caps


@dataclass
class GGUFMetadata:
    file_path: str
    file_size_mb: float
    model_name: str | None
    display_name: str  # parent_folder/filename for UI clarity
    architecture: str | None
    parameter_count: int | None
    quantization: str
    context_length: int | None
    embedding_length: int | None
    num_layers: int | None
    vocab_size: int | None
    capabilities: list[str] | None = None


def infer_quant_from_filename(filename: str) -> str:
    """Infer quantization type from filename."""
    upper = filename.upper()
    for pat in _QUANT_PATTERNS:
        if pat.upper() in upper:
            return pat
    return "unknown"


_ONEBIT_QUANTS = {"Q1_0", "Q1_0_G128"}


def is_onebit_quant(quant: str) -> bool:
    """Return True if the quantization type requires the 1-bit fork binary."""
    return quant.upper() in _ONEBIT_QUANTS


def _read_string(f) -> str:
    """Read a GGUF string: uint64 length, then bytes."""
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f, vtype: int):
    """Read a single GGUF metadata value of the given type."""
    if vtype == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif vtype == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif vtype == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif vtype == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif vtype == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == GGUF_TYPE_BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    elif vtype == GGUF_TYPE_STRING:
        return _read_string(f)
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    elif vtype == GGUF_TYPE_ARRAY:
        (arr_type,) = struct.unpack("<I", f.read(4))
        (arr_len,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, arr_type) for _ in range(arr_len)]
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def read_gguf_metadata(file_path: Path) -> GGUFMetadata | None:
    """Read GGUF file header and extract metadata. Returns None on error."""
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        with open(file_path, "rb") as f:
            # Read magic number
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                log.warning("Not a GGUF file: %s (magic=%08x)", file_path, magic)
                return None

            # Read header
            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

            # Read key-value metadata
            metadata: dict = {}
            for _ in range(metadata_kv_count):
                key = _read_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                value = _read_value(f, vtype)
                metadata[key] = value

        # Extract fields
        model_name = metadata.get("general.name")
        architecture = metadata.get("general.architecture")
        file_type = metadata.get("general.file_type")

        # Quantization from file_type or filename
        quantization = "unknown"
        if file_type is not None and isinstance(file_type, int):
            quantization = FILE_TYPE_MAP.get(file_type, "unknown")
        if quantization == "unknown":
            quantization = infer_quant_from_filename(file_path.name)

        # Architecture-specific keys
        arch = architecture or ""
        context_length = (
            metadata.get(f"{arch}.context_length")
            or metadata.get("llama.context_length")
        )
        embedding_length = (
            metadata.get(f"{arch}.embedding_length")
            or metadata.get("llama.embedding_length")
        )
        block_count = (
            metadata.get(f"{arch}.block_count")
            or metadata.get("llama.block_count")
        )
        vocab_size = (
            metadata.get(f"{arch}.vocab_size")
            or metadata.get("llama.vocab_size")
        )

        # Parameter count estimate: if not in metadata, estimate from layers and embedding
        parameter_count = metadata.get("general.parameter_count")
        if parameter_count is None and embedding_length and block_count:
            # Very rough estimate: 12 * hidden^2 * layers (transformer formula)
            parameter_count = int(12 * embedding_length * embedding_length * block_count)

        # Build display name: parent_folder/filename
        display_name = f"{file_path.parent.name}/{file_path.name}"

        # Infer capabilities from name, path, and metadata tags
        name_for_caps = f"{model_name or ''} {file_path.name} {file_path.parent.name}"
        metadata_tags = metadata.get("general.tags")
        if metadata_tags and not isinstance(metadata_tags, list):
            metadata_tags = None
        capabilities = infer_capabilities(name_for_caps, metadata_tags)

        return GGUFMetadata(
            file_path=str(file_path),
            file_size_mb=round(file_size_mb, 1),
            model_name=model_name,
            display_name=display_name,
            architecture=architecture,
            parameter_count=parameter_count,
            quantization=quantization,
            context_length=int(context_length) if context_length else None,
            embedding_length=int(embedding_length) if embedding_length else None,
            num_layers=int(block_count) if block_count else None,
            vocab_size=int(vocab_size) if vocab_size else None,
            capabilities=capabilities or None,
        )

    except Exception as exc:
        log.error("Failed to read GGUF metadata from %s: %s", file_path, exc)
        return None


def scan_models_directory(models_dir: Path) -> list[GGUFMetadata]:
    """Recursively scan directory for .gguf files, skip mmproj files."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    results: list[GGUFMetadata] = []
    for gguf_path in models_dir.rglob("*.gguf"):
        # Skip vision projector files
        if "mmproj" in gguf_path.name.lower():
            continue
        meta = read_gguf_metadata(gguf_path)
        if meta is not None:
            results.append(meta)

    # Sort by file size descending
    results.sort(key=lambda m: m.file_size_mb, reverse=True)
    return results
