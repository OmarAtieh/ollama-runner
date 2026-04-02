# Optimized Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maximize inference performance on RTX 2070 Max-Q by adding 1-bit model support, TurboQuant KV cache options, speculative decoding config, optimized launch parameters, and process-level tuning — all driven by a single custom-built llama-server binary.

**Architecture:** BinaryManager gains a `BinaryVariant` enum managing two binary slots: `primary` (upstream auto-download) and `custom` (user-provided optimized fork). ModelConfig grows fields for KV cache type and speculative decoding. ModelManager auto-detects 1-bit models, selects the right binary, tunes thread count based on offload ratio, sets CUDA env vars, and elevates process priority. Model scanner gains Q1_0 quant awareness and 1-bit capability detection.

**Tech Stack:** Python 3.11+, FastAPI, aiohttp, psutil, pytest + pytest-asyncio

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `app/services/model_scanner.py` | Modify | Add Q1_0 quant types, `is_onebit_quant()`, 1-bit capability pattern |
| `app/services/binary_manager.py` | Modify | Add `BinaryVariant`, multi-binary paths, `register_binary()`, `binary_path_for()` |
| `app/config.py` | Modify | Add `CUSTOM_BIN_DIR` constant |
| `app/services/model_manager.py` | Modify | Add KV cache/spec fields to `ModelConfig`, variant selection, optimized launch cmd, env vars, process priority |
| `app/routers/system.py` | Modify | Add `/binary/register` endpoint |
| `tests/test_model_scanner_onebit.py` | Create | 1-bit quant detection + capability tests |
| `tests/test_binary_variants.py` | Create | Multi-variant binary management tests |
| `tests/test_model_manager_launch.py` | Create | Launch command construction, variant selection, env var tests |
| `tests/test_system_register.py` | Create | Binary registration endpoint tests |

---

### Task 1: Add Q1_0 Quant Detection and 1-bit Capability to Model Scanner

**Files:**
- Modify: `app/services/model_scanner.py:31-92`
- Create: `tests/test_model_scanner_onebit.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model_scanner_onebit.py
"""Tests for 1-bit quantization detection in model scanner."""

from app.services.model_scanner import (
    infer_quant_from_filename,
    is_onebit_quant,
    infer_capabilities,
)


class TestOnebitQuantDetection:
    def test_infer_q1_0_g128_from_filename(self):
        assert infer_quant_from_filename("Bonsai-8B-Q1_0_g128.gguf") == "Q1_0_g128"

    def test_infer_q1_0_from_filename(self):
        assert infer_quant_from_filename("model-Q1_0.gguf") == "Q1_0"

    def test_is_onebit_q1_0_g128(self):
        assert is_onebit_quant("Q1_0_g128") is True

    def test_is_onebit_q1_0(self):
        assert is_onebit_quant("Q1_0") is True

    def test_is_not_onebit_q4_k_m(self):
        assert is_onebit_quant("Q4_K_M") is False

    def test_is_not_onebit_unknown(self):
        assert is_onebit_quant("unknown") is False

    def test_is_onebit_case_insensitive(self):
        assert is_onebit_quant("q1_0_g128") is True


class TestOnebitCapabilityInference:
    def test_bonsai_detected_as_onebit(self):
        caps = infer_capabilities("Bonsai-8B-Q1_0_g128.gguf")
        assert "1bit" in caps

    def test_q1_0_in_name_detected(self):
        caps = infer_capabilities("model-Q1_0.gguf")
        assert "1bit" in caps

    def test_regular_model_not_onebit(self):
        caps = infer_capabilities("Qwen3.5-4B-Q4_K_M.gguf")
        assert "1bit" not in caps
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_scanner_onebit.py -v`
Expected: FAIL — `is_onebit_quant` not found, `Q1_0_g128` not in patterns

- [ ] **Step 3: Implement in model_scanner.py**

Add `32: "Q1_0"` to `FILE_TYPE_MAP` dict (after entry 31).

Add `"Q1_0_g128"` and `"Q1_0"` to the front of `_QUANT_PATTERNS` list:
```python
_QUANT_PATTERNS = [
    "Q1_0_g128", "Q1_0",
    "IQ2_XXS", "IQ3_XXS", "IQ1_S", "IQ1_M", "IQ2_XS", "IQ2_S",
    # ... rest unchanged
]
```

Add `("1bit", ["q1_0", "1-bit", "1bit", "bonsai"])` to `_CAPABILITY_PATTERNS` list.

Add after `infer_quant_from_filename` function:
```python
_ONEBIT_QUANTS = {"Q1_0", "Q1_0_G128"}


def is_onebit_quant(quant: str) -> bool:
    """Return True if the quantization type requires the 1-bit fork binary."""
    return quant.upper() in _ONEBIT_QUANTS
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_scanner_onebit.py -v`
Expected: all 10 PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/model_scanner.py tests/test_model_scanner_onebit.py
git commit -m "feat: add Q1_0_g128 quantization detection and 1-bit capability inference"
```

---

### Task 2: Add BinaryVariant and Multi-Binary Support to BinaryManager

**Files:**
- Modify: `app/services/binary_manager.py:1-65`
- Modify: `app/config.py:7`
- Create: `tests/test_binary_variants.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_binary_variants.py
"""Tests for multi-variant binary management."""

from pathlib import Path

from app.services.binary_manager import BinaryManager, BinaryVariant


class TestBinaryVariantPaths:
    def test_primary_variant_path(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        path = bm.binary_path_for(BinaryVariant.PRIMARY)
        assert path == tmp_path / "llama-server.exe"

    def test_custom_variant_path(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        path = bm.binary_path_for(BinaryVariant.CUSTOM)
        assert path == tmp_path / "custom" / "llama-server.exe"


class TestBinaryVariantAvailability:
    def test_primary_available_when_exists(self, tmp_path):
        (tmp_path / "llama-server.exe").write_bytes(b"fake")
        bm = BinaryManager(bin_dir=tmp_path)
        assert bm.is_variant_available(BinaryVariant.PRIMARY) is True

    def test_custom_not_available_when_missing(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        assert bm.is_variant_available(BinaryVariant.CUSTOM) is False

    def test_custom_available_when_exists(self, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "llama-server.exe").write_bytes(b"fake")
        bm = BinaryManager(bin_dir=tmp_path)
        assert bm.is_variant_available(BinaryVariant.CUSTOM) is True


class TestRegisterBinary:
    def test_register_copies_file(self, tmp_path):
        src = tmp_path / "source" / "llama-server.exe"
        src.parent.mkdir()
        src.write_bytes(b"CUSTOM_EXE")

        bm = BinaryManager(bin_dir=tmp_path)
        result = bm.register_binary(BinaryVariant.CUSTOM, src)
        assert result is True
        assert bm.is_variant_available(BinaryVariant.CUSTOM) is True
        assert src.exists()  # original not moved

    def test_register_copies_dlls_too(self, tmp_path):
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "llama-server.exe").write_bytes(b"EXE")
        (src_dir / "ggml-cuda.dll").write_bytes(b"DLL1")
        (src_dir / "llama.dll").write_bytes(b"DLL2")

        bm = BinaryManager(bin_dir=tmp_path)
        result = bm.register_binary(BinaryVariant.CUSTOM, src_dir / "llama-server.exe")
        assert result is True
        custom_dir = tmp_path / "custom"
        assert (custom_dir / "ggml-cuda.dll").exists()
        assert (custom_dir / "llama.dll").exists()

    def test_register_rejects_nonexistent(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        result = bm.register_binary(BinaryVariant.CUSTOM, tmp_path / "nope.exe")
        assert result is False


class TestGetStatusIncludesVariants:
    def test_status_has_variants_dict(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        status = bm.get_status()
        assert "variants" in status
        assert "primary" in status["variants"]
        assert "custom" in status["variants"]
        assert status["variants"]["primary"]["available"] is False
        assert status["variants"]["custom"]["available"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_binary_variants.py -v`
Expected: FAIL — `BinaryVariant` does not exist

- [ ] **Step 3: Add CUSTOM_BIN_DIR to config.py**

In `app/config.py`, add after `BIN_DIR = APP_DIR / "bin"` (line 7):
```python
CUSTOM_BIN_DIR = BIN_DIR / "custom"
```

- [ ] **Step 4: Implement BinaryVariant and multi-binary methods in binary_manager.py**

Add imports at top of `app/services/binary_manager.py`:
```python
import enum
import shutil
```

Add after `log = logging.getLogger(__name__)` (line 13), before `GITHUB_LATEST_RELEASE_URL`:
```python
class BinaryVariant(enum.Enum):
    """Supported llama-server binary variants."""
    PRIMARY = "primary"   # upstream llama.cpp (auto-downloaded)
    CUSTOM = "custom"     # user-provided optimized fork


_VARIANT_SUBDIRS: dict[BinaryVariant, str | None] = {
    BinaryVariant.PRIMARY: None,     # lives at bin root
    BinaryVariant.CUSTOM: "custom",  # lives at bin/custom/
}
```

Add methods to `BinaryManager` class, after `instance()` classmethod and before `binary_path` property:
```python
    # -- Variant helpers -----------------------------------------------------

    def _variant_dir(self, variant: BinaryVariant) -> Path:
        """Return the directory for a given variant."""
        subdir = _VARIANT_SUBDIRS[variant]
        if subdir is None:
            return self._bin_dir
        return self._bin_dir / subdir

    def binary_path_for(self, variant: BinaryVariant) -> Path:
        """Return the expected binary path for a given variant."""
        return self._variant_dir(variant) / "llama-server.exe"

    def is_variant_available(self, variant: BinaryVariant) -> bool:
        """Check if a variant's binary exists."""
        return self.binary_path_for(variant).exists()

    def register_binary(self, variant: BinaryVariant, source_path: Path) -> bool:
        """Copy a user-provided binary (and sibling DLLs) into the variant's directory.
        Returns True on success."""
        if not source_path.exists():
            log.error("Source binary does not exist: %s", source_path)
            return False

        dest_dir = self._variant_dir(variant)
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy the exe
            shutil.copy2(source_path, dest_dir / source_path.name)
            # Copy sibling DLLs
            for dll in source_path.parent.glob("*.dll"):
                shutil.copy2(dll, dest_dir / dll.name)
            log.info("Registered %s binary from %s", variant.value, source_path)
            return True
        except Exception as exc:
            log.error("Failed to register %s binary: %s", variant.value, exc)
            return False
```

Update `get_status` method to include variants:
```python
    def get_status(self) -> dict:
        variants = {}
        for v in BinaryVariant:
            variants[v.value] = {
                "available": self.is_variant_available(v),
                "path": str(self.binary_path_for(v)),
            }
        return {
            "available": self.is_available,
            "path": str(self._binary_path),
            "status": self._download_status,
            "progress": self._download_progress,
            "variants": variants,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_binary_variants.py tests/test_binary_manager.py -v`
Expected: all PASS (new + existing)

- [ ] **Step 6: Commit**

```bash
git add app/config.py app/services/binary_manager.py tests/test_binary_variants.py
git commit -m "feat: add BinaryVariant enum with custom fork binary support"
```

---

### Task 3: Add Binary Registration Endpoint

**Files:**
- Modify: `app/routers/system.py`
- Create: `tests/test_system_register.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_system_register.py
"""Tests for binary registration endpoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.binary_manager import BinaryManager, BinaryVariant

client = TestClient(app)


class TestBinaryRegisterEndpoint:
    def test_register_returns_200_on_success(self, tmp_path):
        src = tmp_path / "llama-server.exe"
        src.write_bytes(b"FAKE")

        with patch.object(BinaryManager, "instance") as mock_inst:
            mock_bm = MagicMock()
            mock_bm.register_binary.return_value = True
            mock_bm.get_status.return_value = {
                "available": True, "path": "x", "status": "idle",
                "progress": 0.0, "variants": {},
            }
            mock_inst.return_value = mock_bm

            response = client.post(
                "/api/system/binary/register",
                json={"variant": "custom", "source_path": str(src)},
            )
            assert response.status_code == 200
            mock_bm.register_binary.assert_called_once()

    def test_register_returns_400_for_missing_file(self):
        response = client.post(
            "/api/system/binary/register",
            json={"variant": "custom", "source_path": "C:/nonexistent/llama-server.exe"},
        )
        assert response.status_code == 400

    def test_register_returns_400_for_invalid_variant(self):
        response = client.post(
            "/api/system/binary/register",
            json={"variant": "nope", "source_path": "C:/fake.exe"},
        )
        assert response.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_system_register.py -v`
Expected: FAIL — 404 (endpoint does not exist)

- [ ] **Step 3: Implement endpoint in system.py**

Add imports at top of `app/routers/system.py`:
```python
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.services.binary_manager import BinaryManager, BinaryVariant
```

(Replace existing `from fastapi import APIRouter, BackgroundTasks` and `from app.services.binary_manager import BinaryManager`.)

Add request model and endpoint after the `binary_download` function:
```python
class RegisterBinaryRequest(BaseModel):
    variant: str
    source_path: str


@router.post("/binary/register")
async def register_binary(req: RegisterBinaryRequest):
    """Register a user-provided llama-server binary for a variant."""
    try:
        variant = BinaryVariant(req.variant)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown variant: {req.variant}. Valid: {[v.value for v in BinaryVariant]}")

    source = Path(req.source_path)
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"Source file not found: {req.source_path}")

    bm = BinaryManager.instance()
    success = bm.register_binary(variant, source)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to register binary")

    return bm.get_status()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_system_register.py -v`
Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add app/routers/system.py tests/test_system_register.py
git commit -m "feat: add POST /api/system/binary/register endpoint"
```

---

### Task 4: Add KV Cache and Speculative Decoding Fields to ModelConfig

**Files:**
- Modify: `app/services/model_manager.py:26-39`
- Modify: `app/routers/models.py:19-48`

- [ ] **Step 1: Write failing test**

```python
# tests/test_model_config_fields.py
"""Tests for new ModelConfig fields."""

from app.services.model_manager import ModelConfig


class TestModelConfigNewFields:
    def test_defaults(self):
        m = ModelConfig(id="x", name="test", path="/m.gguf")
        assert m.cache_type_k == "f16"
        assert m.cache_type_v == "f16"
        assert m.speculative == "none"

    def test_custom_values(self):
        m = ModelConfig(
            id="x", name="test", path="/m.gguf",
            cache_type_k="q8_0", cache_type_v="q8_0",
            speculative="ngram",
        )
        assert m.cache_type_k == "q8_0"
        assert m.speculative == "ngram"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_config_fields.py -v`
Expected: FAIL — `ModelConfig` does not accept `cache_type_k`

- [ ] **Step 3: Add fields to ModelConfig**

In `app/services/model_manager.py`, add to the `ModelConfig` dataclass after `notes`:
```python
    cache_type_k: str = "f16"     # KV cache key type: f16, q8_0, q4_0, tbq3_0, tbq4_0
    cache_type_v: str = "f16"     # KV cache value type: f16, q8_0, q4_0, tbq3_0, tbq4_0
    speculative: str = "none"     # none, ngram
```

- [ ] **Step 4: Add fields to Pydantic request models in models.py**

In `app/routers/models.py`, add to `AddModelRequest`:
```python
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    speculative: str = "none"
```

Add to `UpdateModelRequest`:
```python
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    speculative: Optional[str] = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_model_config_fields.py -v`
Expected: all 2 PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/model_manager.py app/routers/models.py tests/test_model_config_fields.py
git commit -m "feat: add KV cache type and speculative decoding fields to ModelConfig"
```

---

### Task 5: Optimized Launch Command Construction

This is the core task — rewires `ModelManager.load_model` to select binary variant, build an optimized command with KV cache flags, speculative decoding, smart thread count, CUDA env vars, and process priority.

**Files:**
- Modify: `app/services/model_manager.py:240-313`
- Create: `tests/test_model_manager_launch.py`

- [ ] **Step 1: Write failing tests for binary variant selection**

```python
# tests/test_model_manager_launch.py
"""Tests for optimized model launch: variant selection, command building, env vars."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.binary_manager import BinaryVariant
from app.services.model_manager import ModelManager, ModelConfig


class TestBinaryVariantSelection:
    def test_selects_custom_for_q1_0_g128(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        variant = mm._select_binary_variant(str(tmp_path / "Bonsai-8B-Q1_0_g128.gguf"))
        assert variant == BinaryVariant.CUSTOM

    def test_selects_primary_for_q4_k_m(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        variant = mm._select_binary_variant(str(tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"))
        assert variant == BinaryVariant.PRIMARY

    def test_selects_custom_when_available_and_not_onebit(self, tmp_path):
        """If a custom binary is registered, prefer it for all models."""
        mm = ModelManager(app_dir=tmp_path)
        # Create custom binary
        custom_dir = tmp_path / "bin" / "custom"
        custom_dir.mkdir(parents=True)
        (custom_dir / "llama-server.exe").write_bytes(b"fake")

        with patch("app.services.binary_manager.BIN_DIR", tmp_path / "bin"):
            from app.services.binary_manager import BinaryManager
            bm = BinaryManager(bin_dir=tmp_path / "bin")
            with patch.object(BinaryManager, "instance", return_value=bm):
                variant = mm._select_binary_variant(str(tmp_path / "regular-Q6_K.gguf"))
                # Custom binary is preferred for ALL models when available
                assert variant == BinaryVariant.CUSTOM
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_manager_launch.py::TestBinaryVariantSelection -v`
Expected: FAIL — `_select_binary_variant` does not exist

- [ ] **Step 3: Write failing tests for command construction**

Add to `tests/test_model_manager_launch.py`:

```python
class TestBuildLaunchCommand:
    def test_includes_kv_cache_flags(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(
            id="x", name="test", path="/m.gguf",
            cache_type_k="q8_0", cache_type_v="q8_0",
        )
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=99,
            context_length=4096,
        )
        assert "--cache-type-k" in cmd
        idx = cmd.index("--cache-type-k")
        assert cmd[idx + 1] == "q8_0"
        assert "--cache-type-v" in cmd
        idx = cmd.index("--cache-type-v")
        assert cmd[idx + 1] == "q8_0"

    def test_skips_kv_cache_flags_when_f16(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(id="x", name="test", path="/m.gguf")
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=99,
            context_length=4096,
        )
        assert "--cache-type-k" not in cmd

    def test_includes_ngram_spec_flags(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(
            id="x", name="test", path="/m.gguf",
            speculative="ngram",
        )
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=99,
            context_length=4096,
        )
        assert "--spec-type" in cmd
        idx = cmd.index("--spec-type")
        assert cmd[idx + 1] == "ngram-mod"
        assert "--draft-max" in cmd

    def test_skips_spec_flags_when_none(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(id="x", name="test", path="/m.gguf")
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=99,
            context_length=4096,
        )
        assert "--spec-type" not in cmd

    def test_threads_1_when_fully_offloaded(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(id="x", name="test", path="/m.gguf")
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=99,
            context_length=4096,
        )
        idx = cmd.index("-t")
        assert cmd[idx + 1] == "1"

    def test_threads_multi_when_partial_offload(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(id="x", name="test", path="/m.gguf")
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=10,
            context_length=4096,
        )
        idx = cmd.index("-t")
        assert int(cmd[idx + 1]) > 1


class TestLaunchEnvironment:
    def test_cuda_env_vars_set(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        env = mm._build_launch_env()
        assert env.get("GGML_CUDA_GRAPH_OPT") == "1"
        assert env.get("CUDA_SCALE_LAUNCH_QUEUES") == "4x"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_model_manager_launch.py -v`
Expected: FAIL — `_build_launch_cmd`, `_build_launch_env` do not exist

- [ ] **Step 5: Implement variant selection, command building, and env construction**

Add imports at top of `app/services/model_manager.py`:
```python
import os

from app.services.binary_manager import BinaryManager, BinaryVariant
from app.services.model_scanner import (
    infer_capabilities,
    infer_quant_from_filename,
    is_onebit_quant,
    read_gguf_metadata,
)
```

(Replace existing `from app.services.binary_manager import BinaryManager` and `from app.services.model_scanner import infer_capabilities, read_gguf_metadata`.)

Add these methods to `ModelManager` class, before `load_model`:

```python
    def _select_binary_variant(self, model_path: str) -> BinaryVariant:
        """Determine which binary variant to use.

        Logic:
        - 1-bit models REQUIRE the custom binary
        - If a custom binary is registered, prefer it for all models
          (it includes all upstream features + optimizations)
        - Otherwise fall back to primary
        """
        bm = BinaryManager.instance()

        # 1-bit models must use custom
        meta = read_gguf_metadata(Path(model_path))
        quant = meta.quantization if meta else infer_quant_from_filename(Path(model_path).name)
        if is_onebit_quant(quant):
            return BinaryVariant.CUSTOM

        # Prefer custom for all models when available (it's a superset)
        if bm.is_variant_available(BinaryVariant.CUSTOM):
            return BinaryVariant.CUSTOM

        return BinaryVariant.PRIMARY

    def _build_launch_cmd(
        self,
        model: ModelConfig,
        binary_path: Path,
        gpu_layers: int,
        context_length: int,
    ) -> list[str]:
        """Build the llama-server command with all optimized flags."""
        # Smart thread count: 1 when fully offloaded, multi otherwise
        fully_offloaded = gpu_layers >= 99
        threads = 1 if fully_offloaded else max(1, (psutil.cpu_count(logical=False) or 2) - 1)

        cmd = [
            str(binary_path),
            "-m", model.path,
            "-c", str(context_length),
            "-ngl", str(gpu_layers),
            "--host", "127.0.0.1",
            "--port", str(self._server_port),
            "-fa", "on",
            "--mlock",
            "-t", str(threads),
            "--batch-size", "512",
            "--ubatch-size", "512",
        ]

        # KV cache quantization
        if model.cache_type_k != "f16":
            cmd.extend(["--cache-type-k", model.cache_type_k])
        if model.cache_type_v != "f16":
            cmd.extend(["--cache-type-v", model.cache_type_v])

        # Speculative decoding
        if model.speculative == "ngram":
            cmd.extend([
                "--spec-type", "ngram-mod",
                "--spec-ngram-size-n", "12",
                "--spec-ngram-size-m", "48",
                "--draft-max", "16",
            ])

        # Windows-specific
        if platform.system() == "Windows":
            cmd.append("--no-mmap")

        return cmd

    def _build_launch_env(self) -> dict[str, str]:
        """Build environment variables for the llama-server process."""
        env = os.environ.copy()
        env["GGML_CUDA_GRAPH_OPT"] = "1"
        env["CUDA_SCALE_LAUNCH_QUEUES"] = "4x"
        return env
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_model_manager_launch.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add app/services/model_manager.py tests/test_model_manager_launch.py
git commit -m "feat: add optimized launch command builder with variant selection and CUDA env vars"
```

---

### Task 6: Rewire load_model to Use New Helpers

**Files:**
- Modify: `app/services/model_manager.py:240-313`

- [ ] **Step 1: Write failing test for the wired-up load path**

```python
# tests/test_model_manager_load_wired.py
"""Test that load_model uses variant selection and optimized command."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import subprocess

import pytest

from app.services.binary_manager import BinaryManager, BinaryVariant
from app.services.model_manager import ModelManager, ModelConfig


class TestLoadModelVariantWiring:
    @pytest.mark.asyncio
    async def test_load_onebit_fails_without_custom_binary(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)

        # Register a 1-bit model in the registry
        model = mm.add_model("Bonsai-8B", str(tmp_path / "Bonsai-8B-Q1_0_g128.gguf"))

        # Mock BinaryManager with no custom binary
        mock_bm = MagicMock()
        mock_bm.is_available = True
        mock_bm.is_variant_available.return_value = False
        mock_bm.binary_path_for.return_value = tmp_path / "custom" / "llama-server.exe"

        with patch.object(BinaryManager, "instance", return_value=mock_bm):
            result = await mm.load_model(model.id)

        assert result is False
        assert "custom" in mm.get_status()["error"].lower() or "1-bit" in mm.get_status()["error"].lower()

    @pytest.mark.asyncio
    async def test_load_passes_env_vars_to_popen(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = mm.add_model("TestModel", str(tmp_path / "test-Q4_K_M.gguf"))

        # Create fake binary
        (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
        (tmp_path / "bin" / "llama-server.exe").write_bytes(b"fake")

        mock_bm = MagicMock()
        mock_bm.is_available = True
        mock_bm.is_variant_available.return_value = True
        mock_bm.binary_path_for.return_value = tmp_path / "bin" / "llama-server.exe"

        with patch.object(BinaryManager, "instance", return_value=mock_bm), \
             patch.object(mm, "estimate_resources", return_value={"feasible": True, "vram_needed_mb": 100, "vram_available_mb": 8000, "ram_needed_mb": 100, "ram_available_mb": 30000}), \
             patch.object(mm, "recommend_gpu_layers", return_value=99), \
             patch("subprocess.Popen") as mock_popen:

            mock_proc = MagicMock()
            mock_proc.poll.return_value = 1  # process exits immediately
            mock_proc.stderr.read.return_value = b"test"
            mock_popen.return_value = mock_proc

            await mm.load_model(model.id)

            # Verify env was passed
            call_kwargs = mock_popen.call_args
            assert "env" in call_kwargs.kwargs or (len(call_kwargs.args) > 1 if call_kwargs.args else False)
            env = call_kwargs.kwargs.get("env", {})
            assert env.get("GGML_CUDA_GRAPH_OPT") == "1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_manager_load_wired.py -v`
Expected: FAIL — load_model still uses old code path

- [ ] **Step 3: Rewrite the load_model command-building and process-launch section**

In `app/services/model_manager.py`, replace lines 251-313 of `load_model` (from `# Check binary` through the `subprocess.Popen` call) with:

```python
        # Select binary variant based on model quantization
        bm = BinaryManager.instance()
        variant = self._select_binary_variant(model.path)

        if not bm.is_variant_available(variant):
            variant_label = "custom (1-bit fork)" if variant == BinaryVariant.CUSTOM else "primary"
            self._last_error = (
                f"llama-server binary not available for {variant_label} variant. "
                + ("Register the custom fork binary via /api/system/binary/register." if variant == BinaryVariant.CUSTOM else "Download the primary binary first.")
            )
            log.error(self._last_error)
            self._loading = False
            return False

        # Unload current model first
        await self.unload_model()

        self._loading = True
        self._load_progress = 0.0

        try:
            ctx = context_length or model.context_default

            # GPU layers
            gpu_layers = model.gpu_layers
            if gpu_layers == -1:
                gpu_layers = self.recommend_gpu_layers(model.path)

            # Pre-load feasibility check
            estimate = self.estimate_resources(model.path, gpu_layers, ctx)
            if not estimate["feasible"]:
                self._last_error = (
                    f"Model does not fit in available resources. "
                    f"VRAM needed: {estimate['vram_needed_mb']:.0f}MB, "
                    f"available: {estimate['vram_available_mb']}MB. "
                    f"RAM needed: {estimate['ram_needed_mb']:.0f}MB, "
                    f"available: {estimate['ram_available_mb']}MB."
                )
                log.error(self._last_error)
                self._loading = False
                return False

            # Build optimized command and environment
            binary_path = bm.binary_path_for(variant)
            cmd = self._build_launch_cmd(model, binary_path, gpu_layers, ctx)
            env = self._build_launch_env()

            log.info("Starting llama-server (%s): %s", variant.value, " ".join(cmd))

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )

            # Elevate process priority for lower latency
            try:
                import psutil as _psutil
                p = _psutil.Process(self._process.pid)
                if platform.system() == "Windows":
                    p.nice(_psutil.HIGH_PRIORITY_CLASS)
                else:
                    p.nice(-10)
            except Exception:
                pass  # non-critical
```

Note: the rest of `load_model` (the health-poll loop from line ~315 onward) stays unchanged.

Also remove the old `# Unload current model first` and `self._loading = True` lines that were before the old binary check, since we moved them after the variant check.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_manager_load_wired.py tests/test_model_manager_launch.py -v`
Expected: all PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/model_manager.py tests/test_model_manager_load_wired.py
git commit -m "feat: rewire load_model with variant selection, CUDA env vars, and process priority"
```

---

### Task 7: Integration Test — Full Flow

**Files:**
- Create: `tests/test_optimized_inference_integration.py`

- [ ] **Step 1: Write integration tests**

```python
# tests/test_optimized_inference_integration.py
"""Integration: 1-bit detection → variant selection → optimized command."""

from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.binary_manager import BinaryManager, BinaryVariant
from app.services.model_manager import ModelManager, ModelConfig
from app.services.model_scanner import (
    infer_quant_from_filename,
    is_onebit_quant,
    infer_capabilities,
)


class TestOnebitFullFlow:
    def test_bonsai_detection_to_variant(self, tmp_path):
        filename = "Bonsai-8B-Q1_0_g128.gguf"

        # Quant detected
        quant = infer_quant_from_filename(filename)
        assert quant == "Q1_0_g128"
        assert is_onebit_quant(quant) is True

        # Capability includes 1bit
        caps = infer_capabilities(filename)
        assert "1bit" in caps

        # Variant selection
        mm = ModelManager(app_dir=tmp_path)
        variant = mm._select_binary_variant(str(tmp_path / filename))
        assert variant == BinaryVariant.CUSTOM

        # Binary path resolves to custom dir
        bm = BinaryManager(bin_dir=tmp_path)
        path = bm.binary_path_for(variant)
        assert "custom" in str(path)

    def test_regular_model_uses_primary(self, tmp_path):
        filename = "Qwen3.5-9B-Q6_K.gguf"

        quant = infer_quant_from_filename(filename)
        assert quant == "Q6_K"
        assert is_onebit_quant(quant) is False

        mm = ModelManager(app_dir=tmp_path)
        variant = mm._select_binary_variant(str(tmp_path / filename))
        assert variant == BinaryVariant.PRIMARY


class TestOptimizedCommandFlow:
    def test_q8_kv_cache_in_command(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(
            id="x", name="test", path="/m.gguf",
            cache_type_k="q8_0", cache_type_v="q8_0",
            speculative="ngram",
        )
        cmd = mm._build_launch_cmd(model, Path("/bin/llama-server.exe"), 99, 8192)

        # KV cache
        assert "--cache-type-k" in cmd
        assert "q8_0" in cmd

        # Speculative
        assert "--spec-type" in cmd
        assert "ngram-mod" in cmd

        # Threads = 1 (fully offloaded)
        t_idx = cmd.index("-t")
        assert cmd[t_idx + 1] == "1"

    def test_env_vars_present(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        env = mm._build_launch_env()
        assert env["GGML_CUDA_GRAPH_OPT"] == "1"
        assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "4x"
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_optimized_inference_integration.py -v`
Expected: all 4 PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_optimized_inference_integration.py
git commit -m "test: add integration tests for optimized inference engine flow"
```

---

## Summary

After all 7 tasks, OllamaRunner will:

1. **Detect 1-bit models** (Q1_0, Q1_0_g128) from GGUF metadata or filename
2. **Auto-select the custom fork binary** when available (preferred for all models since it's a superset), required for 1-bit
3. **Register custom binaries** via `POST /api/system/binary/register`
4. **Build optimized launch commands** with:
   - KV cache quantization flags (`--cache-type-k q8_0`)
   - Draftless speculative decoding (`--spec-type ngram-mod`)
   - Smart thread count (`-t 1` when fully GPU-offloaded)
   - Flash attention (`-fa on`)
5. **Set CUDA env vars** (`GGML_CUDA_GRAPH_OPT=1`, `CUDA_SCALE_LAUNCH_QUEUES=4x`)
6. **Elevate process priority** to High for lower latency
7. **All existing functionality unchanged** — primary auto-download still works as fallback

The user builds the custom fork once with:
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 12
```
Then registers it via the API, and OllamaRunner handles everything else automatically.
