#!/usr/bin/env python3
"""Register a custom llama-server binary with OllamaRunner.

Usage:
    python scripts/register-custom-binary.py <path-to-llama-server.exe>

This copies the binary and any sibling DLLs into ~/.ollamarunner/bin/custom/
so OllamaRunner will prefer it for all model inference.
"""

import shutil
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/register-custom-binary.py <path-to-llama-server.exe>")
        print("\nExamples:")
        print("  python scripts/register-custom-binary.py ./artifact/llama-server.exe")
        print("  python scripts/register-custom-binary.py C:/Downloads/llama-server.exe")
        sys.exit(1)

    source = Path(sys.argv[1]).resolve()
    if not source.exists():
        print(f"Error: {source} does not exist")
        sys.exit(1)

    if source.name != "llama-server.exe":
        print(f"Warning: expected llama-server.exe, got {source.name}")

    dest_dir = Path.home() / ".ollamarunner" / "bin" / "custom"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy exe
    dest_exe = dest_dir / source.name
    shutil.copy2(source, dest_exe)
    print(f"Copied {source.name} -> {dest_exe}")

    # Copy sibling DLLs
    dll_count = 0
    for dll in source.parent.glob("*.dll"):
        shutil.copy2(dll, dest_dir / dll.name)
        dll_count += 1
        print(f"Copied {dll.name} -> {dest_dir / dll.name}")

    # Copy build manifest if present
    manifest = source.parent / "build-manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, dest_dir / "build-manifest.json")
        print(f"Copied build-manifest.json")

    print(f"\nDone! Registered custom binary at {dest_dir}")
    print(f"  Binary: {dest_exe}")
    print(f"  DLLs: {dll_count}")
    print(f"\nOllamaRunner will now prefer this binary for all models.")


if __name__ == "__main__":
    main()
