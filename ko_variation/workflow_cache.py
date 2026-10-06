"""Small stage-cache utility for the S. aureus plotting wrapper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import fingerprint_file, _atomic_write_bytes


def file_signature(path):
    source = Path(path).resolve()
    stat = source.stat()
    return dict(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def stage_identity(stage, inputs, settings, directories=()):
    # Avoid rereading huge original pair/genotype files just to reuse a stage.
    # File path, byte size and nanosecond mtime identify unchanged source files.
    inventory = []
    for directory in directories:
        root = Path(directory).resolve()
        for pattern in ("*.gff3", "*.fna"):
            inventory.extend(file_signature(path) for path in sorted(root.rglob(pattern)))
    return dict(stage=stage, cache_schema=1, inputs=[file_signature(path) for path in inputs],
                settings=settings, directory_files=inventory)


def cache_matches(manifest, identity, outputs):
    try:
        saved = json.loads(Path(manifest).read_text(encoding="utf-8"))
        return (saved.get("cache_identity") == identity and
                saved.get("cache_outputs") == [fingerprint_file(path) for path in outputs])
    except (OSError, ValueError):
        return False


def cache_fields(identity, outputs):
    return dict(cache_identity=identity, cache_outputs=[fingerprint_file(path) for path in outputs])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "store"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--directories", nargs="*", default=[])
    parser.add_argument("--settings", nargs="*", default=[])
    args = parser.parse_args(argv)
    identity = stage_identity(args.stage, args.inputs, args.settings, args.directories)
    if args.action == "check":
        matches = cache_matches(args.manifest, identity, args.outputs)
        if matches:
            print(f"[pipeline] reusing completed {args.stage} stage", flush=True)
        return 0 if matches else 1
    _atomic_write_bytes(Path(args.manifest), json.dumps(cache_fields(identity, args.outputs), indent=2).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
