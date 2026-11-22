#!/usr/bin/env python3
"""Prepare a small compact-router Strata pack over unchanged GGUF experts.

No expert weights are duplicated or requantized. The patched engine translates
compact IDs to the original file offsets and loads only active expert planes.
An all-experts mask returns the original pack as an exact no-op.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys

PINNED_STRATA = "87dfca0e7c3ee24ad36cc3f69a31b859587245f8"
FILE_NAMES = {0: "dense.bin", 1: "embd.bin", 2: "experts.bin", 3: "extra.bin"}


def load_mask(path: Path) -> dict:
    mask = json.loads(path.read_text())
    for name in ("schema_version", "layers", "num_experts", "top_k"):
        if type(mask.get(name)) is not int:
            raise ValueError(f"{name} must be an integer")
    if mask["schema_version"] != 1:
        raise ValueError("Unsupported mask schema")
    nl, ne, k = mask["layers"], mask["num_experts"], mask["top_k"]
    if not (0 < nl <= 1024 and 0 < k <= ne <= 65535):
        raise ValueError("Invalid geometry")
    allowed = mask.get("allowed")
    if not isinstance(allowed, list) or len(allowed) != nl:
        raise ValueError("allowed must contain exactly one row per layer")
    sizes = set()
    for row in allowed:
        if not isinstance(row, list) or any(type(e) is not int for e in row):
            raise ValueError("Expert IDs must be integers")
        if row != sorted(set(row)) or any(e < 0 or e >= ne for e in row):
            raise ValueError("Expert IDs must be sorted, unique and in range")
        if not k <= len(row) <= ne:
            raise ValueError("Each layer must retain at least top_k experts")
        sizes.add(len(row))
    if len(sizes) != 1:
        raise ValueError("Strata compact geometry requires equal retained counts across layers")
    fraction = len(allowed[0]) / ne
    if "keep_fraction" in mask and abs(float(mask["keep_fraction"]) - fraction) > 1e-12:
        raise ValueError("keep_fraction does not match the actual retained count")
    mask["keep_fraction"] = fraction
    return mask


def engine_map_fingerprint(mask: dict) -> str:
    """Match the engine's noncryptographic actual-map checksum."""
    h = 14695981039346656037
    values = [mask["layers"], mask["num_experts"], len(mask["allowed"][0]), mask["top_k"]]
    values.extend(e for row in mask["allowed"] for e in row)
    for value in values:
        for b in struct.pack("<Q", value):
            h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
    return f"0x{h:016x}"


def read_index(path: Path):
    align, rows = None, []
    for line in path.read_text().splitlines():
        if line.startswith("# align "):
            match = re.fullmatch(r"# align (\d+) pool (\d+) tensors (\d+)", line)
            if match is None:
                raise ValueError("Invalid pack index header")
            align = int(match[1])
        elif line.strip() and not line.startswith("#"):
            fields = line.split()
            if len(fields) != 19:
                raise ValueError("Expected Strata index v3 with 19 fields")
            for value in fields[1:]:
                int(value)
            rows.append(fields)
    if align is None or align <= 0 or not rows:
        raise ValueError("Missing index geometry")
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("Duplicate tensor names")
    return align, rows


def read_layout(path: Path, layers: int, ne: int):
    text = path.read_text()
    match = re.search(r"# strata native experts v(\d+):", text)
    if match is None or int(match[1]) not in (3, 4):
        raise ValueError("This adapter supports plain native_experts.txt v3/v4 only; no Hadamard/NVFP4 variants")
    count = re.search(r"\(n_expert (\d+),", text)
    if count is None or int(count[1]) != ne:
        raise ValueError("Source expert count differs from mask")
    rows = [s.split() for s in text.splitlines() if s.strip() and not s.startswith("#")]
    if len(rows) != layers:
        raise ValueError("Unexpected expert layer count")
    for layer, row in enumerate(rows):
        if len(row) not in (8, 9) or int(row[0]) != layer:
            raise ValueError("Expected one ordered layer with GGUF offsets per row")
        if any(int(value) < 0 for value in row[:8]) or int(row[4]) == 0:
            raise ValueError("Invalid native expert layout")
    return int(match[1]), rows


def link_file(src: Path, dst: Path):
    """Share immutable existing payloads; do not silently copy gigabytes."""
    src = src.resolve(strict=True)
    if src.is_dir():
        dst.symlink_to(src, target_is_directory=True)
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src)


def remap_profile(source: Path | None, out: Path, mask: dict):
    nl, ne, keep = mask["layers"], mask["num_experts"], len(mask["allowed"][0])
    maps = [{original: compact for compact, original in enumerate(row)} for row in mask["allowed"]]
    original_pairs = []
    if source:
        data = source.read_bytes()
        if len(data) < 24 or data[:4] != b"STRP":
            raise ValueError("Not a Strata expert profile")
        version, pl, pe, slots, count = struct.unpack_from("<5I", data, 4)
        if version != 1 or (pl, pe) != (nl, ne) or len(data) != 24 + 4 * count + 4 * nl * ne:
            raise ValueError("Source profile geometry/length mismatch")
        original_pairs = [struct.unpack_from("<HH", data, 24 + 4*i) for i in range(count)]
        if any(l >= nl or e >= ne for l, e in original_pairs):
            raise ValueError("Source profile has an out-of-range expert")
    ranked, seen = [], set()
    for layer, expert in original_pairs:
        if expert in maps[layer]:
            pair = (layer, maps[layer][expert])
            if pair not in seen:
                ranked.append(pair)
                seen.add(pair)
    for compact in range(keep):
        for layer in range(nl):
            pair = (layer, compact)
            if pair not in seen:
                ranked.append(pair)
                seen.add(pair)
    table = [[-1] * keep for _ in range(nl)]
    with out.open("wb") as f:
        f.write(b"STRP" + struct.pack("<5I", 1, nl, keep, len(ranked), len(ranked)))
        for slot, (layer, expert) in enumerate(ranked):
            f.write(struct.pack("<HH", layer, expert))
            table[layer][expert] = slot
        for row in table:
            f.write(struct.pack(f"<{keep}i", *row))


def prepare(base: Path, mask_path: Path, out: Path, profile: Path | None = None,
            strict_geometry: bool = True) -> dict:
    base = base.resolve(strict=True)
    mask = load_mask(mask_path)
    nl, ne, k = mask["layers"], mask["num_experts"], mask["top_k"]
    if strict_geometry and (nl, ne, k) != (48, 512, 10):
        raise ValueError("Adapter is pinned to Qwen3.8-Flash-Next 48x512/top-10")
    keep = len(mask["allowed"][0])
    align, rows = read_index(base / "index.txt")
    version, layout = read_layout(base / "native_experts.txt", nl, ne)
    digest = hashlib.sha256(json.dumps(
        {key: mask[key] for key in ("layers", "num_experts", "top_k", "allowed")},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    manifest = {
        "schema_version": 1, "strata_commit": PINNED_STRATA, "mask_sha256": digest,
        "mask_file": str(mask_path.resolve()), "base_pack": str(base), "source_experts": ne,
        "kept_experts": keep, "top_k": k, "layers": nl,
        "expert_payload_bytes": sum(int(row[4]) * keep for row in layout),
        "source_expert_payload_bytes": sum(int(row[4]) * ne for row in layout),
        "normalization": "softmax -> top-k among retained experts -> renormalize selected weights",
        "shared_expert": "unchanged", "all_experts_noop": keep == ne,
        "engine_map_fnv1a64": engine_map_fingerprint(mask) if keep != ne else None,
    }
    if keep == ne:
        manifest.update(pack=str(base), profile=str(profile.resolve()) if profile else None)
        return manifest
    if (base / "review_subset.txt").exists():
        raise ValueError("Base pack is already a subset; start from the full 512-expert pack")
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}; use a new path or delete that generated pack explicitly")
    if any(r[1] == "3" and int(r[4]) for r in rows):
        raise ValueError("Base uses extra.bin; prepare a standalone base with iq_pack.py first")
    router_layers = {}
    for row in rows:
        match = re.fullmatch(r"blk\.(\d+)\.ffn_gate_inp\.weight", row[0])
        if match:
            layer = int(match[1])
            cols = int(row[7])
            if int(row[2]) != 4 or int(row[8]) != ne or int(row[4]) != cols * ne * 2 or int(row[6]) != cols * ne * 2:
                raise ValueError(f"{row[0]} must be a packed, raw BF16 router of the source expert count")
            if strict_geometry and cols != 2560:
                raise ValueError("Unexpected router hidden width")
            if int(row[1]) not in (0, 1):
                raise ValueError("Unexpected router source file")
            router_layers[layer] = row
    if set(router_layers) != set(range(nl)):
        raise ValueError("Missing or extra routed-expert routers")
    out.mkdir(parents=True)
    try:
        # Native quantized dense tensors continue to come from the original GGUF.
        for file_id in {int(r[1]) for r in rows if int(r[4]) and r not in router_layers.values()}:
            if file_id not in (0, 1):
                raise ValueError("Unexpected source payload; only standalone native packs are supported")
            link_file(base / FILE_NAMES[file_id], out / FILE_NAMES[file_id])
        # In practice dense.bin contains the untouched norms and shared-expert gate.
        if not (out / "dense.bin").exists() and (base / "dense.bin").exists():
            link_file(base / "dense.bin", out / "dense.bin")
        if (base / "tokenizer").exists():
            link_file(base / "tokenizer", out / "tokenizer")
        with (out / "extra.bin").open("wb") as dst:
            for layer in range(nl):
                row = router_layers[layer]
                stride = int(row[7]) * 2
                offset = dst.tell()
                with (base / FILE_NAMES[int(row[1])]).open("rb") as src:
                    for original in mask["allowed"][layer]:
                        src.seek(int(row[3]) + original * stride)
                        raw = src.read(stride)
                        if len(raw) != stride:
                            raise ValueError("Truncated router payload")
                        dst.write(raw)
                nbytes = keep * stride
                dst.write(b"\0" * (-nbytes % align))
                row[1], row[3], row[4], row[6], row[8] = "3", str(offset), str(nbytes), str(nbytes), str(keep)
        pool = 0
        for row in rows:
            row[5] = str(pool)
            pool += (int(row[6]) + align - 1) // align * align
        (out / "index.txt").write_text(
            f"# strata pack index v3 -- fixed expert subset\n# align {align} pool {pool} tensors {len(rows)}\n"
            + "".join(" ".join(row) + "\n" for row in rows))
        offset = 0
        for row in layout:
            row[3] = str(offset)
            offset += keep * int(row[4])
        (out / "native_experts.txt").write_text(
            f"# strata native experts v{version}: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
            f"[shard | gate,up,down] (n_expert {keep}, total {offset}; source offsets unchanged)\n"
            + "".join(" ".join(row) + "\n" for row in layout))
        remap_profile(profile, out / "expert-profile.bin", mask)
        (out / "mask.json").write_text(json.dumps(mask, indent=2) + "\n")
        manifest.update(pack=str(out), profile=str(out / "expert-profile.bin"))
        manifest["extra_router_bytes"] = (out / "extra.bin").stat().st_size
        (out / "subset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Marker is last: before it exists the subset pack must not be launched.
        (out / "review_subset.txt").write_text(
            f"STRATA_REVIEW_SUBSET 1\n{nl} {ne} {keep} {k}\n"
            + "".join(f"{l} " + " ".join(map(str, ids)) + "\n" for l, ids in enumerate(mask["allowed"])))
    except BaseException:
        shutil.rmtree(out)
        raise
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-pack", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    args = p.parse_args()
    try:
        print(json.dumps(prepare(args.base_pack, args.mask, args.out, args.profile), indent=2))
    except (ValueError, OSError) as e:
        p.exit(2, f"prepare_subset: {e}\n")


if __name__ == "__main__":
    main()
