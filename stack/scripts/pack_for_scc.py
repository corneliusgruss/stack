"""
Pack sessions into compact format for SCC upload.

Resizes images from 480x360 to 224x224 and strips video/imu,
reducing disk usage by ~70%. Output can be uploaded directly to SCC.

Usage:
    python -m stack.scripts.pack_for_scc --data-dir data/raw --output-dir data/scc_packed
    scp data/scc_packed/*.tar cgruss@scc1.bu.edu:~/stack/data/raw/
"""

import argparse
import tarfile
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def pack_session(session_dir: Path, output_dir: Path, image_size: int = 224):
    """Pack a single session into a compact tar file.

    - Resizes RGB images to image_size x image_size
    - Includes: metadata.json, poses.json, encoders.json, calib.txt
    - Excludes: video.mov, imu.json
    """
    name = session_dir.name
    tar_path = output_dir / f"{name}.tar"

    rgb_dir = session_dir / "rgb"
    if not rgb_dir.exists():
        print(f"  {name}: no rgb/ directory, skipping")
        return None

    with tarfile.open(tar_path, "w") as tar:
        # Add metadata files
        for fname in ["metadata.json", "poses.json", "encoders.json", "calib.txt"]:
            fpath = session_dir / fname
            if fpath.exists():
                tar.add(str(fpath), arcname=f"{name}/{fname}")

        # Add resized images
        frame_paths = sorted(rgb_dir.glob("*.jpg"))
        for fpath in frame_paths:
            img = Image.open(fpath)
            img = img.resize((image_size, image_size), Image.BILINEAR)

            # Save to buffer
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)

            # Add to tar
            info = tarfile.TarInfo(name=f"{name}/rgb/{fpath.name}")
            info.size = len(buf.getvalue())
            tar.addfile(info, buf)

    size_mb = tar_path.stat().st_size / 1e6
    return size_mb


def main():
    parser = argparse.ArgumentParser(description="Pack sessions for SCC upload")
    parser.add_argument("--data-dir", required=True, help="Raw session directory")
    parser.add_argument("--output-dir", default="data/scc_packed", help="Output directory")
    parser.add_argument("--image-size", type=int, default=224, help="Resize images to this size")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_") and (d / "rgb").exists()
    )

    print(f"Packing {len(session_dirs)} sessions to {output_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")

    total_size = 0
    for sd in tqdm(session_dirs, desc="Packing"):
        size = pack_session(sd, output_dir, args.image_size)
        if size is not None:
            total_size += size

    # Also compute original size for comparison
    orig_size = sum(
        f.stat().st_size for sd in session_dirs
        for f in sd.rglob("*") if f.is_file() and f.name != "video.mov" and f.name != "imu.json"
    ) / 1e6

    print(f"\nDone!")
    print(f"  Original (no video/imu): {orig_size:.0f} MB")
    print(f"  Packed: {total_size:.0f} MB")
    print(f"  Savings: {orig_size - total_size:.0f} MB ({(1 - total_size/orig_size)*100:.0f}%)")
    print(f"\nUpload to SCC:")
    print(f"  scp {output_dir}/*.tar cgruss@scc1.bu.edu:~/stack/data/packed/")
    print(f"\nOn SCC, unpack:")
    print(f"  cd ~/stack/data/packed && for t in *.tar; do tar xf $t; done")


if __name__ == "__main__":
    main()
