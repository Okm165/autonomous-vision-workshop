#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Vision & 3D Mapping Workshop — Data Downloader ==="
echo ""

mkdir -p sample_pairs calibration kitti tum

# ---------------------------------------------------------------------------
# KITTI Odometry — sequence 00 excerpt (first 200 frames, ~50 MB)
# Used by: NB07 (VO), NB08 (stereo), NB17 (evaluation), NB18 (full pipeline)
# ---------------------------------------------------------------------------
KITTI_DIR="kitti/sequences/00"
if [ ! -d "$KITTI_DIR/image_0" ]; then
    echo "[1/3] Downloading KITTI odometry sequence 00 (200 frames)..."
    echo "      NOTE: KITTI requires registration at https://www.cvlibs.net/datasets/kitti/"
    echo "      Download 'data_odometry_gray.zip' and extract sequence 00 here,"
    echo "      or use the devkit poses from the KITTI website."
    echo ""
    echo "      Expected layout:"
    echo "        $KITTI_DIR/image_0/000000.png ... 000199.png"
    echo "        $KITTI_DIR/image_1/000000.png ... 000199.png"
    echo "        kitti/poses/00.txt"
    echo ""
    echo "      Skipping automatic download (license restrictions)."
    echo ""
else
    echo "[1/3] KITTI sequence 00 already present — skipping."
fi

# ---------------------------------------------------------------------------
# TUM RGB-D — fr1/desk sequence excerpt (~30 MB)
# Used by: NB18 (full pipeline with RGB-D)
# ---------------------------------------------------------------------------
TUM_DIR="tum/rgbd_dataset_freiburg1_desk"
if [ ! -d "$TUM_DIR" ]; then
    echo "[2/3] Downloading TUM RGB-D fr1/desk..."
    echo "      Visit https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download"
    echo "      Download 'freiburg1_desk' and extract here."
    echo ""
    echo "      Expected layout:"
    echo "        $TUM_DIR/rgb/   (PNG frames)"
    echo "        $TUM_DIR/depth/ (PNG depth maps)"
    echo "        $TUM_DIR/groundtruth.txt"
    echo ""
    echo "      Skipping automatic download (manual registration)."
    echo ""
else
    echo "[2/3] TUM fr1/desk already present — skipping."
fi

# ---------------------------------------------------------------------------
# Middlebury Stereo — 3 pairs (~5 MB)
# Used by: NB08 (stereo vision)
# ---------------------------------------------------------------------------
MIDDLEBURY_DIR="sample_pairs/middlebury"
mkdir -p "$MIDDLEBURY_DIR"

_is_valid_png() {
    local f="$1"
    [ -f "$f" ] && [ "$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")" -gt 256 ] \
        && [ "$(head -c 8 "$f" | od -An -tx1 | tr -d ' \n')" = "89504e470d0a1a0a" ]
}

_download_middlebury() {
    local scene="$1"
    local im="$2"
    local out="$3"
    local url="https://vision.middlebury.edu/stereo/data/scenes2014/datasets/${scene}-perfect/${im}"
    if command -v curl &>/dev/null; then
        curl -sL "$url" -o "$out" 2>/dev/null || true
    elif command -v wget &>/dev/null; then
        wget -q "$url" -O "$out" 2>/dev/null || true
    fi
}

echo "[3/3] Middlebury stereo pairs (left=im0, right=im1)..."
for scene in Adirondack Piano Playroom; do
    left="$MIDDLEBURY_DIR/${scene}_left.png"
    right="$MIDDLEBURY_DIR/${scene}_right.png"
    if ! _is_valid_png "$left"; then
        echo "      Downloading ${scene} left (im0)..."
        _download_middlebury "$scene" "im0.png" "$left"
    fi
    if ! _is_valid_png "$right"; then
        echo "      Downloading ${scene} right (im1)..."
        _download_middlebury "$scene" "im1.png" "$right"
    fi
    if _is_valid_png "$left"; then
        echo "      ✓ ${scene}_left.png"
    else
        echo "      ✗ ${scene}_left.png missing or invalid (403/network?)"
        rm -f "$left"
    fi
    if _is_valid_png "$right"; then
        echo "      ✓ ${scene}_right.png"
    else
        echo "      ✗ ${scene}_right.png missing — notebooks synthesize a fallback right view"
        rm -f "$right"
    fi
done
echo "      Full dataset: https://vision.middlebury.edu/stereo/data/"
echo ""

echo "=== Done ==="
echo ""
echo "Notebooks that don't need external data (NB01-04, NB13) generate"
echo "synthetic data on-the-fly. See each notebook for details."
