#!/usr/bin/env bash
# Fetch all external datasets used by the workshop notebooks.
#
#   [1/3] KITTI odometry seq 00 excerpt — first 200 grayscale frames (via
#         selective HTTP-Range extraction from the official S3 mirror, no
#         registration) + ground-truth poses 00-10.
#   [2/3] TUM RGB-D fr1/desk — direct download (freely redistributable,
#         CC-BY-NC-SA 3.0).
#   [3/3] Middlebury 2014 stereo pairs — 5 scenes with calibration.
#
# Everything is idempotent: completed pieces are skipped on re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

fetch_url() { # fetch_url <url> <out>
    if command -v curl &>/dev/null; then
        curl -fSL --retry 3 "$1" -o "$2"
    elif command -v wget &>/dev/null; then
        wget -q --tries=3 "$1" -O "$2"
    else
        echo "      ✗ need curl or wget" >&2
        return 1
    fi
}

echo "=== Vision & 3D Mapping Workshop — Data Downloader ==="
echo ""

mkdir -p sample_pairs calibration kitti tum

# ---------------------------------------------------------------------------
# KITTI Odometry — sequence 00 excerpt (first 200 frames, ~25 MB)
# Used by: NB07 (VO), NB08 (stereo), NB17 (evaluation), NB18 (full pipeline)
# ---------------------------------------------------------------------------
KITTI_DIR="kitti/sequences/00"
N_KITTI_FRAMES=200
kitti_frames_present() {
    local d="$1/image_0" n=0
    [ -d "$d" ] && n=$(ls "$d" | grep -c '\.png$' || true)
    echo "$n"
}

echo "[1/3] KITTI odometry sequence 00 (grayscale, frames 000000-000199)..."
if [ "$(kitti_frames_present "$KITTI_DIR")" -ge "$N_KITTI_FRAMES" ]; then
    echo "      ✓ images already present ($(kitti_frames_present "$KITTI_DIR") frames)"
else
    echo "      Selective fetch from the official S3 mirror via HTTP Range"
    echo "      requests (data/fetch_kitti_frames.py) — no registration, no 23 GB download."
    PYTHON=""
    [ -x "$PROJECT_DIR/.venv/bin/python" ] && PYTHON="$PROJECT_DIR/.venv/bin/python"
    [ -z "$PYTHON" ] && command -v python3 &>/dev/null && PYTHON="python3"
    if [ -n "$PYTHON" ]; then
        "$PYTHON" "$SCRIPT_DIR/fetch_kitti_frames.py" --frames "$N_KITTI_FRAMES" || true
    else
        echo "      ✗ need python3 — run data/fetch_kitti_frames.py manually"
    fi
    if [ "$(kitti_frames_present "$KITTI_DIR")" -ge "$N_KITTI_FRAMES" ]; then
        echo "      ✓ $(kitti_frames_present "$KITTI_DIR") frames + calib.txt + times.txt"
    else
        echo "      ✗ incomplete — see message above"
    fi
fi

# Ground-truth poses (data_odometry_poses.zip, ~4 KB, public S3 bucket)
if [ -s "kitti/poses/00.txt" ]; then
    echo "      ✓ ground-truth poses already present (00-10)"
else
    echo "      Downloading ground-truth poses (00-10)..."
    if fetch_url "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip" \
        "kitti/data_odometry_poses.zip"; then
        (cd kitti && unzip -oq data_odometry_poses.zip -d poses_tmp)
        # zip layout: dataset/poses/00.txt -> kitti/poses/00.txt
        if [ -d "kitti/poses_tmp/dataset/poses" ]; then
            mv kitti/poses_tmp/dataset/poses/*.txt kitti/poses/
        else
            mv kitti/poses_tmp/poses/*.txt kitti/poses/ 2>/dev/null ||
                mv kitti/poses_tmp/*.txt kitti/poses/ 2>/dev/null || true
        fi
        rm -rf kitti/poses_tmp kitti/data_odometry_poses.zip
        [ -s "kitti/poses/00.txt" ] && echo "      ✓ poses 00-10 installed"
    else
        echo "      ✗ pose download failed (network?) — poses from the KITTI devkit"
        rm -f kitti/data_odometry_poses.zip
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# TUM RGB-D — fr1/desk sequence (~1.4 GB tgz -> ~30 MB of frames used)
# Used by: NB18 (full pipeline with RGB-D)
# ---------------------------------------------------------------------------
TUM_DIR="tum/rgbd_dataset_freiburg1_desk"
echo "[2/3] TUM RGB-D fr1/desk..."
if [ -d "$TUM_DIR/rgb" ] && [ -s "$TUM_DIR/groundtruth.txt" ]; then
    echo "      ✓ already present ($(ls "$TUM_DIR/rgb" | wc -l) rgb frames)"
else
    echo "      Downloading rgbd_dataset_freiburg1_desk.tgz (~1.4 GB)..."
    if fetch_url "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz" \
        "tum/fr1_desk.tgz"; then
        tar -xzf tum/fr1_desk.tgz -C tum
        rm -f tum/fr1_desk.tgz
        if [ -d "$TUM_DIR/rgb" ]; then
            echo "      ✓ extracted: $(ls "$TUM_DIR/rgb" | wc -l) rgb frames, depth/, groundtruth.txt"
        else
            echo "      ✗ unexpected archive layout — inspect $PWD/tum/"
        fi
    else
        echo "      ✗ download failed — fetch manually from"
        echo "        https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download"
        rm -f tum/fr1_desk.tgz
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# Middlebury Stereo 2014 — 5 scenes (~15 MB)
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

echo "[3/3] Middlebury stereo pairs (left=im0, right=im1) + per-scene calibration..."
for scene in Backpack Jadeplant Motorcycle Piano Playroom; do
    left="$MIDDLEBURY_DIR/${scene}_left.png"
    right="$MIDDLEBURY_DIR/${scene}_right.png"
    calib="$MIDDLEBURY_DIR/${scene}_calib.txt"
    if ! _is_valid_png "$left"; then
        echo "      Downloading ${scene} left (im0)..."
        _download_middlebury "$scene" "im0.png" "$left"
    fi
    if ! _is_valid_png "$right"; then
        echo "      Downloading ${scene} right (im1)..."
        _download_middlebury "$scene" "im1.png" "$right"
    fi
    if [ ! -s "$calib" ]; then
        _download_middlebury "$scene" "calib.txt" "$calib" || true
    fi
    if _is_valid_png "$left" && _is_valid_png "$right"; then
        echo "      ✓ ${scene} (left + right + calib)"
    elif _is_valid_png "$left"; then
        echo "      ~ ${scene}: left ok, right missing — notebooks fetch or synthesize it"
        rm -f "$right"
    else
        echo "      ✗ ${scene}_left.png missing or invalid (403/network?)"
        rm -f "$left"
    fi
done
echo "      Full dataset: https://vision.middlebury.edu/stereo/data/"
echo ""

echo "=== Done ==="
echo ""
echo "Notebooks that don't need external data (NB01-04, NB13) generate"
echo "synthetic data on-the-fly, as does NB18 when no dataset is present."
echo "See each notebook for details."
