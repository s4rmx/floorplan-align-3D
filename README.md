# floorplan-align-3D

**Walk-to-floorplan pipeline for 360° construction-site videos.**

Convert an Insta360 walkthrough video into a 2D trajectory and wall map aligned to an architectural floorplan — no GPS, no markers.

---

## What This Does

Given:
- An **Insta360 360° video** (equirectangular, e.g. 1920×960) of a person walking through a building
- The embedded **IMU data** (accelerometer + gyroscope at 200 Hz)
- A **rasterised floorplan** (PNG)

Produce:
- A **camera trajectory** overlaid on the floorplan
- A **Bird's Eye View (BEV) wall map** aligned to the floorplan

---

## Repository Structure

```
floorplan-align-3D/
│
├── vio/                          # Phase 1–3: IMU PDR, classical VIO, MASt3R
│   ├── insv.py                   # Insta360 INSV binary parser (IMU extraction)
│   ├── pdr.py                    # Pedestrian dead reckoning (gyro heading + step detection)
│   ├── vio.py                    # Classical VIO: optical flow + IMU, no optimisation
│   ├── mast3r_odom.py            # MASt3R sparse global alignment odometry
│   ├── mast3r_backend.py         # MASt3R backend wrapper
│   ├── backend.py                # Backend factory (classical | mast3r)
│   ├── align.py                  # Sim(2) / Umeyama similarity alignment
│   ├── floorplan.py              # Floorplan wall extraction + free-space scoring
│   ├── optimize.py               # Post-alignment wall-avoidance refinement
│   ├── click_align.py            # Interactive landmark click UI
│   ├── run_walk.py               # Entry point: IMU PDR → floorplan overlay
│   └── run_align.py              # Entry point: VIO/MASt3R → align → overlay
│
├── stella_pipeline/              # Phase 4–9: StellaVSLAM + SAM3 + BEV + alignment
│   ├── src/
│   │   ├── 00_sam3_roof_mask.py  # SAM3 roof/ceiling mask (per-face cubemap prompting)
│   │   ├── 00_run_sam3_roof.sh   # Shell wrapper to run SAM3 in Docker
│   │   ├── 00_smoke_sam3_roof.sh # Quick smoke test for SAM3 mask
│   │   ├── 01_run_stella.sh      # Shell wrapper to run StellaVSLAM in Docker
│   │   ├── 02_bev_extract.py     # 3D point cloud → 2D BEV occupancy grid
│   │   ├── 03_auto_align.py      # Sim(2) alignment: BEV → floorplan
│   │   ├── 04_overlay.py         # Trajectory + BEV overlay visualisation
│   │   ├── 05_click_align.py     # Interactive 3-click correspondence UI
│   │   ├── 06_hull_scan_align.py # Hull-based auto alignment (experimental)
│   │   ├── 07_start_fit_align.py # Start-pinned alignment (1-click + geometry)
│   │   ├── common.py             # Shared utilities
│   │   ├── run_pipeline.py       # Full end-to-end pipeline runner
│   │   └── smoke_test.py         # Smoke test (no Stella required)
│   ├── config/
│   │   └── pipeline.yaml.example # Config template — copy to pipeline.yaml, fill paths
│   ├── outputs/                  # Pipeline outputs (gitignored except .gitkeep)
│   └── requirements.txt          # Python dependencies
│
└── docs/
    └── technical_report.md       # Full technical report: approach history, results, lessons
```

---

## Approach History (Short Version)

This project evolved through 9 phases. See [`docs/technical_report.md`](docs/technical_report.md) for the full story with metrics and visual outputs per phase.

| Phase | Approach | Key Limitation |
|-------|----------|---------------|
| 1 | IMU Pedestrian Dead Reckoning | Gyro drift, no map |
| 2 | Classical VIO (optical flow + IMU) | No optimisation → drift, scale errors |
| 3 | MASt3R dense stereo | No dense map, sparse trajectory only |
| 4 | StellaVSLAM | Ceiling features flood the BEV |
| 5 | SAM3 roof mask | Needed: removes ceiling before SLAM |
| 6 | StellaVSLAM + roof mask | Clean BEV wall map ✓ |
| 7 | Manual 3-click alignment | Works but requires user input |
| 8 | Convex hull auto-alignment | Fails on incomplete buildings |
| 9 | Start-pinned alignment (1 click) | Best current result, scale prior needed |

---

## Setup

### Python dependencies (VIO + stella_pipeline)

```bash
pip install -r stella_pipeline/requirements.txt
# Also needed for VIO: scipy, matplotlib, pillow (already in the requirements)
```

### Docker images (stella_pipeline only)

```bash
# StellaVSLAM dense — build from source:
cd ~/stella-vslam-dense/stella_vslam_dense
docker build -t stella_vslam_dense -f Dockerfile.viser .

# SAM3 — build the sam3-pipeline image per your local setup
# Point sam3_ckpt in pipeline.yaml to your sam3.pt checkpoint
```

### Config

```bash
cp stella_pipeline/config/pipeline.yaml.example stella_pipeline/config/pipeline.yaml
# Edit pipeline.yaml — fill in all /path/to/... entries
```

---

## Running

### VIO pipeline (no Docker required)

```bash
# 1. Extract IMU + run PDR (IMU-only dead reckoning)
python vio/run_walk.py --insv /path/to/video.insv --floorplan /path/to/floorplan.png

# 2. Run classical VIO + align to floorplan
python vio/run_align.py \
  --insv /path/to/video.insv \
  --floorplan /path/to/floorplan.png \
  --run my-run

# 3. Use MASt3R instead of classical VIO
python vio/run_align.py \
  --insv /path/to/video.insv \
  --floorplan /path/to/floorplan.png \
  --backend mast3r \
  --mast3r-fps 2 --mast3r-max-frames 50
```

### StellaVSLAM pipeline (Docker required)

```bash
# Full end-to-end
python stella_pipeline/src/run_pipeline.py \
  --run my-run \
  --equirect /path/to/video.mp4 \
  --imu /path/to/imu.csv \
  --floorplan /path/to/floorplan.png

# Skip Stella if already run, resume from BEV onwards
python stella_pipeline/src/run_pipeline.py --run my-run --skip-stella

# Skip Stella + BEV, re-run alignment only
python stella_pipeline/src/run_pipeline.py --run my-run --skip-stella --skip-bev --click-align

# 3-click manual alignment UI only
python stella_pipeline/src/05_click_align.py \
  --run-dir stella_pipeline/outputs/my-run \
  --floorplan /path/to/floorplan.png

# Start-pinned alignment (1 click, geometry optimises scale + rotation)
python stella_pipeline/src/07_start_fit_align.py \
  --src-run stella_pipeline/outputs/my-run \
  --out-dir stella_pipeline/outputs/my-run-start-fit \
  --floorplan /path/to/floorplan.png

# Smoke test (no Stella, ~30s)
python stella_pipeline/src/smoke_test.py
```

---

## Key Concepts

**Sim(2) transform** — the 4-DoF similarity transform (scale, rotation, translation) mapping the SLAM world frame to floorplan pixels. SLAM is up-to-scale, so scale must be recovered from correspondences.

**BEV (Bird's Eye View)** — 3D point cloud projected to XY plane, binned into a 2D occupancy grid. Provides a top-down wall map.

**SAM3 roof mask** — SAM3 is prompted with ceiling/roof text on each cubemap face to generate a binary mask. StellaVSLAM ignores features in masked pixels, keeping the point cloud wall-only.

**Distance transform scan matching** — the floorplan is converted to a distance transform (each pixel = distance to nearest wall). BEV points are scored by how close they land to floorplan walls after applying the candidate Sim(2).

**Start-pinned alignment** — fixes the trajectory start to a 1-click floorplan point, then grid-searches scale and rotation using BEV wall chamfer + free-space scoring.

---

## Requirements

- Python ≥ 3.9
- CUDA GPU recommended for MASt3R and SAM3
- Docker for StellaVSLAM and SAM3 containers
- See `stella_pipeline/requirements.txt` for Python packages
