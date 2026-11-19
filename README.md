# RMCS core algorithms and example calibration data

## Scope
Core implementations for Traditional, Hybrid and Pure ML spectroscopy, plus RGB-guided hyperspectral interpolation and optimization. No training scripts, plotting scripts, benchmark suites, saved results or trained models are included.

Authors: Shuoyang Wang, Yulei Zhang, Chengtao Lu, Zhuning Wang, Sijie Pian, Yubin Gao, Maowei Liang, Qianwen Cao, Junkai Li and Yaoguang Ma. Zhejiang University.

## Data
The example comprises 101 complete RGB TIFF frames at **+5 degrees**, nominal wavelengths 550–650 nm in 1-nm increments. Each frame contains the complete 100-channel field. Original file bytes and pixels are preserved; frames are not cropped, resized or normalized. Exposure recorded in the acquisition configuration is 30,000 microseconds. The source setting is 100; its physical unit is unspecified.

Download all `calibration-5deg-part*.zip` release assets and extract each independently into `data/`. These are independent ZIPs, not multipart volumes. Alternatively, pass `--archives PATH_TO_RELEASE_ASSETS` to the calibration command to read ZIPs without extracting them. `data/labels.csv` maps filenames to nominal wavelength, angle, exposure, image dimensions and SHA-256. No multi-peak or held-out test data are included. Wavelength labels do not establish independent wavelength stability.

## Run single-angle physical calibration
Use Python 3.10 or newer and install `pip install numpy scipy Pillow`. From this directory run:

```sh
python calibrate.py --data-root data --calibration-out local_calibration.npz
python calibrate.py --data-root data --calibration local_calibration.npz --image PATH_TO_MEASUREMENT.tiff
```

The standalone entry point fits the 101 single-line responses at the known +5-degree angle, using nominal wavelength-bin coefficients. Supply an independently acquired image with matching geometry and exposure for reconstruction. Optionally use `--roi x1 y1 x2 y2` consistently in both calls for a chosen channel; no unverified channel coordinates are assumed. The default integrates information across the full frame. Feature resizing occurs only during execution; released images remain unchanged. Set `--exposure-us` to the actual measurement exposure. No output spectra are saved unless the user does so; reconstruction prints wavelength/intensity pairs. The saved calibration is generated locally, not a bundled trained model.

This minimal single-angle NNLS entry point illustrates physical calibration and reconstruction. It is not the complete angle-search benchmark or a substitute for the measured finite-linewidth targets used in the paper. Calibration on the same frame is only a self-consistency check, not independent validation.

On Windows with Intel MKL and PyTorch, set `MKL_THREADING_LAYER=SEQUENTIAL` before starting Python if conflicting OpenMP runtimes are reported. Do not enable duplicate OpenMP runtimes.

## Core source map
- `rmcs/runtime_v6.py`: the existing three-method inference logic, physical action selection and conditional BM3D/unsharp feedback.
- `rmcs/shared_runtime.py`, `shared_physics.py`, `compact_physics.py`: shared physical representation, matching and inversion.
- `rmcs/mobile_networks.py`: TinyAngleCNN architecture (32 x 32 input).
- `rmcs/physics.py`, `data.py`, `common.py`: minimal numerical and preprocessing dependencies.
- `rmcs/imaging.py`: positive cubic B-spline mixing and bounded RGB-guided spectral-tilt optimization, extracted from the existing imaging calculation; historical paths, plotting and export code are removed. It expects a registered initial spectral cube, linear RGB image, response matrix, regional gains and boundaries supplied by the caller. Imaging data are not included.

The spectroscopy core is extracted from the existing portable RMCS framework; the imaging core is extracted from the existing spline-mixing implementation. Learned angle weights, linear spectral weights, selector and deployment policy artifacts are excluded. Hybrid/Pure ML therefore require caller-supplied learned artifacts. This single-angle dataset cannot train or validate the full angle classifier or reproduce all paper results.

## Status and reuse
Prepared for a **private** GitHub repository. No upload, public availability, DOI or repository URL is claimed. A software/data license has not yet been selected. No full manuscript or Supplementary Information is bundled.
