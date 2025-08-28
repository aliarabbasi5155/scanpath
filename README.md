# Scanpath for Stroop — Multimodal IRL (EEG + Eye‑Tracking)

Code accompanying the paper **“Transforming Stroop Task Cognitive Assessments with Multimodal Inverse Reinforcement Learning” (Smart Health, 2025)**. This repo adapts an IRL‑based scanpath framework to the **Stroop task**, integrating **EEG** with **eye‑tracking** to infer the latent reward and the viewing policy that generate human gaze behavior.

> **Heads‑up:** This is **not** the original COCO‑Search18 visual‑search project anymore. It is a redesigned pipeline for **Stroop cognitive assessment** with **EEG‑conditioned scanpath generation**.

---

## Highlights
- **Multimodal IRL:** Learn a gaze policy and reward conditioned on visual context and EEG features.
- **EEG‑conditioned scanpaths:** Generate fixation sequences aligned with cognitive state during Stroop.
- **Dynamic Contextual Belief (DCB) maps:** Use precomputed belief maps as state features when needed.
- **Full train/eval/vis scripts:** Reproducible experiments and visualizations.
- **PyTorch implementation:** CPU/GPU friendly.

---

## Repository structure
```
scanpath/
├─ hparams/                     # YAML configs & hyperparameters
├─ images/                      # Example stimuli (optional)
├─ irl_dcb/                     # Precomputed DCB tensors (HR/LR)
├─ trained_models/              # Trained checkpoints (optional)
├─ assets/                      # Logs, samples, figures
├─ dataset.py                   # Dataset class and I/O
├─ train.py                     # Training entry‑point
├─ eeg_test_scanpath_gen.py     # EEG‑conditioned scanpath generation (single sample)
├─ customized_scanpath_gen.py   # Core custom scanpath utils
├─ new_customized_test_scanpath_gen.py      # Batch/custom inference
├─ customized_test_scanpath_gen_real_data.py# Inference on real‑world data
├─ extract_DCBs_demo.py         # Demo for computing/loading DCBs
├─ plot_scanpath.py             # Plot fixations/scanpaths over images
├─ fixation_data.json           # Demo fixation structure
├─ entries_corrected.json       # Demo metadata/labels
├─ handson.ipynb                # Hands‑on walkthrough (demo)
└─ requirements.txt             # Dependencies
```
*(Some file names/paths may evolve; check `git log` and `--help` of each script.)*

---

## Installation
**Requirements**
- Python 3.8+ (recommended 3.10)
- PyTorch (choose build matching your CUDA)
- Other packages in `requirements.txt`

```bash
# clone and create a virtual env
git clone https://github.com/aliarabbasi5155/scanpath.git
cd scanpath
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data layout
This project expects **Stroop stimuli**, **EEG signals** time‑aligned to trials/fixations, and (optionally) **DCB maps**.
```
data/
└─ stroop/
   ├─ images/                  # stimuli (PNG/JPG)
   ├─ eeg/                     # EEG files (e.g., .npy/.csv/.mat), aligned to trials
   ├─ fixations/               # human scanpaths (JSON)
   │   └─ fixation_data.json
   └─ meta/                    # trial/task metadata
       └─ entries_corrected.json

DCBs/
├─ HR/                         # high‑resolution belief maps per image
└─ LR/                         # low‑resolution belief maps per image
```
If you do not have DCBs yet, see the next section.

---

## (Optional) Compute DCBs
If your pipeline uses DCB features, precompute them with:
```bash
python extract_DCBs_demo.py --images data/stroop/images --out_dir DCBs --device cuda:0
```
This produces `HR/` and `LR/` tensors per image under `DCBs/`.

---

## Training
Train the IRL model on Stroop data using a YAML config under `hparams/`:
```bash
python train.py hparams/irl_stroop.yaml data/stroop --cuda=0
```
- The YAML should define data paths, optimization settings, feature toggles (e.g., EEG usage), and output directories.
- Checkpoints/logs will be saved to the configured output directory (e.g., `trained_models/` or `runs/...`).

---

## EEG‑conditioned generation / evaluation
Generate scanpaths for a single EEG+image pair:
```bash
python eeg_test_scanpath_gen.py \
  --eeg   data/stroop/eeg/subject_01_trial_15.npy \
  --image data/stroop/images/000123.png \
  --ckpt  trained_models/irl_stroop.ckpt \
  --out_dir outputs/subject01_trial15
```
Batch/custom evaluation:
```bash
python new_customized_test_scanpath_gen.py --config hparams/inference_stroop.yaml --out_dir outputs/custom_eval
```
Inference on real‑world data:
```bash
python customized_test_scanpath_gen_real_data.py --config hparams/inference_real.yaml --out_dir outputs/real_eval
```

---

## Visualization
Plot predicted (or human) scanpaths over stimuli:
```bash
python plot_scanpath.py --fixation_path outputs/subject01_trial15/000123_scanpath.json --image_dir data/stroop/images
```

---

## Metrics & reporting
- **FixOnTarget**, **time/steps‑to‑target**, **scanpath length**.
- Optional similarity metrics (e.g., sequence/area‑based, MultiMatch‑style) depending on your setup.
- Compare **congruent vs. incongruent** Stroop conditions and corresponding **EEG profiles**.

Evaluation scripts emit JSON/CSV suitable for statistical analysis and plotting.

---

## Reproducing the paper
1. Prepare and align Stroop stimuli, EEG, and fixations.
2. (Optional) Precompute DCBs for stimuli.
3. Train IRL with the paper configuration.
4. Generate EEG‑conditioned scanpaths and compute metrics.
5. Visualize and report quantitative/qualitative results.

A short walkthrough is available in `handson.ipynb`.

---

## Citation
If you use this code, please cite:
```bibtex
@article{Abbasi2025StroopIRL,
  title   = {Transforming Stroop Task Cognitive Assessments with Multimodal Inverse Reinforcement Learning},
  author  = {Abbasi, Ali and Gong, Jiaqi and Korivand, Soroush},
  journal = {Smart Health},
  volume  = {36},
  pages   = {100567},
  year    = {2025},
  doi     = {10.1016/j.smhl.2025.100567}
}
```

---

## Acknowledgements
This repository was initially cloned from an IRL scanpath project and then substantially re‑designed for the **Stroop + EEG** setting. Thanks to the authors/maintainers of the prior IRL codebase for their groundwork.

## License
Released under the **MIT License** (see `LICENSE`).

## Contributing
- Please open an Issue for bugs/feature requests.
- Pull Requests are welcome (follow code style and include minimal tests where applicable).
