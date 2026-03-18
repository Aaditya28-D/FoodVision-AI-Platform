# FoodVision AI Platform

FoodVision AI Platform is an end-to-end food image intelligence system built for **food classification**, **multi-model inference**, **similar-dish retrieval**, **Grad-CAM explainability**, and **bulk image classification**.

The project is designed so that even a **lazy or low-skill user** can set it up with minimal effort.

## Final user setup flow

```bash
git clone git@github.com:Aaditya28-D/FoodVision-AI-Platform.git
cd FoodVision-AI-Platform
./setup_and_run.sh
```

That is all the user needs to do.

The setup script automatically:
- creates the backend virtual environment
- installs backend Python dependencies
- installs frontend npm dependencies
- downloads the dataset parts from Google Drive
- downloads the model archive from Google Drive
- joins the split dataset archive automatically
- extracts data and models into the correct folders
- verifies required files exist
- launches backend and frontend

---

# Table of Contents

- [What this project is](#what-this-project-is)
- [What the project does](#what-the-project-does)
- [Final benchmark results](#final-benchmark-results)
- [How the system works](#how-the-system-works)
- [Main models used](#main-models-used)
- [Prediction strategies](#prediction-strategies)
- [Project features](#project-features)
- [How setup works](#how-setup-works)
- [What to expect during setup](#what-to-expect-during-setup)
- [Quick start](#quick-start)
- [How to run the app later](#how-to-run-the-app-later)
- [Project structure](#project-structure)
- [Backend overview](#backend-overview)
- [Frontend overview](#frontend-overview)
- [Retrieval system](#retrieval-system)
- [Explainability](#explainability)
- [Bulk classification](#bulk-classification)
- [Data and model assets](#data-and-model-assets)
- [Data cleaning and preparation](#data-cleaning-and-preparation)
- [API overview](#api-overview)
- [Troubleshooting](#troubleshooting)
- [Important notes](#important-notes)
- [Future improvement ideas](#future-improvement-ideas)

---

# What this project is

This project is a food image classification platform based on the **Food-101** dataset.

A user can upload a food image and the system can:

- predict what food it is
- show confidence scores
- use different model strategies
- retrieve similar dishes from the dataset
- generate Grad-CAM visual explanations
- classify many images in bulk

This is not just a single model demo. It is a **full platform** with:
- backend API
- frontend interface
- model management
- retrieval index
- explainability layer
- bulk workflow
- automated setup for non-technical users

---

# What the project does

At a high level, the system takes an uploaded food image and performs one or more of the following:

## 1. Food classification
It predicts which of the 101 food classes the image belongs to.

## 2. Multi-model strategy inference
It supports:
- single-model inference
- ensemble inference
- smart routing between models

## 3. Similar dish retrieval
It retrieves similar food images from the dataset based on embeddings.

## 4. Explainability
It generates Grad-CAM heatmaps showing which image regions influenced the prediction.

## 5. Bulk classification
It allows many images to be classified and organized into class-based folders.

---

# Final benchmark results

Final benchmark results on the **Food-101 test set**:

- **Smart Router**: **86.61%**
- **Ensemble (EfficientNet-B0 + ResNet50)**: **86.04%**
- **EfficientNet-B0**: **84.11%**
- **ResNet50**: **80.60%**
- **MobileNetV3-Large**: **80.13%**

## Final conclusions

- **Best overall system**: **Smart Router**
- **Best ensemble-style strategy**: **Ensemble**
- **Best single model**: **EfficientNet-B0**

---

# How the system works

## Core flow

1. User uploads an image
2. Frontend sends the image to the backend
3. Backend runs the selected prediction strategy
4. The model returns predictions and confidence scores
5. Optional retrieval finds similar dishes
6. Optional explainability produces heatmaps
7. Frontend displays the final result

## System layers

The project is built in layers:

### Frontend
The user-facing web application.

### Backend API
FastAPI-based service that handles:
- prediction
- retrieval
- analysis
- comparison
- model info
- bulk classification

### Inference layer
Runs:
- EfficientNet-B0
- ResNet50
- MobileNetV3-Large
- ensemble logic
- smart router logic

### Retrieval layer
Uses image embeddings and nearest-neighbor similarity.

### Explainability layer
Uses Grad-CAM for visual interpretation.

### Asset/bootstrap layer
Automatically downloads and sets up required large files.

---

# Main models used

The final project keeps 3 important models:

## EfficientNet-B0
- best single-model result
- strongest standalone performer

## ResNet50
- useful for model diversity
- contributes to ensemble performance
- also used for retrieval embeddings

## MobileNetV3-Large
- lightweight alternative
- contributes to smart router diversity

## Models not used as final production strategies
The project keeps focus on the best practical models and does **not** present weaker experimental models as final recommended strategies.

---

# Prediction strategies

The final system supports these prediction strategies:

## `smart`
The best-performing mode.
It uses routing logic across multiple models and chooses the final result intelligently.

## `ensemble`
Combines EfficientNet-B0 and ResNet50 by averaging predictions.

## `efficientnet_b0`
Best single-model option.

## `resnet50`
Single-model ResNet50 inference.

## `mobilenet_v3_large`
Single-model MobileNetV3-Large inference.

## Final recommendation
For normal use:
- use **Smart Router** as default
- use **Ensemble** if you want a strong simpler combined strategy
- use **EfficientNet-B0** if you want the best single model only

---

# Project features

## Food prediction
- upload a single image
- get top-k predicted food classes
- confidence scores returned

## Similar dish retrieval
- retrieves visually similar dataset images
- supports exact-match detection
- returns same-class and other-class results

## Explainability
- Grad-CAM heatmaps
- helps understand what influenced the prediction

## Bulk classification
- many images can be uploaded
- images can be classified and organized into class folders

## Strategy-based analysis
- different inference strategies available
- comparison of strategies and models

## Automated setup
- lazy-user oriented setup
- minimal manual work
- no need to manually arrange dataset folders

---

# How setup works

The project includes a zero-config bootstrap flow.

## Main script
```bash
./setup_and_run.sh
```

This script does all the heavy work.

## What it performs

### 1. Checks prerequisites
It checks whether:
- Python 3 is installed
- Node.js is installed
- npm is installed

### 2. Creates backend virtual environment
If `backend/.venv` does not exist, it creates it automatically.

### 3. Installs backend dependencies
It installs packages from `backend/requirements.txt`.

### 4. Installs frontend dependencies
It installs npm dependencies for the frontend.

### 5. Downloads large assets
It downloads:
- the split data archive parts
- the model archive

### 6. Automatically joins split data parts
The data archive is split into 5 parts for reliability.
The script joins them automatically into:

```text
downloads/data_assets.tar.gz
```

### 7. Extracts assets
It extracts:
- data into `data/...`
- model weights into `backend/models/weights/...`

### 8. Verifies assets
It checks that critical files and folders exist.

### 9. Launches the application
It starts:
- backend server
- frontend dev server

---

# What to expect during setup

When running:

```bash
./setup_and_run.sh
```

these are the normal expected stages.

## Stage 1: environment setup
You will see:
- virtual environment creation
- pip upgrade
- Python package installation

## Stage 2: frontend setup
You will see:
- npm install
- frontend package installation

## Stage 3: asset downloads
You will see:
- 5 data parts downloading one by one
- model archive downloading
- retry messages if a part fails

Because the data is large, this may take time depending on internet speed.

## Stage 4: joining and extraction
You will see:
- data parts being joined into one archive
- data extracted
- models extracted

## Stage 5: verification
You will see:
- asset verification passed

## Stage 6: startup
You will see:
- backend running on `http://127.0.0.1:8000`
- frontend running on `http://localhost:5173`

## Important note
On the very first setup, the process can take quite some time because of:
- package installation
- large data downloads
- archive extraction

Later runs are much faster.

---

# Quick start

## Prerequisites
Make sure your machine has:
- Python 3
- Node.js
- npm
- Git

## Clone and run

```bash
git clone git@github.com:Aaditya28-D/FoodVision-AI-Platform.git
cd FoodVision-AI-Platform
./setup_and_run.sh
```

That is the full first-time setup.

## Access URLs

Frontend:
- `http://localhost:5173`

Backend:
- `http://127.0.0.1:8000`

---

# How to run the app later

After the first setup has completed once, you do **not** need the full setup again.

Use:

```bash
./run_app.sh
```

On macOS you can also use:

```text
start_app.command
```

which can be double-clicked.

---

# Project structure

A simplified project structure:

```text
FoodVision-AI-Platform/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── utils/
│   ├── ml/
│   │   ├── cleaning/
│   │   ├── evaluation/
│   │   ├── explainability/
│   │   ├── inference/
│   │   ├── retrieval/
│   │   └── training/
│   ├── models/
│   │   ├── weights/
│   │   ├── reports/
│   │   ├── histories/
│   │   └── checkpoints/
│   ├── artifacts/
│   └── requirements.txt
├── web/
│   ├── src/
│   └── package.json
├── data/
│   ├── food-101/
│   ├── metadata/
│   ├── embeddings/
│   └── food_info/
├── scripts/
│   ├── download_assets.sh
│   └── verify_assets.py
├── setup_and_run.sh
├── run_app.sh
├── start_app.command
└── README.md
```

---

# Backend overview

The backend is built with **FastAPI**.

It handles:
- prediction requests
- analysis requests
- retrieval requests
- model info listing
- bulk classification workflows

Key backend areas include:

## `backend/app/api`
Route definitions.

## `backend/app/services`
Application-level logic.

## `backend/ml/inference`
Model loading and prediction strategies.

## `backend/ml/retrieval`
Similarity search and retrieval index handling.

## `backend/ml/explainability`
Grad-CAM generation.

## `backend/ml/cleaning`
Dataset preparation and cleaning utilities.

---

# Frontend overview

The frontend is built with **Vite**.

It provides:
- image upload UI
- strategy selection
- confidence results
- analysis summaries
- similar-dish display
- bulk classification interface
- user-friendly descriptions and disclaimers

The frontend is intended for users who are not necessarily technical or ML-aware.

---

# Retrieval system

The retrieval system helps the project go beyond classification.

## How retrieval works

1. The query image is embedded using ResNet50 features
2. The system compares the query embedding against a stored retrieval index
3. Similar images are ranked by similarity
4. Results are split into:
   - same-class results
   - other-class results
   - exact match detection when applicable

## Retrieval index
Stored in:

```text
data/embeddings/food101_resnet50_index.npz
```

This index was rebuilt against the cleaned keep manifest so removed images are excluded.

---

# Explainability

Grad-CAM is used to show where the model is focusing.

## Why it matters
It helps users see:
- which food regions influenced the model
- whether the model focused on the actual dish
- whether confusion may come from background/context

This makes the system more interpretable and more educational.

---

# Bulk classification

The platform includes a bulk workflow.

## Intended use
A user can provide many images and the system can:
- classify them
- organize them by predicted class
- prepare structured output

This is useful for:
- personal photo sorting
- dataset organization
- fast batch food labeling

---

# Data and model assets

Large assets are **not stored in GitHub** directly.

## Why
Because:
- dataset is large
- model weights are large
- GitHub repo should stay code-focused and manageable

## Data assets
The data archive is split into **5 parts** for reliable download.

## Model assets
The model archive is stored separately and extracted into:

```text
backend/models/weights/
```

## Final weight files used
- `efficientnet_b0_best.pth`
- `resnet50_best.pth`
- `mobilenet_v3_large_best.pth`

---

# Data cleaning and preparation

This project includes a full data-preparation effort, not just model training.

The pipeline included:
- suspicious image detection
- dark/blurry sample removal
- confusion-pair cleaning
- expanded outlier mining
- class consistency checking
- rollback analysis
- cleaned split generation
- retrieval index rebuilding

## Important final note
Several experiments were tested.
The final project uses the **best benchmarked practical weights**, rather than endlessly retraining every variation.

That means the project prioritizes **stable final performance** over endless experimentation.

---

# API overview

## `/predict`
Returns:
- top predictions
- confidence values
- selected strategy output

## `/analyze`
Returns:
- prediction
- summary
- profile info
- Grad-CAM/explanation details

## `/retrieve/similar`
Returns:
- similar dishes
- exact match info
- same-class and other-class similar items

## `/models`
Returns:
- available strategies
- default strategy metadata

---

# Troubleshooting

## 1. Setup script says a command is missing
Make sure these are installed:
- Python 3
- Node.js
- npm
- Git

## 2. Download fails for one of the data parts
The setup script retries automatically.
If a part still fails repeatedly:
- rerun `./setup_and_run.sh`

Because downloads are split into parts, a single part failure is easier to recover from than one huge file.

## 3. Asset verification fails
That means one or more required files/folders were not extracted correctly.
Rerun the setup script.

## 4. Backend starts but frontend does not
Check:
- frontend npm dependencies installed correctly
- no local port conflict on 5173

## 5. Frontend starts but backend does not
Check:
- backend virtual environment exists
- Python dependencies installed correctly
- no local port conflict on 8000

## 6. Retrieval fails
Make sure:
- `data/embeddings/food101_resnet50_index.npz` exists
- data assets extracted properly
- setup completed fully

## 7. Script permission issue
Run:

```bash
chmod +x setup_and_run.sh run_app.sh start_app.command scripts/download_assets.sh
```

---

# Important notes

## Do not commit local-only large or sensitive files
Do not commit:
- downloaded archives
- local extracted giant asset folders if not intended
- local virtual environment
- node_modules
- macOS `.DS_Store`
- other temporary files

## External asset strategy
The repo contains code.
Large data and models are downloaded automatically.

## Final recommended default mode
The final recommended default user mode is:

- **Smart Router**

because it gives the best overall benchmark.

---

# Future improvement ideas

Possible improvements for the future:

- replace Google Drive with a more reliable large-file host
- package the app into a desktop-friendly installer
- improve progress UI during setup
- add more resume support for downloads
- add richer result reporting for bulk classification
- improve class-specific confusion handling even more
- add optional cloud deployment mode
- package into Docker if needed for advanced users

---

# Final summary

FoodVision AI Platform is a complete food image intelligence project with:

- food image classification
- smart multi-model inference
- ensemble prediction
- similar dish retrieval
- Grad-CAM explainability
- bulk classification
- data-cleaning pipeline
- zero-config lazy-user setup

## Final practical recommendation
- Use **Smart Router** as default
- Use **Ensemble** as a strong fallback combined strategy
- Use **EfficientNet-B0** as the best single-model option
