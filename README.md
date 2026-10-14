<p align="center">
  <img src="./src/xlife.png" width="300" alt="X-Life System Logo">
</p>

<h1 align="center">X-Life</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Unity 2022.3](https://img.shields.io/badge/Unity-2022.3.x-000000?logo=unity&logoColor=white)](https://unity.com/)
[![Citation](https://img.shields.io/badge/Citation-CITATION.cff-lightgrey)](./CITATION.cff)

<p align="center">
  <strong>X-Life</strong> is a metabolic world-model system for adaptive lifestyle medicine that translates continuous physiological data and sensing into personalised, quantitative dietary and postprandial activity prescriptions, concurrently meeting individual lifestyle preferences, contextual settings and safety constraints.
  
## 📋 Table of Contents

- [📋 Table of Contents](#-table-of-contents)
- [🌟 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
  - [System Requirements](#system-requirements)
    - [Software Dependencies](#software-dependencies)
    - [Hardware Requirements](#hardware-requirements)
  - [Installation](#installation)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Create Python Environment](#2-create-python-environment)
    - [3. Install Python Dependencies](#3-install-python-dependencies)
    - [4. Install Neo4j Service](#4-install-neo4j-service)
- [🏗️ Core Modules](#️-core-modules)
  - [1. 🕸️ Lifestyle Prescription Generation Module](#1-️-lifestyle-prescription-generation-module)
    - [Configuration Steps](#configuration-steps)
    - [Run Pipelines](#run-pipelines)
  - [2. 🧠 Diet/Exercise Sub-world model and Ranking Module](#2--dietexercise-sub-world-model-and-ranking-module)
  - [3. 🛡️ Safety Module](#3-️-safety-module)
  - [4. 🧬 Multi-omics Model](#4--multi-omics-model)
  - [5. 🥽 AR Deployment](#5--ar-deployment)
    - [Hardware / Platform](#hardware--platform)
    - [Software Environment](#software-environment)
    - [Dependencies](#dependencies)
      - [A. Vuplex 3D WebView (Required, Not Included)](#a-vuplex-3d-webview-required-not-included)
      - [B. XREAL XR Plugin (Required, Provided in the External Unity Archive)](#b-xreal-xr-plugin-required-provided-in-the-external-unity-archive)
      - [C. Unity Package Manager Dependencies (Auto-installed)](#c-unity-package-manager-dependencies-auto-installed)
    - [Project Setup](#project-setup)
      - [1 Configure the WebView Initial URL](#41-configure-the-webview-initial-url)
      - [2 (Optional) Configure iFLYTEK Speech Recognition](#42-optional-configure-iflytek-speech-recognition)
    - [Build APK](#build-apk)
      - [Pre-build Checklist](#pre-build-checklist)
      - [Build Steps in Unity (Android)](#build-steps-in-unity-android)
- [📊 Minimal Dataset](#-minimal-dataset)
- [📄 License](#-license)

## 🌟 Overview

This repository contains the Python code and minimal dataset for **"A metabolic world model system for personalised lifestyle medicine"**. The Unity AR client is distributed separately as `XLIFE-AR.zip`.


## 🚀 Quick Start

### System Requirements

#### Software Dependencies
- **Python**: 3.10+
- **PyTorch**: 2.7.0
- **JDK**: 17.0.18+
- **Unity3D**: 2022.3.x
- **Neo4j**: 5.26.21+

#### Hardware Requirements
- **Training**: NVIDIA A800 or equivalent GPU
- **AR Deployment**: Android 14+ compatible device

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/2025XLife/X-Life.git
cd X-Life
```

#### 2. Create Python Environment
```bash
# Using conda (recommended)
conda create -n xlife python=3.10 -y
conda activate xlife
```

#### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Install Neo4j Service
```bash
# Install JDK
sudo apt install openjdk-17-jdk
# Verify JDK version
java -version
```

Download the Neo4j Community 5.26.21 tarball from the [Neo4j Deployment Center](https://neo4j.com/deployment-center/?gdb-selfmanaged) and start the service:
```bash
tar zxf neo4j-community-5.26.21-unix.tar.gz
cd neo4j-community-5.26.21/bin/
./neo4j start
```

## 🏗️ Core Modules


### 1. 🕸️ Lifestyle Prescription Generation Module

This module generates diet and exercise prescriptions from a hybrid graph and vector knowledge base.

#### Configuration Steps

**1. Configure Database and Models**
Install the agent dependencies and copy the configuration template:

```bash
pip install -r requirements-agents.txt
cp kg_agents/config.example.json kg_agents/config.json
```

Configure the Neo4j database, local DeepSeek-R1 service and model paths in `./kg_agents/config.json`:

```json
{
    "neo4j": {
        "uri": "bolt://127.0.0.1:7687",
        "username": "your_username",
        "password": "your_password"
    },
    "api_model": {
        "api_key": "local",
        "base_url": "http://localhost:8000/v1",
        "model": "DeepSeek-R1"
    },
    "local_model_path": "",
    "local_emb_path": "your_local_embedding_model_path",
    "world_model": {
        "checkpoint_path": "outputs/user/inference.pt",
        "config_path": "configs/local.json",
        "device": "auto"
    }
}
```

**2. Prepare Guidelines**
Prepare guidelines following the descriptions in our paper:
```
kg_agents/
└── data/
    ├── diet/
    │   ├── diet_guideline_1.pdf
    │   └── ...
    └── exer/
        ├── exercise_guideline_1.pdf
        └── ...
```

**3. Extract and Embed Entities**
Make sure the Neo4j server has started:
```bash
cd kg_agents
# Extract knowledge from ./data
python -m core.build_kg_2_steps
# Import knowledge graph
python -m core.import_kg
# (Optional) Embedding knowledge graph
python -m core.embed_kg
```

#### Run Pipelines

**Generation and Assessment Pipelines**
```bash
# Run these commands from the repository root.
cd kg_agents
# Run diet prescription pipeline
python -m pipeline.diet_pipeline --bn 1 --vn 5 --query "I want a sandwich with just veggies, no meat." --use_vector --rag_topk 5
# Run exercise prescription pipeline
python -m pipeline.exer_pipeline --bn 1 --vn 4 --query "I want to do some back exercises at the gym." --use_vector --rag_topk 5
```

**Start Flask Service**

Set `world_model.checkpoint_path` to the trained ranking checkpoint. For `/diet/generate` and `/exercise/generate`, include the current event as `world_model_context` with `subject_id`, `metadata`, `sleep_data` and `cgm_pre`.

```bash
cd kg_agents
python server.py
```

**API Requests**
```bash
# Diet prescription generation
curl -X POST http://localhost:5000/api/v1/diet/generate-only \
  -H "Content-Type: application/json" \
  -d '{"user_metadata":{},"user_requirement":{},"user_query":"a balanced lunch"}'

# Exercise prescription generation
curl -X POST http://localhost:5000/api/v1/exercise/generate-only \
  -H "Content-Type: application/json" \
  -d '{"user_metadata":{},"user_requirement":{},"user_query":"a moderate walk"}'

# Safety module assess prescription
curl -X POST http://localhost:5000/api/v1/safety/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "plan_type": "diet",
    "user_metadata": {},
    "plan": {}
  }'
```

Prompt examples for generation, assessment and knowledge-graph extraction are in [kg_agents/sample_prompts.md](./kg_agents/sample_prompts.md).

### 2. 🧠 Diet/Exercise Sub-world model and Ranking Module

Diet/Exercise sub-world models simulate future glucose state and Ranking module ranks candidate diet and exercise prescriptions.

**1. Data Preparation**

Prepare participant metadata, meal images and descriptions, exercise records and CGM trajectories using [minimal_dataset/sample_data.json](./minimal_dataset/sample_data.json). Add training annotations and preference pairs when running encoder pre-training or alignment.

**2. Model Configuration**

Copy the configuration template and set the local Qwen3-8B and Qwen2.5-VL-72B-Instruct paths, image directory and training parameters:

```bash
cp configs/local.example.json configs/local.json
```

> **Note.** The diet action encoder uses a Qwen2.5-VL-72B-Instruct backbone fine-tuned on structured dietary information, including nutrition analysis, ingredients and preparation methods.

**3. Training**

Run these commands from the repository root:

```bash
# Encode structured training annotations with local Qwen3
python 'world_model+ranking_module/encode_annotations.py' \
  --input data/annotations.json --output data/train.json \
  --model-path /path/to/local/Qwen3-8B --image-dir data/images --device cuda

# Pre-train the action encoders
python 'world_model+ranking_module/train.py' --config configs/local.json \
  --phase encoder_pretraining --data data/train.json --output outputs/encoders

# Train the world model
python 'world_model+ranking_module/train.py' --config configs/local.json \
  --phase world_model --data data/train.json \
  --init-checkpoint outputs/encoders/checkpoint.pt --output outputs/world
```

**4. Multi-stage Alignment**

Preference alignment proceeds through `professional`, `physiological` and `user` stages:

```bash
python 'world_model+ranking_module/train.py' --config configs/local.json \
  --phase preference_alignment --stage professional --data data/train.json \
  --pairs data/professional_pairs.json --init-checkpoint outputs/world/checkpoint.pt \
  --output outputs/professional
```

For physiological and user alignment, select the corresponding `--stage`, set `--participant-id`, initialize from the previous stage and pass earlier preference files through `--replay`. Use `--phase participant_update` for later updates and `--export-inference` to save inference weights.

Run `python 'world_model+ranking_module/train.py' --help` for available options.

### 3. 🛡️ Safety Module

The safety module uses retrieval-augmented generation and explicit safety constraints. Configure the local DeepSeek-R1 service in `kg_agents/config.json`. Prompts are in [kg_agents/sample_prompts.md](./kg_agents/sample_prompts.md).

### 4. 🧬 Multi-omics Model

Structured omics annotations provide auxiliary targets during world-model training. The omics prediction head is removed at inference. 

### 5. 🥽 AR Deployment

The external Unity project targets **XREAL AR** devices. It embeds the web UI with **Vuplex 3D WebView**, captures images and video with the **XREAL RGB Camera**, and sends the results back to the web page through `postMessage`.

#### Hardware / Platform

- Target platform: Android
- Device: XREAL devices / runtimes that support XREAL XR Plugin (and any Android device that can install an APK)
- Debugging: USB cable + USB debugging enabled (required for Build & Run)


#### Software Environment

- **Unity**: `2022.3.x` (this project uses `2022.3.61t8` in `ProjectSettings/ProjectVersion.txt`)
- **Unity Hub modules**: Android Build Support (SDK / NDK / OpenJDK)
- **Optional**: `adb` (usually not needed if you use Unity's Build & Run; needed for CLI install/debugging)
- **Download**: Download the complete Unity project [XLIFE-AR.zip](https://drive.google.com/file/d/1Ul-z-lDQSRvFw9aqk6uxIVhMamKpvAJM/view?usp=sharing)

#### Dependencies

The Python and Neo4j modules are separate from the Unity client. The external `XLIFE-AR.zip` archive contains the Unity project; `Assets/`, `Packages/` and `ProjectSettings/` are not included here.

> Vuplex must be purchased and imported separately. The XREAL XR Plugin is included in the external archive.

##### A. Vuplex 3D WebView

- Purchase and import (Unity Asset Store): [Vuplex 3D WebView for Android (with Gecko Engine)](https://assetstore.unity.com/packages/tools/network/vuplex-3d-webview-for-android-with-gecko-engine-245867)
- After importing, you should see: `Assets/Vuplex/WebView/...`, and be able to use `CanvasWebViewPrefab` in scenes
- Note: if you want to run WebView directly in the macOS/Windows Editor, you may need to install the native plugins for those platforms per Vuplex docs; building for Android only is fine

##### B. XREAL XR Plugin

- Included in the external `XLIFE-AR.zip` archive: `Packages/com.xreal.xr.tar.gz`
- The external Unity project's `Packages/manifest.json` references the local package: `"com.xreal.xr": "@com.xreal.xr.tar.gz"`
- To upgrade the plugin: replace `Packages/com.xreal.xr.tar.gz` and keep the `manifest.json` reference consistent

##### C. Unity Package Manager Dependencies

After opening the project, Unity resolves the dependencies in `Packages/manifest.json` automatically.

#### Project Setup

##### 1. Configure the WebView Initial URL

1. Open the scene: `Assets/Scenes/WebView.unity`
2. Select `CanvasWebViewPrefab` in the scene hierarchy
3. Set `InitialUrl` to your web URL
   - Current example value: `https://xlife.chat/MainPage/AR`

##### 2. (Optional) Configure iFLYTEK Speech Recognition

Required for speech recognition.

The speech recognition config file is: `Assets/Resources/XunfeiConfig.json`

1. Register at [iFLYTEK Open Platform](https://www.xfyun.cn/) and create an application
2. Obtain your API credentials (appId, apiKey, apiSecret)
3. Edit `Assets/Resources/XunfeiConfig.json` and fill in:
   - `appId`: Your application ID
   - `apiKey`: Your API Key
   - `apiSecret`: Your API Secret

Alternatively, create a ScriptableObject config in Unity:

- Right-click in Project window → Create → XLIFE AR → Xunfei Speech Recognition Config
- Set the credentials in the Inspector
- Place the config asset in `Assets/Resources/` folder

#### Build APK

##### Pre-build Checklist

- Vuplex is imported correctly (otherwise you will see compilation errors like missing `Vuplex.WebView` namespace)
- `Packages/com.xreal.xr.tar.gz` exists and Unity Package Manager shows no errors
- Scenes are included in Build Settings (this project should already be configured, but double-check):
  - `Assets/Scenes/WebView.unity`
  - `Assets/Scenes/RGBCameraAndCapture.unity`

##### Build Steps in Unity (Android)

1. `File -> Build Settings...`
2. Select `Android`, then click `Switch Platform`
3. In `Scenes In Build`, make sure `WebView` and `RGBCameraAndCapture` are enabled (recommended: put `WebView` first)
4. `Edit -> Project Settings -> XR Plug-in Management`
   - In the `Android` tab, enable **XREAL** (XREAL XR Loader)
5. `Player Settings -> Other Settings` (recommended values; adjust as needed)
   - `Scripting Backend`: IL2CPP
   - `Target Architectures`: ARM64
6. Connect an Android device (Developer Mode + USB debugging enabled), then click `Build And Run`
   - Or click `Build` to export an `.apk` and install it manually

> Note: this project includes a custom manifest at `Assets/Plugins/Android/AndroidManifest.xml`, which declares network/microphone/screen-capture related permissions, including the Android 14+ foreground service permission `FOREGROUND_SERVICE_MEDIA_PROJECTION`.

## 📊 Minimal Dataset

The minimal dataset is available at [`./minimal_dataset/`](./minimal_dataset/). It contains three pseudonymous participants with:

- 2 meal events and four meal photographs;
- 2 exercise events with pre/post-event CGM trajectories;
- 2 nocturnal windows with rule-derived hypoglycemia labels;
- `manifest.json` with event counts and image hashes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
