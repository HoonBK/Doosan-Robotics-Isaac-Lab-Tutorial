<img width="865" height="424" alt="image" src="https://github.com/user-attachments/assets/61f2ae52-ca99-4c2f-9379-56bb5f98fbf3" /># Doosan-Robotics-Isaac-Lab-Tutorial

This repository explains how to install NVIDIA Isaac Sim and Isaac Lab, and how to train the Doosan M0609 manipulator using Reinforcement Learning.

본 저장소는 NVIDIA Isaac Sim / Isaac Lab 환경에서 두산 로보틱스 M0609 로봇을 강화학습으로 학습하는 전체 과정을 튜토리얼 형태로 정리한 자료입니다.

## Environment

Operating System: Windows 11

GPU: NVIDIA GPU (RTX series recommended)

Python: Conda environment (recommended)

## 1. NVIDIA 그래픽 드라이버 / CUDA 설치

Isaac Sim과 Isaac Lab은 GPU 가속을 필수로 사용하므로, NVIDIA 드라이버와 CUDA가 반드시 필요합니다.

### 1.1 NVIDIA 그래픽 드라이버

NVIDIA 공식 홈페이지에서 최신 드라이버 설치
👉 https://www.nvidia.com/Download/index.aspx

설치 후 아래 명령으로 정상 인식 확인:

```bash
nvidia-smi
```

### 1.2 CUDA Toolkit

Isaac Sim 권장 버전에 맞는 CUDA 설치
(일반적으로 CUDA 11.8 또는 12.x)

CUDA Toolkit 다운로드:
👉 https://developer.nvidia.com/cuda-downloads

설치 후 확인:

```bash
nvcc --version
```

## 2. Isaac Sim 설치

Isaac Sim은 NVIDIA Omniverse 기반 시뮬레이터입니다.


👉 https://docs.isaacsim.omniverse.nvidia.com/latest/installation/quick-install.html

<img width="865" height="424" alt="image" src="https://github.com/user-attachments/assets/a6843f17-a212-452e-98d5-6e9ae26c5fa0" />

해당되는 운영체제에 맞게 다운로드

(주의)최초 실행 시 셰이더 컴파일로 시간이 오래 걸릴 수 있습니다.

## 3. Isaac Lab 설치

Isaac Lab은 Isaac Sim 위에서 동작하는 강화학습/로봇 학습 프레임워크입니다.

### 3.1 Conda 환경 생성
```bash
conda create -n env_isaac_lab python=3.10
conda activate env_isaac_lab
```
### 3.2 Isaac Lab 클론 및 설치
```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
pip install -e .
```

설치 후 예제 실행으로 정상 여부 확인:

python scripts/tutorials/00_sim/create_empty_scene.py

## 4. m0609_cabinet 폴더 위치

다음 경로에 Doosan M0609 강화학습 환경 폴더를 위치시킵니다.

IsaacLab/
└─ source/
   └─ isaaclab_tasks/
      └─ isaaclab_tasks/
         └─ direct/
            └─ m0609_cabinet/


이 폴더에는 다음과 같은 파일이 포함됩니다:

- m0609_cabinet_env.py

- m0609_cabinet_env_cfg.py

- agents/

- __init__.py

## 5. direct/__init__.py 수정

아래 파일을 수정하여 m0609_cabinet 환경이 Gymnasium에 등록되도록 합니다.

경로

IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/__init__.py


수정 내용

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Direct workflow environments.
"""

import gymnasium as gym
from .m0609_cabinet import *
```

## 6. m0609_cabinet_env.py의 USD 파일 경로 수정

Doosan M0609 로봇과 환경에 사용되는 USD 파일 경로를
본인 PC 환경에 맞게 수정해야 합니다.

<img width="1013" height="107" alt="image" src="https://github.com/user-attachments/assets/101a2296-7f3d-44cf-b55c-5008db7141c5" />


⚠️ 경로 오류가 있으면 시뮬레이션이 시작되지 않습니다.

## 7. 학습 실행

Isaac Lab의 기본 디렉토리에서 실행

```bash
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-M0609-Cabinet-Direct-v0 --num_envs 4096 --headless
```

학습 로그와 체크포인트는 다음 경로에 저장됩니다.

IsaacLab/logs/rsl_rl/

## 8. 학습 결과 확인

학습된 policy를 로드하여 시뮬레이터에서 직접 동작 확인

```bash
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-M0609-Cabinet-Direct-v0 --num_envs 1 --checkpoint "C:\Users\HBK\IL\IsaacLab\logs\rsl_rl\m0609_cabinet_direct\2025-12-11_14-07-33\model_1000.pt"
```

해당 명령어처럼 본인이 학습한 pt파일의 경로를 복사하여 실행

