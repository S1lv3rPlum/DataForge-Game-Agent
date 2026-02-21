# Setup Instructions — DataForge-Game-Agent

Setup guide for Matt's PC. Follow these steps in order before running anything.

---

## Step 1 — Clone the Repo

Open VS Code, open the terminal (Ctrl + backtick), and run:

``bash
git clone https://github.com/S1lv3rPlum/DataForge-Game-Agent.git
cd DataForge-Game-Agent

## Step 2 — Check Your Python Version
python --version
You need Python 3.10 or higher. If you're below that, download the latest
Python 3.10+ from python.org before continuing.

## Step 3 — Create a Virtual Environment
This keeps the project's packages separate from everything else on your PC.
python -m venv venv
Then activate it:
venv\Scripts\activate
You should see (venv) appear at the start of your terminal line.
You need to do this activation step every time you open a new terminal.

## Step 4 — Install the Requirements
pip install -r requirements.txt
This will take a few minutes. Let it run.

## Step 5 — Install PyTorch with CUDA Support
This is separate because we need the GPU-specific version for your GTX 1050 Ti.
Run this exact command:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## Step 6 — Verify GPU is Working
Run this in the terminal:
python -c "import torch; print(torch.cuda.is_available())"
You should see True. If you see False stop here and let Renee know
before going any further.

## Step 7 — You're Ready
If Step 6 printed True, the environment is fully set up.
Check back with Renee for the next script to run.

## Troubleshooting
If venv\Scripts\activate gives an error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
If pip install fails on any package, try running VS Code as administrator
If CUDA shows False, make sure your NVIDIA drivers are up to date at nvidia.com
---

## Step 8:
python games/minesweeper/random_agent.py
