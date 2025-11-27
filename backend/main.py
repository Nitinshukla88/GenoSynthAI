import sys
import modal
from pydantic import BaseModel

# ------------------------------------------------------------------
# 1. CLOUD ENVIRONMENT SETUP (The Image)
# ------------------------------------------------------------------
# We use Micromamba (Conda) to install the strict dependencies 
# required by the new Evo 2 docs (Transformer Engine 2.3.0).
evo2_image = (
    modal.Image.micromamba(python_version="3.12")
    .apt_install(
        "git", "gcc", "g++", "wget", "cmake", "ninja-build"
    )
    # Install NVIDIA CUDA libraries via Conda (Official & Stable)
    .micromamba_install(
        "cuda-nvcc",
        "cuda-cudart-dev",
        channels=["nvidia"]
    )
    # Install Transformer Engine 2.3.0 via Conda (Strict Requirement)
    .micromamba_install(
        "transformer-engine-torch=2.3.0",
        channels=["conda-forge"]
    )
    # Install Python Dependencies (Evo2 + Your requirements.txt)
    .pip_install(
        "flash-attn==2.8.0.post2",  # Strongly recommended by new docs
        "torch",
        "setuptools",
        "gitpython",
        "biopython",    # Required for 'Bio.SeqIO'
        "requests",     # Required for UCSC API calls
        "fastapi[standard]",
        "matplotlib",
        "pandas",
        "seaborn",
        "scikit-learn",
        "openpyxl"      # Required to read the BRCA1 Excel file
    )
    # Clone Evo2 Repo and Install
    .run_commands(
        "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git /root/evo2",
        "cd /root/evo2 && pip install ."
    )
)

app = modal.App("GenoSynthAI-Backend", image=evo2_image)

# Volume to cache downloaded models (saves time on subsequent runs)
volume = modal.Volume.from_name("hf_cache", create_if_missing=True)
mount_path = "/root/.cache/huggingface"