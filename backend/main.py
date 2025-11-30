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

# ------------------------------------------------------------------
# 2. DATA MODELS
# ------------------------------------------------------------------
class VariantRequest(BaseModel):
    variant_position: int
    alternative: str
    genome: str
    chromosome: str

# ------------------------------------------------------------------
# 3. HELPER FUNCTIONS (Business Logic)
# ------------------------------------------------------------------
def get_genome_sequence(position, genome: str, chromosome: str, window_size=8192):
    import requests

    half_window = window_size // 2
    # Convert to 0-based start, 1-based end (UCSC API style)
    start = max(0, position - 1 - half_window)
    end = position - 1 + half_window + 1

    print(f"Fetching genome window: {chromosome}:{start}-{end} ({genome})...")

    api_url = f"https://api.genome.ucsc.edu/getData/sequence?genome={genome};chrom={chromosome};start={start};end={end}"
    response = requests.get(api_url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch genome sequence: {response.status_code}")

    data = response.json()
    if "dna" not in data:
        raise Exception(f"UCSC API Error: {data.get('error')}")

    # Return the sequence (uppercase) and the start position for alignment
    return data["dna"].upper(), start

def analyze_variant(relative_pos, reference, alternative, window_seq, model):
    # Construct the variant sequence by splicing the string
    var_seq = window_seq[:relative_pos] + alternative + window_seq[relative_pos+1:]

    # Score both sequences (Model Forward Pass)
    # We use list inputs ([seq]) as per standard Evo2 usage
    ref_score = model.score_sequences([window_seq])[0]
    var_score = model.score_sequences([var_seq])[0]

    delta_score = var_score - ref_score

    # Hardcoded thresholds from the Evo2 BRCA1 Notebook/Paper
    threshold = -0.0009178519
    lof_std = 0.0015140239
    func_std = 0.0009016589

    if delta_score < threshold:
        prediction = "Likely pathogenic"
        confidence = min(1.0, abs(delta_score - threshold) / lof_std)
    else:
        prediction = "Likely benign"
        confidence = min(1.0, abs(delta_score - threshold) / func_std)

    return {
        "reference": reference,
        "alternative": alternative,
        "delta_score": float(delta_score),
        "prediction": prediction,
        "classification_confidence": float(confidence)
    }