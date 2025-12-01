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


# ------------------------------------------------------------------
# 4. API ENDPOINT CLASS (Modal Service)
# ------------------------------------------------------------------
@app.cls(
    gpu="H100",  # H100 required for FP8 support (Evo2 requirement)
    volumes={mount_path: volume},
    timeout=600,
    max_containers=3,
    retries=2,
    scaledown_window=120
)
class Evo2Model:
    @modal.enter()
    def setup(self):
        from evo2 import Evo2
        import os
        
        # Move to repo dir to ensure relative paths (like notebooks) work if needed
        os.chdir("/root/evo2")
        
        print("Loading Evo2 model (7B)...")
        self.model = Evo2("evo2_7b")
        print("Model loaded successfully.")

    @modal.fastapi_endpoint(method="POST")
    def analyze_single_variant(self, request: VariantRequest):
        print(f"Received Request: {request}")

        # 1. Fetch Sequence from UCSC
        window_seq, seq_start = get_genome_sequence(
            request.variant_position, request.genome, request.chromosome
        )

        # 2. Align Variant Position
        rel_pos = request.variant_position - 1 - seq_start

        # Validation
        if rel_pos < 0 or rel_pos >= len(window_seq):
             raise ValueError(f"Position {request.variant_position} is outside the fetched window.")
        
        reference = window_seq[rel_pos]
        print(f"Reference at pos {request.variant_position} is '{reference}'")

        # 3. Run Analysis
        result = analyze_variant(
            rel_pos, reference, request.alternative, window_seq, self.model
        )

        result["position"] = request.variant_position
        return result