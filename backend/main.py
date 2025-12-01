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
    

# ------------------------------------------------------------------
# 5. BATCH ANALYSIS FUNCTION (Notebook Replica)
# ------------------------------------------------------------------
@app.function(gpu="H100", volumes={mount_path: volume}, timeout=1200)
def run_brca1_analysis():
    # Imports must be inside the function for Modal execution
    import base64
    from io import BytesIO
    from Bio import SeqIO
    import gzip
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import roc_auc_score, roc_curve
    from evo2 import Evo2

    WINDOW_SIZE = 8192
    print("Loading Evo2 model for batch analysis...")
    model = Evo2('evo2_7b')
    
    # Path to the notebooks data inside the cloned repo
    base_path = "/root/evo2/notebooks/brca1"
    
    print("Reading BRCA1 Dataset...")
    brca1_df = pd.read_excel(f'{base_path}/41586_2018_461_MOESM3_ESM.xlsx', header=2)
    
    # Cleaning Data
    brca1_df = brca1_df[['chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class']]
    brca1_df.rename(columns={'chromosome': 'chrom', 'position (hg19)': 'pos', 'reference': 'ref', 'alt': 'alt', 'function.score.mean': 'score', 'func.class': 'class'}, inplace=True)
    brca1_df['class'] = brca1_df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')

    # Reading Reference Genome (Chromosome 17)
    with gzip.open(f'{base_path}/GRCh37.p13_chr17.fna.gz', "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_chr17 = str(record.seq)
            break

    # Analysis Loop
    ref_seqs = []
    ref_seq_to_index = {}
    ref_seq_indexes = []
    var_seqs = []
    
    # Running on first 500 rows (same as your old code)
    brca1_subset = brca1_df.iloc[:500].copy()

    for _, row in brca1_subset.iterrows():
        p = row["pos"] - 1 
        full_seq = seq_chr17
        ref_seq_start = max(0, p - WINDOW_SIZE//2)
        ref_seq_end = min(len(full_seq), p + WINDOW_SIZE//2)
        ref_seq = seq_chr17[ref_seq_start:ref_seq_end]
        snv_pos_in_ref = min(WINDOW_SIZE//2, p)
        var_seq = ref_seq[:snv_pos_in_ref] + row["alt"] + ref_seq[snv_pos_in_ref+1:]

        if ref_seq not in ref_seq_to_index:
            ref_seq_to_index[ref_seq] = len(ref_seqs)
            ref_seqs.append(ref_seq)
        ref_seq_indexes.append(ref_seq_to_index[ref_seq])
        var_seqs.append(var_seq)

    ref_seq_indexes = np.array(ref_seq_indexes)
    
    print(f'Scoring {len(ref_seqs)} reference sequences...')
    ref_scores = model.score_sequences(ref_seqs)
    
    print(f'Scoring {len(var_seqs)} variant sequences...')
    var_scores = model.score_sequences(var_seqs)

    delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_seq_indexes]
    brca1_subset[f'evo2_delta_score'] = delta_scores

    # Calculate AUROC
    y_true = (brca1_subset['class'] == 'LOF')
    if len(y_true.unique()) > 1:
        auroc = roc_auc_score(y_true, -brca1_subset['evo2_delta_score'])
    else:
        auroc = 0.0
    
    # Calculate Confidence Params (Thresholds)
    y_true = (brca1_subset["class"] == "LOF")
    fpr, tpr, thresholds = roc_curve(y_true, -brca1_subset["evo2_delta_score"])
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = -thresholds[optimal_idx]

    lof_scores = brca1_subset.loc[brca1_subset["class"] == "LOF", "evo2_delta_score"]
    func_scores = brca1_subset.loc[brca1_subset["class"] == "FUNC/INT", "evo2_delta_score"]
    
    confidence_params = {
        "threshold": optimal_threshold,
        "lof_std": lof_scores.std(),
        "func_std": func_scores.std()
    }
    print("Confidence params:", confidence_params)

    # Generate Plot
    plt.figure(figsize=(4, 2))
    p = sns.stripplot(data=brca1_subset, x='evo2_delta_score', y='class', hue='class', order=['FUNC/INT', 'LOF'], palette=['#777777', 'C3'], size=2, jitter=0.3)
    sns.boxplot(showmeans=True, meanline=True, meanprops={'visible': False}, medianprops={'color': 'k', 'ls': '-', 'lw': 2}, whiskerprops={'visible': False}, zorder=10, x="evo2_delta_score", y="class", data=brca1_subset, showfliers=False, showbox=False, showcaps=False, ax=p)
    plt.xlabel('Delta likelihood score, Evo 2')
    plt.ylabel('BRCA1 SNV class')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plot_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {'variants': brca1_subset.to_dict(orient="records"), "plot": plot_data, "auroc": auroc}


@app.function()
def brca1_example():
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    print("Running BRCA1 variant analysis with Evo2...")

    # Run inference
    result = run_brca1_analysis.remote()

    if "plot" in result:
        plot_data = base64.b64decode(result["plot"])
        with open("brca1_analysis_plot.png", "wb") as f:
            f.write(plot_data)

        img = mpimg.imread(BytesIO(plot_data))
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.show()