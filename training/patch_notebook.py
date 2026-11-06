"""Apply the training/integration fixes to the teammate's notebook (idempotent: re-applies from the original).

Usage:
    python training/patch_notebook.py <original.ipynb> training/SIDD_CC_RAW_Complete_Pipeline.ipynb

Every change is an exact text replacement; the script fails loudly if a target text is missing,
so it can never silently produce a half-patched notebook.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(i: int) -> str:
    return "".join(cells[i]["source"])


def set_src(i: int, text: str) -> None:
    cells[i]["source"] = text.splitlines(keepends=True)
    if cells[i]["cell_type"] == "code":
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None


def replace(i: int, old: str, new: str, count: int = 1) -> None:
    s = src(i)
    n = s.count(old)
    if n != count:
        raise SystemExit(f"cell {i}: expected {count} occurrence(s) of {old[:70]!r}, found {n}")
    set_src(i, s.replace(old, new))


def find_cell(marker: str) -> int:
    hits = [i for i, c in enumerate(cells) if marker in "".join(c["source"])]
    if len(hits) != 1:
        raise SystemExit(f"marker {marker!r} matched cells {hits}")
    return hits[0]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


C_CFG = find_cell("SECTION 0: MASTER CONFIGURATION")
C_ENV = find_cell("%pip install -q -U huggingface_hub")
C_UTIL = find_cell("def select_instances(")
C_PLAN = find_cell("def plan_download(")
C_LOSS = find_cell("class CharbonnierLoss")
C_SMOKE = find_cell("def run_smoke_test(")
C_TRAIN = find_cell("def run_full_training(")
C_TEST = find_cell("def evaluate_test_set(")
C_EXPORT = find_cell("def export_deployment_artifacts(")

# ----------------------------------------------------------------------------- Section 0
set_src(C_CFG, '''# ==============================================================================
# SECTION 0: MASTER CONFIGURATION & SAFETY SWITCHES (Edit Here Only)
# ==============================================================================
# Defaults are set for a FULL training run on a free cloud GPU (Kaggle T4 or Colab T4):
#   Kaggle: Settings -> Accelerator "GPU T4 x1", Internet ON, then "Save Version -> Save & Run All".
#   Colab : Runtime -> Change runtime type -> T4 GPU, then Runtime -> Run all (keep the tab open).
# Results land in exported_model/ (+ exported_model.zip): weights, ONNX FP32 + INT8, metrics JSON.
import os, sys
from pathlib import Path

QUICK = os.environ.get("ISP_NB_QUICK", "0") == "1"   # tiny CPU dry run of "Run all" (used before the cloud run)

# --- Remote Hugging Face Dataset ---
REPO_ID          = "MhdMohammadi/SIDD-CC"
TRAIN_PREFIX     = "Raw/train/Data"
SCENE_LIST_PATH  = "Raw/train/Scene_Instances.txt"
MAT_KEY          = "data"  # Documented key holding the Bayer mosaic array

# --- Sample Validation Pair (Previously Inspected & Verified) ---
SAMPLE_INSTANCE  = "0001_001_S6_00100_00060_3200_L"
SAMPLE_IDX       = None  # None = first complete pair in that folder

# --- Stage 2 & 5: Dataset Download Switches ---
RUN_FULL_DOWNLOAD     = True   # Download the multi-scene subset
RUN_DOWNLOAD          = RUN_FULL_DOWNLOAD  # Convenience alias
CONFIRM_FULL_DOWNLOAD = True   # Secondary safety gate
SUBSET_INSTANCES      = 30     # ~7 GB of .MAT; spans all 10 SIDD scenes and all 5 phones
MIN_SCENES_SUBSET     = 10     # Every physical scene represented -> 7 train / 1 val / 2 test scenes
DOWNLOAD_WORKERS      = 8      # Number of concurrent download worker threads
DOWNLOAD_TEST_BLOCKS  = False  # Optional: download official SIDD benchmark test blocks (~0.5 GB)
QUICK_PAIRS_PER_INSTANCE = None  # None = all pairs of each chosen instance (QUICK mode uses 1)

# --- Stage 6: RAW Conversion & Scene Splitting ---
RUN_CONVERT                 = True   # Convert downloaded MAT -> lossless float32 .npy
DELETE_MAT_AFTER_CONVERSION = True   # Free disk once each pair is safely converted
VAL_FRACTION                = 0.10   # Scene-level validation share (cut strictly by scene ID)
TEST_FRACTION               = 0.20   # Scene-level held-out test share (2 of 10 scenes)
SPLIT_SEED                  = 42     # Deterministic split seed

# --- Stages 10-14: Model & Training Execution Gates ---
RUN_SMOKE_TEST             = True   # Verify model overfits 4 training patches before full training
SMOKE_ITERS                = 120
RUN_TRAINING               = True   # Run the full training loop
RESUME_CHECKPOINT          = True   # Resume from latest_checkpoint.pth if it exists (no-op on a fresh run)
CHECKPOINT_PATH            = None   # Path to custom checkpoint or None for default
REQUIRE_GPU                = True   # Stop early with instructions if no GPU is attached

# Hyperparameters
PATCH_SIZE    = 128    # Bayer-packed patch size (H/2, W/2) -> corresponds to 256x256 in raw mosaic
BATCH_SIZE    = 16     # Mini-batch size (T4 has room; was 8)
NUM_EPOCHS    = 60     # Maximum training epochs (time cap below stops earlier if needed)
LEARNING_RATE = 2e-4   # AdamW learning rate
WEIGHT_DECAY  = 1e-4   # AdamW weight decay
GRAD_CLIP     = 1.0    # Max gradient norm for clipping
PATIENCE      = 10     # Early stopping patience on validation PSNR
NUM_WORKERS   = 2      # DataLoader workers
TRAIN_PATCHES_PER_PAIR = 300
VAL_PATCHES_PER_PAIR   = 50
MAX_TRAIN_HOURS        = 5.0  # Hard time cap: stop after the epoch that crosses it (best checkpoint is kept)

# --- Stages 16 & 17: Efficiency Experiments & Model Export ---
RUN_EFFICIENCY_EXPERIMENTS = True   # Benchmark FP16 vs FP32, latency, memory, and tiled inference
RUN_EXPORT                 = True   # Export standalone model, inference script, and deployment bundle
RUN_ONNX_EXPORT            = True   # ONNX FP32 + static INT8 for the web app (CPU deployment)
EXPORT_DIR_NAME            = "exported_model"

# --- Cloud Storage (Google Drive, Colab only) ---
COPY_ARTIFACTS_TO_DRIVE    = False  # True: checkpoints + export go to Drive, so a disconnect can resume
DRIVE_DIR                  = "/content/drive/MyDrive/sidd_cc_artifacts"

# Global Reproducibility Seed
SEED = 42

if QUICK:  # Local CPU dry run: 3 scenes x 1 pair, 1 epoch, small patches. Exercises every cell.
    SUBSET_INSTANCES, MIN_SCENES_SUBSET, QUICK_PAIRS_PER_INSTANCE = 3, 3, 1
    VAL_FRACTION, TEST_FRACTION = 0.34, 0.34
    PATCH_SIZE, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS = 64, 4, 1, 0
    TRAIN_PATCHES_PER_PAIR, VAL_PATCHES_PER_PAIR, MAX_TRAIN_HOURS = 8, 4, 0.25
    REQUIRE_GPU, RESUME_CHECKPOINT = False, False

print("Master configuration loaded successfully." + ("  [QUICK dry-run mode]" if QUICK else ""))
print(f"Safety Switches: RUN_FULL_DOWNLOAD={RUN_FULL_DOWNLOAD} | CONFIRM_FULL_DOWNLOAD={CONFIRM_FULL_DOWNLOAD} | RUN_SMOKE_TEST={RUN_SMOKE_TEST} | RUN_TRAINING={RUN_TRAINING}")
''')

# ----------------------------------------------------------------------------- environment
replace(C_ENV, "%pip install -q -U huggingface_hub scipy pandas matplotlib tqdm torch torchvision torchinfo\n",
        "# NOTE: torch is NOT reinstalled/upgraded here: Kaggle/Colab ship a CUDA build matched to their driver.\n"
        "%pip install -q huggingface_hub torchinfo onnx onnxruntime onnxscript\n")
replace(C_ENV, "import os, sys, re, json, math, random, shutil, hashlib, time, itertools, datetime\n",
        "import os, sys, re, json, math, random, shutil, hashlib, time, itertools, datetime, warnings\n")
replace(C_ENV, '''IN_COLAB = "google.colab" in sys.modules
WORK = Path("/content/sidd_cc") if IN_COLAB else Path("./sidd_cc").resolve()
DL_DIR, NPY_DIR, OUT_DIR = WORK / "hf_download", WORK / "npy", WORK / "artifacts"
EXPORT_DIR = WORK / EXPORT_DIR_NAME
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
''', '''IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.isdir("/kaggle/working")
if os.environ.get("ISP_NB_WORK"):
    WORK = Path(os.environ["ISP_NB_WORK"]).resolve()
elif IN_COLAB:
    WORK = Path("/content/sidd_cc")
elif IN_KAGGLE:
    WORK = Path("/tmp/sidd_cc")          # large scratch disk; /kaggle/working keeps only the results
else:
    WORK = Path("./sidd_cc").resolve()

# PERSIST = where checkpoints + the export package go (kept after the session ends).
PERSIST = Path("/kaggle/working") if IN_KAGGLE else WORK
if IN_COLAB and COPY_ARTIFACTS_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    PERSIST = Path(DRIVE_DIR)

DL_DIR, NPY_DIR, OUT_DIR = WORK / "hf_download", WORK / "npy", WORK / "artifacts"
EXPORT_DIR = PERSIST / EXPORT_DIR_NAME
CHECKPOINT_DIR = PERSIST / "checkpoints"
''')
replace(C_ENV, '''    HW_DESC = f"CPU ({os.cpu_count()} cores)"
''', '''    HW_DESC = f"CPU ({os.cpu_count()} cores)"

if DEVICE.type != "cuda" and REQUIRE_GPU and RUN_TRAINING:
    raise RuntimeError(
        "No GPU detected - training on CPU would take days.\\n"
        "  Kaggle: right panel -> Settings -> Accelerator -> 'GPU T4 x1' (and Internet ON).\\n"
        "  Colab : Runtime -> Change runtime type -> 'T4 GPU'.\\n"
        "Then run all cells again. (Set REQUIRE_GPU=False in Section 0 to force CPU.)")
''')
replace(C_ENV, 'print(f"Working Dir     : {WORK}")\n',
        'print(f"Working Dir     : {WORK}")\nprint(f"Results Dir     : {PERSIST}  (checkpoints + {EXPORT_DIR_NAME}/)")\n')

# ----------------------------------------------------------------------------- CCM fix (render)
OLD_CCM = '''def apply_color_matrix(rgb, color_matrix2):
    """Transforms sensor RGB into CIE XYZ / sRGB space using ColorMatrix2 metadata."""
    if color_matrix2 is None or len(color_matrix2) != 9:
        return rgb
    cm = np.asarray(color_matrix2, dtype=np.float32).reshape(3, 3)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums
    return np.clip(np.matmul(rgb, cm_norm.T), 0, None)'''
NEW_CCM = '''_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)

def cam_to_srgb_matrix(color_matrix2):
    """DNG ColorMatrix2 maps XYZ -> camera RGB. Camera -> linear sRGB is the inverse of the
    row-normalised (ColorMatrix2 @ sRGB->XYZ) matrix (dcraw/libraw convention)."""
    cm = np.asarray(color_matrix2, dtype=np.float64).reshape(3, 3)
    rgb2cam = cm @ _SRGB_TO_XYZ
    rgb2cam = rgb2cam / rgb2cam.sum(axis=1, keepdims=True)
    return np.linalg.inv(rgb2cam).astype(np.float32)

def apply_color_matrix(rgb, color_matrix2):
    """Transforms white-balanced camera RGB into linear sRGB using ColorMatrix2 metadata."""
    if color_matrix2 is None or len(color_matrix2) != 9:
        return rgb
    try:
        m = cam_to_srgb_matrix(color_matrix2)
    except np.linalg.LinAlgError:
        return rgb
    return np.clip(np.matmul(rgb, m.T), 0, None)'''
replace(C_UTIL, OLD_CCM, NEW_CCM)

# ----------------------------------------------------------------------------- QUICK download filter
replace(C_PLAN, '''    sel = complete_df[complete_df.instance.isin(chosen)]
    done_ids = converted_ok_ids()''', '''    sel = complete_df[complete_df.instance.isin(chosen)]
    if QUICK_PAIRS_PER_INSTANCE:
        sel = sel.sort_values("idx").groupby("instance", group_keys=False).head(QUICK_PAIRS_PER_INSTANCE)
    done_ids = converted_ok_ids()''')

# ----------------------------------------------------------------------------- missing compute_ssim
replace(C_LOSS, 'print("Loss functions and evaluation metrics defined.")', '''_SSIM_EVAL = {}

def compute_ssim(pred, target):
    """Mean SSIM over the 4 packed RAW channels (same Gaussian window as SSIMLoss). Returns a tensor."""
    key = (str(pred.device), pred.shape[1])
    if key not in _SSIM_EVAL:
        _SSIM_EVAL[key] = SSIMLoss(channels=pred.shape[1]).to(pred.device)
    return 1.0 - _SSIM_EVAL[key](pred.float(), target.float())

print("Loss functions and evaluation metrics defined.")''')

# ----------------------------------------------------------------------------- smoke test iterations
replace(C_SMOKE, "run_smoke_test(smoke_model, manifest, DEVICE, num_patches=4, iterations=120)",
        "run_smoke_test(smoke_model, manifest, DEVICE, num_patches=4, iterations=SMOKE_ITERS)")

# ----------------------------------------------------------------------------- training loop
replace(C_TRAIN, "def run_full_training():", '''def _seed_worker(worker_id):
    """Give every DataLoader worker its own crop/augmentation RNG (otherwise all workers repeat the same crops)."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = random.Random(torch.initial_seed() % (2 ** 32))

def run_full_training():''')
replace(C_TRAIN, "train_ds = SIDDPatchDataset(train_mf, patch_size=PATCH_SIZE, augment=True, patches_per_pair=200, seed=SEED)",
        "train_ds = SIDDPatchDataset(train_mf, patch_size=PATCH_SIZE, augment=True, patches_per_pair=TRAIN_PATCHES_PER_PAIR, seed=SEED)")
replace(C_TRAIN, "val_ds = SIDDPatchDataset(val_mf, patch_size=PATCH_SIZE, augment=False, patches_per_pair=50, seed=SEED + 1)",
        "val_ds = SIDDPatchDataset(val_mf, patch_size=PATCH_SIZE, augment=False, patches_per_pair=VAL_PATCHES_PER_PAIR, seed=SEED + 1)")
replace(C_TRAIN, 'train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))',
        'train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"), worker_init_fn=_seed_worker)')
replace(C_TRAIN, '''    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()''', '''    train_start = time.time()
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()''')
replace(C_TRAIN, '''        # Check early stopping
        if patience_counter >= PATIENCE:''', '''        # Hard wall-clock cap (best checkpoint is already saved)
        hours = (time.time() - train_start) / 3600
        if hours >= MAX_TRAIN_HOURS:
            print(f"\\nTime cap reached ({hours:.2f} h >= MAX_TRAIN_HOURS={MAX_TRAIN_HOURS}); stopping after epoch {epoch}.")
            break

        # Check early stopping
        if patience_counter >= PATIENCE:''')

# ----------------------------------------------------------------------------- test evaluation -> results
replace(C_TEST, '''    results = evaluate_test_set(eval_model, test_loader, criterion, DEVICE)
    res_df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    display(res_df)''', '''    results = evaluate_test_set(eval_model, test_loader, criterion, DEVICE)
    # Baseline: the noisy input itself (what the network has to improve on)
    noisy_psnr = []
    for noisy, gt in test_loader:
        for i in range(noisy.size(0)):
            noisy_psnr.append(compute_psnr(noisy[i], gt[i]))
    results["Noisy input RAW PSNR (dB)"] = float(np.mean(noisy_psnr))
    results["Test scenes"] = sorted(test_mf["scene"].unique().tolist())
    results["Test pairs"] = int(len(test_mf))
    res_df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    display(res_df)''')
replace(C_TEST, '''    plt.savefig(OUT_DIR / "test_visual_comparison.png", dpi=150)
    plt.show()
''', '''    plt.savefig(OUT_DIR / "test_visual_comparison.png", dpi=150)
    plt.show()
    return results
''')
replace(C_TEST, '''if RUN_TRAINING:
    run_evaluation_and_visualizations()''', '''TEST_RESULTS = {}
if RUN_TRAINING:
    TEST_RESULTS = run_evaluation_and_visualizations() or {}''')

# ----------------------------------------------------------------------------- tiled inference bug
C_TILE = find_cell("class TiledRAWInference")
replace(C_TILE, '''            x_pad = F.pad(packed_raw_tensor, (0, pad_right, 0, pad_bottom), mode="reflect")''',
        '''            # reflect padding must be smaller than the input size; fall back to replicate otherwise
            mode = "reflect" if (pad_bottom < H and pad_right < W) else "replicate"
            x_pad = F.pad(packed_raw_tensor, (0, pad_right, 0, pad_bottom), mode=mode)''')
replace(C_TILE, '''if RUN_EFFICIENCY_EXPERIMENTS:
    run_efficiency_benchmarks()''', '''if RUN_EFFICIENCY_EXPERIMENTS:
    try:  # optional diagnostics must never block the export below
        run_efficiency_benchmarks()
    except Exception as e:
        print("Efficiency benchmark failed (continuing to export):", repr(e)[:300])''')

# ----------------------------------------------------------------------------- export fixes
replace(C_EXPORT_PLACEHOLDER := find_cell("def export_deployment_artifacts("),
        '''        x_pad = F.pad(packed_tensor.to(self.device), (0, pad_right, 0, pad_bottom), mode="reflect")''',
        '''        mode = "reflect" if (pad_bottom < H and pad_right < W) else "replicate"
        x_pad = F.pad(packed_tensor.to(self.device), (0, pad_right, 0, pad_bottom), mode=mode)''')
# The standalone unet_isp.py written by the export was a DIFFERENT (3-level, additive-skip) network than the
# one trained, so the exported package could not load its own weights. Write the trained class verbatim.
_s = src(C_EXPORT)
_a = _s.index('    unet_code = """')
_b = _s.index('    (export_path / "unet_isp.py").write_text(unet_code.strip())')
_true_cls = (Path(__file__).resolve().parent / "unet_isp.py").read_text(encoding="utf-8")
set_src(C_EXPORT, _s[:_a] + "    # Exact class used for training (was a different architecture before)\n"
        + "    unet_code = " + repr(_true_cls) + "\n" + _s[_b:])

replace(C_EXPORT, 'ckpt = torch.load(ckpt_target, map_location="cpu")',
        'ckpt = torch.load(ckpt_target, map_location="cpu", weights_only=False)  # checkpoint holds RNG/history too')
OLD_CCM_PKG = '''def apply_color_matrix(rgb, color_matrix2):
    if color_matrix2 is None or len(color_matrix2) != 9:
        return rgb
    cm = np.asarray(color_matrix2, dtype=np.float32).reshape(3, 3)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums
    return np.clip(np.matmul(rgb, cm_norm.T), 0, None)'''
NEW_CCM_PKG = '''_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)

def apply_color_matrix(rgb, color_matrix2):
    if color_matrix2 is None or len(color_matrix2) != 9:
        return rgb
    cm = np.asarray(color_matrix2, dtype=np.float64).reshape(3, 3)
    rgb2cam = cm @ _SRGB_TO_XYZ
    rgb2cam = rgb2cam / rgb2cam.sum(axis=1, keepdims=True)
    m = np.linalg.inv(rgb2cam).astype(np.float32)
    return np.clip(np.matmul(rgb, m.T), 0, None)'''
replace(C_EXPORT, OLD_CCM_PKG, NEW_CCM_PKG)
replace(C_EXPORT, '''    print(f"\\nSUCCESS: Complete deployment package exported to: {export_path.resolve()}")''',
        '''    # Keep the split/manifest/curves with the weights (the web app's eval scripts read split.json)
    for extra in ("split.json", "manifest.csv", "training_curves.png", "test_visual_comparison.png"):
        if (OUT_DIR / extra).exists():
            shutil.copy2(OUT_DIR / extra, export_path / extra)

    print(f"\\nSUCCESS: Complete deployment package exported to: {export_path.resolve()}")''')
replace(C_EXPORT, '''if RUN_EXPORT:
    export_deployment_artifacts({"patch_size": PATCH_SIZE}, {}, EXPORT_DIR_NAME, force_untrained=False)''',
        '''if RUN_EXPORT:
    _eval = dict(globals().get("TEST_RESULTS") or {})
    _hist = (globals().get("train_history") or {})
    if _hist.get("val_psnr"):
        _eval.update(best_val_psnr=float(max(_hist["val_psnr"])), epochs_trained=len(_hist["val_psnr"]))
    export_deployment_artifacts({"patch_size": PATCH_SIZE}, _eval, EXPORT_DIR, force_untrained=False)''')

# ----------------------------------------------------------------------------- new: ONNX + INT8 + efficiency
ONNX_MD = '''## Section 17b. ONNX Export, INT8 Quantisation & CPU Efficiency (web-app integration)
The web app runs on a CPU-only server without PyTorch, so the trained network is exported to **ONNX**:
1. `_NoPadExport` wraps the trained network with the *same weights and layers* but without the internal reflect-pad branch (the app pads inputs to a multiple of 8 itself), so the ONNX graph accepts any H, W divisible by 8.
2. **Parity check**: ONNX Runtime output must match PyTorch to < 1e-4 at two sizes.
3. **Static INT8 post-training quantisation** (ONNX Runtime, QDQ, per-channel weights) calibrated on validation patches. The final residual add stays FP32 so quantisation noise is not added on top of the image.
4. **Efficiency table** (the brief's metrics 4 and 5): parameters, file size, CPU latency at 1 MP RAW, peak memory, and quality (RAW PSNR / SSIM on held-out test patches) for FP32 vs INT8 vs the noisy input.
'''
ONNX_CODE = r'''# ==============================================================================
# SECTION 17b: ONNX EXPORT + INT8 QUANTISATION + CPU EFFICIENCY (feeds the web app)
# ==============================================================================
class _NoPadExport(nn.Module):
    """LightweightISPUNet.forward without the internal pad/crop (inputs are already a multiple of 8)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = self.net
        e1 = n.enc1(n.head(x)); d1 = n.down1(e1)
        e2 = n.enc2(d1); d2 = n.down2(e2)
        e3 = n.enc3(d2); d3 = n.down3(e3)
        b = n.bottleneck(d3)
        dec3 = n.dec3(torch.cat([n.up3(b), e3], dim=1))
        dec2 = n.dec2(torch.cat([n.up2(dec3), e2], dim=1))
        dec1 = n.dec1(torch.cat([n.up1(dec2), e1], dim=1))
        return x + n.tail(dec1)


def _onnx_export(module, path, size=256):
    dummy = torch.randn(1, 4, size, size)
    names = dict(input_names=["packed_raw"], output_names=["denoised_raw"])
    axes = {"packed_raw": {2: "h", 3: "w"}, "denoised_raw": {2: "h", 3: "w"}}
    try:
        torch.onnx.export(module, (dummy,), str(path), opset_version=17, dynamic_axes=axes, dynamo=False, **names)
        return "torchscript"
    except TypeError:  # older torch: no 'dynamo' argument
        torch.onnx.export(module, (dummy,), str(path), opset_version=17, dynamic_axes=axes, **names)
        return "torchscript"
    except Exception as e:  # newer torch without the TorchScript exporter
        print("TorchScript ONNX exporter unavailable (", type(e).__name__, ") -> using torch.export (dynamo)")
        from torch.export import Dim
        h8, w8 = Dim("h8", min=1, max=2048), Dim("w8", min=1, max=2048)
        prog = torch.onnx.export(module, (dummy,), dynamo=True, dynamic_shapes={"x": {2: 8 * h8, 3: 8 * w8}}, **names)
        prog.save(str(path))
        return "dynamo"


def _median_ms(fn, runs=10, warmup=2):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return float("nan")


def run_onnx_export_and_quantisation():
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    ckpt_path = CHECKPOINT_DIR / "best_isp_unet.pth"
    if not ckpt_path.exists():
        print("No trained checkpoint yet; skipping ONNX export.")
        return None
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    net = LightweightISPUNet(in_channels=4, out_channels=4, base_channels=32)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    net.eval()
    wrapped = _NoPadExport(net).eval()

    # Wrapper must be numerically identical to the original forward on multiple-of-8 inputs
    with torch.no_grad():
        xt = torch.rand(1, 4, 64, 96)
        assert torch.allclose(net(xt), wrapped(xt), atol=1e-6), "export wrapper differs from model.forward"

    fp32_path, int8_path = EXPORT_DIR / "isp_unet.onnx", EXPORT_DIR / "isp_unet_int8.onnx"
    exporter = _onnx_export(wrapped, fp32_path)
    onnx.checker.check_model(str(fp32_path))
    so = ort.SessionOptions(); so.intra_op_num_threads = max(1, os.cpu_count() or 1)
    s32 = ort.InferenceSession(str(fp32_path), so, providers=["CPUExecutionProvider"])

    parity = {}
    for hw in [(256, 256), (192, 320)]:
        x = torch.rand(1, 4, *hw) * 0.5
        with torch.no_grad():
            ref = wrapped(x).numpy()
        got = s32.run(None, {"packed_raw": x.numpy()})[0]
        parity[f"{hw[0]}x{hw[1]}"] = float(np.abs(ref - got).max())
    print("ONNX parity (max abs diff vs PyTorch):", parity, "| exporter:", exporter)
    assert max(parity.values()) < 1e-4, f"ONNX parity failed: {parity}"

    # Calibration on validation patches (fall back to train if val is empty)
    cal_mf = manifest[manifest.split == "val"]
    if cal_mf.empty:
        cal_mf = manifest[manifest.split == "train"]
    cal_ds = SIDDPatchDataset(cal_mf, patch_size=PATCH_SIZE, augment=False, patches_per_pair=8, seed=SEED + 7)
    cal_arrays = [cal_ds[i][0].unsqueeze(0).numpy() for i in range(min(32, len(cal_ds)))]

    class _Calib(CalibrationDataReader):
        def __init__(self, arrays):
            self._it = iter([{"packed_raw": a} for a in arrays])

        def get_next(self):
            return next(self._it, None)

    g = onnx.load(str(fp32_path)).graph
    keep_fp32 = [n.name for n in g.node if n.op_type == "Add" and "denoised_raw" in n.output]
    tail_conv = [n.name for n in g.node if n.op_type == "Conv"][-1:]
    quantize_static(str(fp32_path), str(int8_path), _Calib(cal_arrays), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv"], nodes_to_exclude=keep_fp32 + tail_conv,
                    extra_options={"DedicatedQDQPair": True})
    s8 = ort.InferenceSession(str(int8_path), so, providers=["CPUExecutionProvider"])

    # Quality on held-out test patches (RAW domain): noisy input vs FP32 vs INT8
    q_mf = manifest[manifest.split == "test"]
    if q_mf.empty:
        q_mf = cal_mf
    q_ds = SIDDPatchDataset(q_mf, patch_size=PATCH_SIZE, augment=False, patches_per_pair=10, seed=SEED + 9)
    qual = {"noisy": [], "fp32": [], "int8": []}
    ssim = {"noisy": [], "fp32": [], "int8": []}
    for i in range(min(40, len(q_ds))):
        n_t, g_t = q_ds[i]
        x = n_t.unsqueeze(0).numpy(); gt = g_t.unsqueeze(0)
        outs = {"noisy": n_t.unsqueeze(0),
                "fp32": torch.from_numpy(s32.run(None, {"packed_raw": x})[0]),
                "int8": torch.from_numpy(s8.run(None, {"packed_raw": x})[0])}
        for k, o in outs.items():
            qual[k].append(compute_psnr(o, gt)); ssim[k].append(float(compute_ssim(o, gt)))

    # CPU latency + memory at 1 MP RAW (= 512 x 512 packed)
    x1 = (np.random.rand(1, 4, 512, 512).astype(np.float32) * 0.3)
    xt1 = torch.from_numpy(x1)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rows = []
    params = int(sum(p.numel() for p in net.parameters()))
    for name, fn, size_mb in [
        ("PyTorch FP32 (CPU)", lambda: wrapped(xt1), params * 4 / 1e6),
        ("ONNX Runtime FP32", lambda: s32.run(None, {"packed_raw": x1}), fp32_path.stat().st_size / 1e6),
        ("ONNX Runtime INT8", lambda: s8.run(None, {"packed_raw": x1}), int8_path.stat().st_size / 1e6),
    ]:
        before = _rss_mb()
        with torch.no_grad():
            ms = _median_ms(fn)
        rows.append({"variant": name, "params": params, "size_mb": round(size_mb, 3),
                     "latency_ms_1mp_cpu": round(ms, 1), "rss_delta_mb": round(_rss_mb() - before, 1)})
    key = {"PyTorch FP32 (CPU)": "fp32", "ONNX Runtime FP32": "fp32", "ONNX Runtime INT8": "int8"}
    for r in rows:
        k = key[r["variant"]]
        r["raw_psnr_db"] = round(float(np.mean(qual[k])), 3)
        r["raw_ssim"] = round(float(np.mean(ssim[k])), 4)

    eff = {
        "device_for_latency": f"CPU x{os.cpu_count()} ({'Kaggle' if IN_KAGGLE else 'Colab' if IN_COLAB else 'local'})",
        "onnx_exporter": exporter, "onnx_parity_max_abs_diff": parity,
        "noisy_input": {"raw_psnr_db": round(float(np.mean(qual["noisy"])), 3), "raw_ssim": round(float(np.mean(ssim["noisy"])), 4)},
        "variants": rows,
        "int8_vs_fp32_psnr_drop_db": round(float(np.mean(qual["fp32"]) - np.mean(qual["int8"])), 3),
        "quality_patches": len(qual["fp32"]), "quality_split": "test" if not manifest[manifest.split == "test"].empty else "val",
    }
    (EXPORT_DIR / "efficiency.json").write_text(json.dumps(eff, indent=2))
    (EXPORT_DIR / "model.json").write_text(json.dumps({
        "name": "LightweightISPUNet (SIDD-CC RAW denoiser)",
        "file": "isp_unet_int8.onnx" if eff["int8_vs_fp32_psnr_drop_db"] < 0.3 else "isp_unet.onnx",
        "input": "raw4", "input_wb": False, "output": "raw4", "upscale": 1,
        "pad_multiple": 8, "layout": "NCHW", "tile": 1024, "overlap": 32}, indent=2))
    display(pd.DataFrame(rows))
    print(f"Noisy input RAW PSNR: {eff['noisy_input']['raw_psnr_db']} dB | INT8 drop vs FP32: {eff['int8_vs_fp32_psnr_drop_db']} dB")
    print(f"Saved: {fp32_path.name}, {int8_path.name}, efficiency.json, model.json -> {EXPORT_DIR}")
    return eff

EFFICIENCY = run_onnx_export_and_quantisation() if RUN_ONNX_EXPORT else None
'''
ZIP_CODE = r'''# ==============================================================================
# FINAL: package everything for the web app
# ==============================================================================
if EXPORT_DIR.exists() and any(EXPORT_DIR.iterdir()):
    zip_path = shutil.make_archive(str(PERSIST / EXPORT_DIR_NAME), "zip", EXPORT_DIR)
    print("Download this file and give it to the app owner:", zip_path)
    print("Contents:", sorted(p.name for p in EXPORT_DIR.iterdir()))
    if IN_KAGGLE:
        print("Kaggle: it is in the notebook's Output tab (Data -> Output) once the version finishes.")
    if IN_COLAB:
        try:
            from google.colab import files
            files.download(zip_path)
        except Exception as e:
            print("Use the Files panel to download it:", e)
else:
    print("Nothing exported yet (training did not produce a checkpoint).")
'''
CHANGES_MD = '''> **Changes in this version (for the web-app integration)**: cloud-GPU defaults (Kaggle/Colab, all 10 scenes, 2 held-out test scenes, 5 h time cap, resume), no torch reinstall (keeps the platform's CUDA build), fixed `compute_ssim` (was undefined, crashed test evaluation), fixed export `torch.load(weights_only=False)` (crashed on torch >= 2.6), exported `unet_isp.py` is now the exact trained architecture (it was a different 3-level network that could not load the weights), tiled-inference reflect-pad crash on small inputs fixed (and benchmarks can no longer block the export), per-worker DataLoader RNG (workers were repeating crops), correct `ColorMatrix2` inversion in the sRGB render, test results passed into `training_results.json`, new Section 17b (ONNX FP32 + INT8 + efficiency table + `model.json` for the app), and a final zip of `exported_model/`. Set `ISP_NB_QUICK=1` for a tiny CPU dry run.
'''
# Insert after the export cell, then the zip cell; changes note after the title cell.
cells.insert(C_EXPORT + 1, md(ONNX_MD))
cells.insert(C_EXPORT + 2, code(ONNX_CODE))
cells.insert(C_EXPORT + 3, md("## Final. Package the results\n"))
cells.insert(C_EXPORT + 4, code(ZIP_CODE))
cells.insert(1, md(CHANGES_MD))

import uuid

for c in cells:  # clear stale outputs from the teammate's last (CPU) run; nbformat 4.5 needs cell ids
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
    c.setdefault("id", uuid.uuid4().hex[:8])
nb["nbformat"], nb["nbformat_minor"] = 4, max(5, nb.get("nbformat_minor", 5))

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"patched notebook written: {DST} ({len(cells)} cells)")
