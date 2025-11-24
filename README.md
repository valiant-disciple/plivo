# PII NER for Noisy STT Transcripts

**🎯 Trained Model Weights:** [Google Drive link](https://drive.google.com/drive/folders/1wd1ttWBLKTVvnPi84LJUqOF-IXlUaKXB?usp=drive_link)
Place this in out/ dir and follow first_models.ipynb for usage.

High-precision NER system for detecting PII in speech-to-text transcripts. Achieves **90.5% PII precision** with **10.3ms p95 latency** on CPU.

---

## 📊 Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **PII Precision** | 0.905 | ≥0.80 | ✅ +13% |
| **Overall F1** | 0.880 | - | ✅ |
| **p95 Latency (CPU)** | 10.3 ms | ≤20 ms | ✅ 2x faster |

**Per-Entity F1:** DATE (1.00), PHONE (0.946), CREDIT_CARD (0.944), LOCATION (0.857), CITY (0.845), EMAIL (0.762), PERSON_NAME (0.782)

**Best Model:** `microsoft/MiniLM-L12-H384-uncased`

---

## 🗂️ Key Files

- **`data/`** - 900 train + 180 dev examples generated via Groq (LLaMA 3.1)
- **`src/train.py`** - Training script with configurable hyperparameters
- **`src/predict.py`** - Inference with email validation post-processing
- **`src/eval_span_f1.py`** - Span-level metrics (per-entity + PII-only)
- **`src/measure_latency.py`** - CPU latency benchmarking (p50/p95)
- **`optunas.py`** - Hyperparameter search using Optuna
- **`first_models.ipynb`** - Full experimentation + ablation studies
- **`out/dev_pred_minilm_l6h384.json`** - Final predictions on dev set

---

## 🚀 Approach

### 1. Synthetic Data Generation (Groq API)
- Generated 1,080 examples using LLaMA 3.1-8B-instant via Groq
- STT-style noise: lowercase, no punctuation, "at"/"dot" notation, spelled-out numbers
- Inline XML tagging for exact character offset computation
- Balanced distribution across 7 entity types + 10% hard negatives
- Temperature 0.8, top-p 0.9 for high variance

### 2. Model Selection (Ablation Study)
Evaluated 3 lightweight transformers:

| Model | Parameters | PII Precision | p95 Latency | Decision |
|-------|------------|---------------|-------------|----------|
| **MiniLM-L12-H384** | 33M | **0.905** | 10.3 ms | ✅ Chosen |
| ELECTRA-small | 14M | 0.861 | 9.98 ms | ❌ -4.4% precision |
| MobileBERT | 25M | 0.832 | 12.1 ms | ❌ Slowest + lowest F1 |

**Decision:** Chose MiniLM despite ELECTRA being 0.3ms faster—**precision on PII is safety-critical** and worth the negligible latency cost.

### 3. Hyperparameter Tuning (Optuna)
Systematic search over 50 trials:
- **Learning rate:** 2.8e-5 (vs default 5e-5)
- **Epochs:** 5
- **Batch size:** 16
- **Dropout:** 0.15
- **Weight decay:** 0.01

**Impact:** +1.2% PII precision from baseline

### 4. Targeted Optimization
- **Data augmentation:** Generated 150 additional examples for weak classes (EMAIL, PERSON_NAME)
- **Email validation trick:** Post-processing rule requiring " at " + " dot " in EMAIL spans
  - **Result:** EMAIL precision **0.68 → 0.92** (+24% boost!)
  - Final PII precision: **0.861 → 0.905**

---

## 🛠️ Usage

# Install dependencies
pip install -r requirements.txt

# Train from scratch
python src/train.py \
  --model_name microsoft/MiniLM-L12-H384-uncased \
  --train data/train.jsonl --dev data/dev.jsonl \
  --out_dir out --epochs 5 --batch_size 16 --lr 2.8e-5

# Or download trained weights from Google Drive and predict
python src/predict.py \
  --model_dir path/to/downloaded/model \
  --input data/test.jsonl \
  --output predictions.json

# Evaluate
python src/eval_span_f1.py --gold data/dev.jsonl --pred predictions.json

# Measure latency
python src/measure_latency.py --model_dir path/to/model --runs 50 --device cpu---

## 🎯 Key Takeaways

1. **LLM-generated data is production-ready** - Groq API + careful prompting produced diverse, high-quality training data
2. **Precision > Speed for PII** - Sacrificed 0.3ms for +4.4% precision (safety-critical use case)
3. **Simple rules complement ML** - One validation function gave 24% precision boost on EMAIL
4. **Target weak spots** - Augmented specific classes and post-processed outputs → significant gains
