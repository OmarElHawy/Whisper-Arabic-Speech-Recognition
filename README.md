# 🎙️ Arabic Speech Recognition — Fine-tuned Whisper

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-Demo-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)

**End-to-end Arabic Automatic Speech Recognition (ASR) system built by fine-tuning OpenAI Whisper-small on Mozilla Common Voice Arabic.**

*Neural Networks Course Project — E-JUST*

</div>

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Word Error Rate (WER)** | **24.24%** (validation) |
| **Character Error Rate (CER)** | **11.14%** (test) |
| Correct Words | 68.7% |
| Training Time | ~3.8 hours |
| Training Samples | 20,000 |
| Test Samples | 1,000 |

---

## 🧠 Architecture

```
Audio (.mp3 / .wav)
        │
        ▼
Load & Resample (16kHz mono)
        │
        ▼
Log-Mel Spectrogram  ──  80 frequency bins × 3000 time frames
        │
        ▼
┌──────────────────────────────────┐
│         CNN Encoder              │  ← FROZEN during fine-tuning
│  Conv1D(80 → 512, kernel=3, GELU)│
│  Conv1D(512 → 512, kernel=3, GELU│
│  stride=2 → time axis compressed │
└──────────────┬───────────────────┘
               │
        Positional Embedding
               │
        6× Transformer Encoder Blocks  ←  FROZEN
               │
        6× Transformer Decoder Blocks  ←  FINE-TUNED
               │
        Arabic Token Sequence (autoregressive)
               │
               ▼
         Arabic Text Output
```

**Why Conv1D not Conv2D?**
Audio spectrograms have time flowing in one direction only. Conv1D slides along the time axis, treating all 80 frequency bins as channels at each time step — correctly learning local acoustic patterns (phoneme edges, consonant bursts) without incorrectly mixing unrelated frequency bins.

**Why freeze the encoder?**
With only 20,000 training samples, fine-tuning the full 244M parameter model would cause heavy overfitting. Freezing the encoder (CNN + Transformer blocks) preserves the acoustic representations learned during Whisper's large-scale pretraining, while the decoder adapts to Arabic-specific patterns.

---

## 📁 Project Structure

```
arabic_asr/
├── Arabic_ASR_Whisper.ipynb    # Full training pipeline (16 cells)
├── demo.py                     # Standalone Gradio web demo
├── requirements.txt            # Dependencies
├── models/
│   └── whisper-arabic/
│       ├── best/               # Best checkpoint (lowest val WER)
│       └── final/              # Final epoch checkpoint
├── data/                       # Generated during training
└── results/
    ├── architecture.png        # Model architecture diagram
    ├── audio_example.png       # Waveform + spectrogram visualization
    ├── dataset_stats.png       # Dataset statistics
    ├── training_curves.png     # Loss + WER per epoch
    ├── evaluation_results.png  # WER/CER bar + error breakdown pie
    ├── error_analysis.png      # WER vs sentence length scatter
    ├── prediction_examples.png # Best/worst predictions table
    ├── summary_table.png       # Final metrics summary
    ├── metrics.json            # Evaluation metrics
    └── test_results.csv        # All predictions vs references
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/arabic-asr.git
cd arabic-asr
```

### 2. Install PyTorch with CUDA
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download **Mozilla Common Voice Arabic v24** from:
https://commonvoice.mozilla.org/en/datasets

Extract to a folder and note the path (you'll need it in the notebook).

---

## 🚀 Usage

### Train from scratch

Open `Arabic_ASR_Whisper.ipynb` and edit **Cell 2**:

```python
DATA_DIR = r'path/to/cv-corpus-24.0/ar'   # ← your dataset path
MAX_TRAIN_SAMPLES = 20000                  # adjust as needed
NUM_EPOCHS = 10
```

Then run all cells top to bottom. Training takes **~3–4 hours** on an RTX 3060 6GB.

### Run the demo (after training)

```bash
python demo.py
```

Opens automatically at `http://127.0.0.1:7860`

The demo supports:
- 🎤 **Microphone recording** — record Arabic speech directly
- 📁 **File upload** — WAV (always), MP3/FLAC/OGG (requires ffmpeg)

---

## 📓 Notebook Structure

| Cell | Description |
|------|-------------|
| 0 | Install dependencies |
| 1 | Imports & GPU check |
| 2 | ⚙️ Configuration — edit `DATA_DIR` here |
| 3 | Data loading & Arabic text normalization |
| 4 | Dataset statistics visualization |
| 5 | Audio waveform + log-mel spectrogram visualization |
| 6 | Architecture diagram (CNN highlighted) |
| 7 | Load Whisper processor & model |
| 8 | PyTorch Dataset & DataLoader |
| 9 | Optimizer, scheduler, fp16 scaler |
| 10 | 🚀 Custom training loop with live tqdm progress |
| 11 | Training curves (loss + WER per epoch) |
| 12 | Test set evaluation (WER, CER, error counts) |
| 13 | Results visualization (3 charts) |
| 14 | Error analysis (scatter + bucket distribution) |
| 15 | Best & worst predictions |
| 16 | Gradio demo |

---

## 🔧 Training Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| Base model | `openai/whisper-small` | Best WER/speed tradeoff for 6GB VRAM |
| Batch size | 4 | Safe for 6GB GPU |
| Gradient accumulation | 2 | Effective batch = 8 |
| Learning rate | 1e-5 | Conservative for fine-tuning |
| fp16 | ✅ | Halves VRAM usage |
| Freeze encoder | ✅ | Prevents overfitting on 20k samples |
| Early stopping | patience=3 epochs | Saves best checkpoint automatically |

---

## 📈 Training History

```
Epoch 1/10 — Train: 0.4912 | Val: 0.2847 | WER: 29.82%
Epoch 2/10 — Train: 0.2366 | Val: 0.2591 | WER: 26.86%
Epoch 3/10 — Train: 0.1371 | Val: 0.2549 | WER: 26.32%
Epoch 4/10 — Train: 0.0745 | Val: 0.2614 | WER: 25.05%
Epoch 5/10 — Train: 0.0397 | Val: 0.2759 | WER: 24.98%
Epoch 6/10 — Train: 0.0227 | Val: 0.2899 | WER: 25.04%
Epoch 7/10 — Train: 0.0138 | Val: 0.3046 | WER: 26.32%
Epoch 8/10 — Train: 0.0088 | Val: 0.3162 | WER: 24.84%
Epoch 9/10 — Train: 0.0060 | Val: 0.3207 | WER: 24.24% ← Best
Epoch 10/10 — Train: 0.0049 | Val: 0.3225 | WER: 25.14%
```

---

## 📦 Dataset

**Mozilla Common Voice Arabic v24**
- Source: https://commonvoice.mozilla.org/en/datasets
- Language: Arabic (MSA + dialects)
- Training split used: 20,000 samples
- Validation split: 1,000 samples
- Test split: 1,000 samples
- Audio format: MP3, 16kHz mono after preprocessing
- Mean sentence length: 29 characters / 5.8 words

**Arabic Text Normalization applied:**
- Remove diacritics (tashkeel)
- Normalize alef variants (أ إ آ → ا)
- Normalize teh marbuta (ة → ه)
- Normalize alef maqsura (ى → ي)
- Remove tatweel (ـ)
- Keep Arabic characters only

---

## 🖥️ Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 3060 Laptop 6GB |
| RAM | 16GB |
| CPU | AMD Ryzen 7 5800H |
| OS | Windows 11 |
| CUDA | 11.8 |

---

## 📚 References

- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356) — Radford et al., 2022
- [Mozilla Common Voice](https://commonvoice.mozilla.org) — Open source voice dataset
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) — Alternative ASR architecture considered
- [Word Error Rate (WER)](https://en.wikipedia.org/wiki/Word_error_rate) — Evaluation metric

---

## 📄 License

This project is for academic purposes — E-JUST Neural Networks Course.

---

<div align="center">
Built with ❤️ using PyTorch · HuggingFace Transformers · Gradio · Mozilla Common Voice
</div>
