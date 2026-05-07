"""
Arabic ASR — Standalone Gradio Demo
Run: python demo.py
"""

import os
import sys
import time
import tempfile
import numpy as np
import torch
import librosa
import soundfile as sf
import gradio as gr
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR     = 'models/whisper-arabic/best'
TARGET_SR     = 16000
MAX_AUDIO_SEC = 30.0

os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Loading model from {MODEL_DIR}...')
if not Path(MODEL_DIR).exists():
    print(f'\n❌ Model not found at "{MODEL_DIR}"')
    sys.exit(1)

t0        = time.time()
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model     = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

GENERATE_KWARGS = {
    "language": "arabic",
    "task": "transcribe",
    "max_new_tokens": 128,
}

print(f'✅ Model ready in {time.time()-t0:.1f}s on {device}')
if torch.cuda.is_available():
    print(f'   GPU  : {torch.cuda.get_device_name(0)}')


# ── Audio helpers ─────────────────────────────────────────────────────────────
def load_from_file(path):
    """Load any audio file with librosa (WAV/MP3/FLAC/OGG)."""
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)[:int(MAX_AUDIO_SEC * TARGET_SR)]


def run_whisper(audio_data):
    """Run Whisper on float32 numpy array at 16kHz."""
    inputs = processor.feature_extractor(
        audio_data, sampling_rate=TARGET_SR, return_tensors='pt'
    ).input_features.to(device)

    with torch.no_grad():
        pred_ids = model.generate(inputs, **GENERATE_KWARGS)

    return processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]


def format_info(text, audio_data, elapsed):
    dur  = len(audio_data) / TARGET_SR
    wc   = len(text.split()) if text.strip() else 0
    return f'⏱ {elapsed:.2f}s  |  🎵 {dur:.1f}s  |  📝 {wc} words'


# ── Transcription: microphone ─────────────────────────────────────────────────
def transcribe_mic(audio):
    """
    Gradio mic with type='numpy' → (sample_rate, numpy_array).
    We save it as a temp WAV then load with librosa — most reliable cross-platform.
    """
    if audio is None:
        return '', '⚠️ No audio recorded. Press 🔴 Record, speak, then ⏹ Stop.'

    t0 = time.time()
    try:
        sr, arr = audio

        # Normalise int16 → float32
        arr = arr.astype(np.float32)
        if np.abs(arr).max() > 1.0:
            arr = arr / 32768.0
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        # Save to temp WAV then reload with librosa (handles any sr mismatch cleanly)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, arr, sr)

        audio_data = load_from_file(tmp_path)
        os.unlink(tmp_path)   # cleanup

        text = run_whisper(audio_data)
        return text, format_info(text, audio_data, time.time()-t0)

    except Exception as e:
        return f'Error: {e}', ''


# ── Transcription: file upload ────────────────────────────────────────────────
def transcribe_file(filepath):
    """File upload → filepath string → librosa load."""
    if filepath is None:
        return '', ''
    t0 = time.time()
    try:
        audio_data = load_from_file(filepath)
        text       = run_whisper(audio_data)
        return text, format_info(text, audio_data, time.time()-t0)
    except Exception as e:
        return f'Error: {e}', ''


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
    --bg:#0f1117; --surface:#1a1d27; --surf2:#222637; --border:#2e3250;
    --accent:#4f8ef7; --accent2:#7c5cfc; --green:#22d3a5;
    --text:#e8eaf6; --muted:#8892b0; --r:12px;
}
body,.gradio-container{background:var(--bg) !important;font-family:'IBM Plex Mono',monospace !important;}
.hdr{background:linear-gradient(135deg,var(--surface),#1a1f3a);border:1px solid var(--border);
     border-radius:var(--r);padding:32px 40px;margin-bottom:24px;position:relative;overflow:hidden;}
.hdr::before{content:'';position:absolute;top:-60px;right:-60px;width:200px;height:200px;
     background:radial-gradient(circle,rgba(79,142,247,.15) 0%,transparent 70%);border-radius:50%;}
.htitle{font-family:'Tajawal',sans-serif !important;font-size:2.2em !important;font-weight:700 !important;
     background:linear-gradient(90deg,var(--accent),var(--accent2));
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 8px 0 !important;}
.hsub{color:var(--muted) !important;font-size:.85em !important;margin:0 !important;}
.badge{display:inline-block;background:rgba(79,142,247,.12);border:1px solid rgba(79,142,247,.3);
     color:var(--accent);padding:4px 12px;border-radius:20px;font-size:.75em;margin-top:12px;margin-right:8px;}
.panel{background:var(--surface) !important;border:1px solid var(--border) !important;
     border-radius:var(--r) !important;padding:20px !important;}
.txbox textarea{font-family:'Tajawal',sans-serif !important;font-size:1.6em !important;
     direction:rtl !important;text-align:right !important;background:var(--surf2) !important;
     border:1px solid var(--border) !important;color:var(--text) !important;
     border-radius:10px !important;padding:20px !important;line-height:1.9 !important;min-height:120px !important;}
.txbox label{color:var(--green) !important;font-size:.8em !important;letter-spacing:.1em !important;text-transform:uppercase !important;}
.infobox textarea{font-family:'IBM Plex Mono',monospace !important;font-size:.82em !important;
     background:var(--surf2) !important;border:1px solid var(--border) !important;
     color:var(--muted) !important;border-radius:8px !important;padding:10px 14px !important;}
.infobox label{color:var(--muted) !important;font-size:.75em !important;letter-spacing:.1em !important;text-transform:uppercase !important;}
.pbtn button{background:linear-gradient(135deg,var(--accent),var(--accent2)) !important;
     border:none !important;border-radius:10px !important;color:white !important;
     font-family:'IBM Plex Mono',monospace !important;font-size:1em !important;
     font-weight:500 !important;padding:14px 32px !important;width:100% !important;
     letter-spacing:.05em !important;transition:opacity .2s !important;}
.pbtn button:hover{opacity:.88 !important;}
.stats{display:flex;gap:12px;margin-top:16px;}
.sc{flex:1;background:var(--surf2);border:1px solid var(--border);border-radius:10px;padding:14px;text-align:center;}
.sv{font-size:1.4em;font-weight:700;color:var(--accent);}
.sl{font-size:.7em;color:var(--muted);margin-top:4px;letter-spacing:.08em;text-transform:uppercase;}
.ftr{text-align:center;color:var(--muted);font-size:.75em;margin-top:20px;padding:16px;border-top:1px solid var(--border);}
.steps{color:#8892b0;font-size:.78em;line-height:2.0;margin-top:12px;}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title='Arabic ASR') as demo:

    gr.HTML("""
    <div class="hdr">
        <div class="htitle">🎙️ Arabic Speech Recognition</div>
        <div class="htitle" style="font-size:1.3em;margin-top:-8px;">نظام التعرف على الكلام العربي</div>
        <p class="hsub">Fine-tuned Whisper-small · Mozilla Common Voice Arabic · 20,000 samples</p>
        <span class="badge">WER 24.24%</span>
        <span class="badge">CER 11.14%</span>
        <span class="badge">Whisper-small 244M</span>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Microphone ─────────────────────────────────────────────────
        with gr.Tab('🎤  Microphone  |  ميكروفون'):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=['panel']):
                    mic_in = gr.Audio(
                        sources=['microphone'],
                        type='numpy',
                        label='RECORD  |  تسجيل',
                    )
                    mic_btn = gr.Button(
                        '🔠  Transcribe  |  نسخ الكلام',
                        elem_classes=['pbtn']
                    )
                    gr.HTML("""
                    <div class="steps">
                        1️⃣  Click the microphone icon to start recording<br>
                        2️⃣  Speak in Arabic<br>
                        3️⃣  Click Stop (■)<br>
                        4️⃣  Click <b>Transcribe</b> button above
                    </div>
                    """)

                with gr.Column(scale=1, elem_classes=['panel']):
                    mic_text = gr.Textbox(
                        label='TRANSCRIPTION  |  النص المُستخرج',
                        lines=5, placeholder='النص سيظهر هنا...',
                        elem_classes=['txbox'], interactive=False
                    )
                    mic_info = gr.Textbox(
                        label='PROCESSING INFO', lines=1,
                        interactive=False, elem_classes=['infobox']
                    )

            mic_btn.click(
                fn=transcribe_mic,
                inputs=[mic_in],
                outputs=[mic_text, mic_info]
            )

        # ── Tab 2: File Upload ────────────────────────────────────────────────
        with gr.Tab('📁  Upload File  |  رفع ملف'):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=['panel']):
                    file_in = gr.Audio(
                        sources=['upload'],
                        type='filepath',
                        label='UPLOAD  |  رفع ملف صوتي',
                    )
                    file_btn = gr.Button(
                        '🔠  Transcribe  |  نسخ الكلام',
                        elem_classes=['pbtn']
                    )
                    gr.HTML("""
                    <div class="steps">
                        ✅ WAV — always works<br>
                        ✅ MP3 / FLAC / OGG — works (ffmpeg installed)
                    </div>
                    """)

                with gr.Column(scale=1, elem_classes=['panel']):
                    file_text = gr.Textbox(
                        label='TRANSCRIPTION  |  النص المُستخرج',
                        lines=5, placeholder='النص سيظهر هنا...',
                        elem_classes=['txbox'], interactive=False
                    )
                    file_info = gr.Textbox(
                        label='PROCESSING INFO', lines=1,
                        interactive=False, elem_classes=['infobox']
                    )
                    gr.HTML("""
                    <div class="stats">
                        <div class="sc"><div class="sv">24.24%</div><div class="sl">Best Val WER</div></div>
                        <div class="sc"><div class="sv">11.14%</div><div class="sl">Test CER</div></div>
                        <div class="sc"><div class="sv">20k</div><div class="sl">Train Samples</div></div>
                        <div class="sc"><div class="sv">3.8h</div><div class="sl">Train Time</div></div>
                    </div>
                    """)

            file_btn.click(fn=transcribe_file, inputs=[file_in], outputs=[file_text, file_info])
            file_in.change(fn=transcribe_file, inputs=[file_in], outputs=[file_text, file_info])

    gr.HTML("""
    <div class="ftr">
        Model: <code>openai/whisper-small</code> fine-tuned &nbsp;·&nbsp;
        Dataset: Mozilla Common Voice Arabic v24 &nbsp;·&nbsp;
        Hardware: RTX 3060 6GB &nbsp;·&nbsp;
        Framework: PyTorch + HuggingFace Transformers
    </div>
    """)


if __name__ == '__main__':
    print('\n' + '='*55)
    print('  Arabic ASR Demo')
    print('  Open : http://127.0.0.1:7860')
    print('  Stop : Ctrl+C')
    print('='*55 + '\n')
    demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=True, share=False)
