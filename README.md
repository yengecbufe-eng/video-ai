# VidyoAI 🎬

**A text-to-video AI that runs 100% locally — no LLMs, no APIs, no PyTorch.**

You type a prompt in plain Turkish or English, and VidyoAI generates a complete
video: scene, motion, ambient soundtrack, subtitles, and even a spoken
narration. Everything is computed on your machine with nothing but
**NumPy + OpenCV**.

```
python vidyo_ai.py "sunset over the sea with birds" --konus "Hello! I am VidyoAI."
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Trained understanding** | A pure-NumPy MLP (~21,000-word auto-generated training corpus) maps your prompt words to scene labels |
| 🌄 **Procedural scenes** | Sky gradients, sun/moon with glow, twinkling stars, nebulas, clouds, mountains, city skylines with lit windows, animated sea with light shimmer |
| 🌧️ **Effects** | Rain, lightning, snow, fire particles, flying birds, fish schools, underwater light beams, film grain, vignette |
| 🎵 **Ambient soundtrack** | Procedurally synthesized with NumPy (chord pads per mood, sea waves, rain, thunder, wind, fire crackle, birdsong) and muxed into the MP4 |
| 📝 **Subtitles** | Cinematic fade-in caption of your prompt (full Turkish character support) |
| 🗣️ **Speech** | Optional offline text-to-speech via Windows' built-in speech engine (`--konus`) |
| 🎨 **Style network** | A tiny Fourier-feature MLP trained per-prompt gives every video a unique color signature |

**No internet connection is required at any point.**

---

## 📦 Requirements

- Python 3.8+
- NumPy
- OpenCV (`opencv-python`)
- Pillow *(optional — improves subtitle font rendering)*
- ffmpeg on PATH *(optional — embeds audio into the MP4; otherwise audio is saved as a separate `.wav`)*

```bash
pip install numpy opencv-python pillow
```

---

## 🚀 Usage

```bash
# Basic (defaults: 6 s, 24 fps, 960x540, audio + subtitles on)
python vidyo_ai.py "sunset over the sea with birds"

# Full control
python vidyo_ai.py "rainy night city" \
    --saniye 8 --fps 30 --cozunurluk 1280x720 \
    --cikis my_clip.mp4

# Make it talk (uses Windows offline TTS)
python vidyo_ai.py "space nebula stars" --konus "Hello! I am VidyoAI."

# Image only
python vidyo_ai.py "snowy mountains" --sessiz --altyazisiz
```

### Options

| Flag | Default | Description |
|---|---|---|
| `prompt` | `"sunset over the sea with birds"` | Free-text scene description (Turkish or English) |
| `--saniye` | `6.0` | Video duration in seconds (auto-extends to fit speech) |
| `--fps` | `24` | Frame rate |
| `--cozunurluk` | `960x540` | Resolution as `WIDTHxHEIGHT` |
| `--cikis` | `vidyo.mp4` | Output file |
| `--konus` | — | Text the video should speak (offline Windows TTS) |
| `--hiz` | `0` | Speech rate, `-10` (slow) to `+10` (fast) |
| `--sessiz` | off | Disable the ambient soundtrack |
| `--altyazisiz` | off | Disable subtitles |
| `--ornek` | `5000` | Number of training sentences generated per run |
| `--epoch` | `400` | Training epochs for the understanding network |

---

## 🗣️ Scene vocabulary

The understanding network is trained on these concepts — mix and match freely
(Turkish keywords shown, English equivalents work too):

| Concept | Keywords |
|---|---|
| Sea | `deniz`, `okyanus`, `dalga`, `sahil`, `kumsal` |
| Mountains | `dağ`, `tepe`, `zirve`, `vadi` |
| City | `şehir`, `kent`, `gökdelen`, `sokak` |
| Forest | `orman`, `ağaç`, `çam` |
| Space | `uzay`, `galaksi`, `nebula`, `yıldız`, `gezegen` |
| Rain | `yağmur`, `sağanak`, `fırtına`, `şimşek` |
| Snow | `kar`, `buz`, `kış` |
| Fire | `ateş`, `alev`, `lav`, `volkan` |
| Birds | `kuş`, `kuşlar`, `martı`, `uçan` |
| Sunset | `günbatımı`, `alacakaranlık`, `akşam` |
| Night | `gece`, `karanlık`, `ay` |
| Neon | `neon`, `siberpunk`, `synthwave`, `retro` |
| Clouds | `bulut`, `sis`, `pus` |
| Underwater | `balık`, `denizaltı`, `sualtı`, `mercan` |

Unknown or misspelled words are matched fuzzily (`denizde` → sea), so you don't
have to be exact.

---

## 🔬 How it works

1. **Corpus generation (at runtime).** Every run synthesizes thousands of
   Turkish scene sentences (~21,000 words by default) from grammar templates
   and a curated vocabulary, each labeled with the scenes it describes. The
   corpus is *not* stored in the source code — it is built deterministically
   from a fixed seed each time the program starts.

2. **Understanding network.** A 2-layer MLP (bag-of-words → 14 sigmoid scene
   labels) is trained from scratch with hand-written gradient descent and
   binary cross-entropy loss — no PyTorch, no autograd. It reaches ~93% label
   accuracy. Your prompt is tokenized, fuzzy-matched to the vocabulary, and
   turned into scene flags.

3. **Style network.** A separate tiny Fourier-feature MLP is trained per
   prompt (seeded from a hash of your text) to modulate the scene's color
   field over time — every video gets a unique visual signature.

4. **Procedural renderer.** Each frame is painted with vectorized NumPy:
   value-noise/fBm terrain and clouds, analytic sun/moon glow, particle
   systems, silhouettes, and post-processing. Frames are written to MP4 via
   OpenCV.

5. **Sound.** The ambient track is synthesized with NumPy (chord pads, filtered
   noise layers, event-based chirps and thunder) and mixed with the optional
   TTS narration. If `ffmpeg` is available the mix is embedded into the MP4;
   otherwise it is saved next to it as a `.wav`.

---

## 🌍 Enabling Turkish speech

The narrator automatically prefers a Turkish voice. If only an English voice
(e.g. *Microsoft Zira*) is installed, Turkish text is read with an English
accent. To fix:

> **Windows Settings → Time & Language → Speech → Manage voices → Add voices → Turkish**

No code changes needed — VidyoAI picks the voice up automatically.

---

## 📁 Project layout

```
vidyo_ai.py      # the whole AI (training, rendering, sound, speech) in one file
README.md        # this file
```

---

*Built as a from-scratch experiment: an "AI" with no LLM, no API and no deep
learning framework — just math, NumPy, and a bit of imagination.*
