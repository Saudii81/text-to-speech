# VoxScribe — Real-Time Speech to Text

A premium web-based speech recognition system that converts spoken language into accurate text output in real time.

## Features

- **Real-time transcription** — Instant interim and final results with confidence scores
- **30+ languages & dialects** — English, Spanish, French, Chinese, Arabic, Hindi, and many more
- **Noise filtering** — High-pass `BiquadFilter` via Web Audio API reduces background noise
- **Live waveform visualization** — Frequency-bar canvas animation responds to your voice
- **Export** — Copy to clipboard or download transcript as a `.txt` file
- **Keyboard shortcut** — Press `Space` to toggle recording
- **Responsive** — Works on desktop, tablet, and mobile screens
- **Zero dependencies** — Pure HTML, CSS, and JavaScript

## How to Run

1. Open `index.html` in **Google Chrome** or **Microsoft Edge** (required for Web Speech API)
2. Click the 🎤 microphone button (or press `Space`)
3. Grant microphone permission when prompted
4. Start speaking — your words appear in real time

## Browser Compatibility

| Browser | Supported |
|---------|-----------|
| Chrome  | ✅ Full    |
| Edge    | ✅ Full    |
| Safari  | ⚠️ Partial |
| Firefox | ❌ No Web Speech API |

## Architecture

```
index.html       — Semantic HTML structure
index.css        — Dark glassmorphism design system
app.js           — Speech engine, visualizer, transcript manager
languages.js     — Language/dialect configuration (30 languages)
main.py          — TensorFlow speech command classifier (ML pipeline)
```

## License

MIT
