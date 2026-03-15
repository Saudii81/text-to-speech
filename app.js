/**
 * VoxScribe — Speech-to-Text Recognition Engine
 * Real-time transcription with noise filtering and waveform visualization
 */

(function () {
    'use strict';

    /* ========================================
       DOM References
       ======================================== */
    const $ = (sel) => document.querySelector(sel);
    const micBtn = $('#micBtn');
    const micWrapper = $('#micWrapper');
    const micStatus = $('#micStatus');
    const languageSelect = $('#languageSelect');
    const dialectSelect = $('#dialectSelect');
    const noiseToggle = $('#noiseFilterToggle');
    const canvas = $('#visualizer');
    const transcriptBody = $('#transcriptBody');
    const placeholder = $('#placeholder');
    const copyBtn = $('#copyBtn');
    const downloadBtn = $('#downloadBtn');
    const clearBtn = $('#clearBtn');
    const wordCountEl = $('#wordCount');
    const avgConfEl = $('#avgConfidence');
    const durationEl = $('#duration');
    const currentLangEl = $('#currentLang');
    const toast = $('#toast');
    const errorBanner = $('#errorBanner');
    const errorMessage = $('#errorMessage');
    const errorClose = $('#errorClose');
    const browserWarning = $('#browserWarning');
    const controlsSection = $('#controlsSection');

    /* ========================================
       Feature Detection
       ======================================== */
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        browserWarning.classList.add('visible');
        controlsSection.style.display = 'none';
        return;
    }

    /* ========================================
       State
       ======================================== */
    let isRecording = false;
    let recognition = null;
    let audioCtx = null;
    let analyser = null;
    let micStream = null;
    let sourceNode = null;
    let noiseFilter = null;
    let animFrameId = null;
    let startTime = null;
    let durationInterval = null;

    // Transcript data
    const segments = [];          // { text, confidence, timestamp }
    let interimText = '';
    let confidenceSum = 0;
    let confidenceCount = 0;

    /* ========================================
       Language Population
       ======================================== */
    function populateLanguages() {
        LANGUAGES.forEach((lang, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = `${lang.flag}  ${lang.name}`;
            languageSelect.appendChild(opt);
        });
        updateDialects();
    }

    function updateDialects() {
        dialectSelect.innerHTML = '';
        const lang = LANGUAGES[languageSelect.value];
        lang.dialects.forEach((d) => {
            const opt = document.createElement('option');
            opt.value = d.code;
            opt.textContent = d.name;
            dialectSelect.appendChild(opt);
        });
        currentLangEl.textContent = dialectSelect.value;
    }

    languageSelect.addEventListener('change', () => {
        updateDialects();
        if (isRecording) restartRecognition();
    });

    dialectSelect.addEventListener('change', () => {
        currentLangEl.textContent = dialectSelect.value;
        if (isRecording) restartRecognition();
    });

    populateLanguages();

    /* ========================================
       Speech Recognition Engine
       ======================================== */
    function createRecognition() {
        const rec = new SpeechRecognition();
        rec.lang = dialectSelect.value;
        rec.continuous = true;
        rec.interimResults = true;
        rec.maxAlternatives = 1;

        rec.onresult = handleResult;
        rec.onerror = handleError;
        rec.onend = handleEnd;
        return rec;
    }

    function handleResult(event) {
        let interim = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const result = event.results[i];
            const transcript = result[0].transcript;

            if (result.isFinal) {
                const conf = result[0].confidence;
                segments.push({
                    text: transcript.trim(),
                    confidence: conf,
                    timestamp: elapsedTime(),
                });
                confidenceSum += conf;
                confidenceCount++;
                interimText = '';
            } else {
                interim += transcript;
            }
        }
        interimText = interim;
        renderTranscript();
        updateStats();
    }

    function handleError(event) {
        console.error('SpeechRecognition error:', event.error);
        const messages = {
            'no-speech': 'No speech detected. Please speak into the microphone.',
            'audio-capture': 'No microphone found. Ensure a mic is connected.',
            'not-allowed': 'Microphone access denied. Please allow microphone permission.',
            'network': 'Network error. Check your internet connection.',
            'aborted': 'Recognition aborted.',
        };
        const msg = messages[event.error] || `Error: ${event.error}`;

        if (event.error === 'no-speech') {
            // Don't stop — just show a brief toast
            showToast('🔇 No speech detected — keep talking');
            return;
        }

        showError(msg);
        if (event.error !== 'aborted') {
            stopRecording();
        }
    }

    function handleEnd() {
        // Auto-restart if still in recording mode (browser may stop on silence)
        if (isRecording) {
            try {
                recognition = createRecognition();
                recognition.start();
            } catch (e) {
                console.warn('Auto-restart failed:', e);
                stopRecording();
            }
        }
    }

    function restartRecognition() {
        if (recognition) {
            try { recognition.abort(); } catch (_) { }
        }
        recognition = createRecognition();
        try { recognition.start(); } catch (_) { }
    }

    /* ========================================
       Audio Visualizer & Noise Filter
       ======================================== */
    async function initAudio() {
        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            sourceNode = audioCtx.createMediaStreamSource(micStream);
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.75;

            // High-pass noise filter
            noiseFilter = audioCtx.createBiquadFilter();
            noiseFilter.type = 'highpass';
            noiseFilter.frequency.value = noiseToggle.checked ? 200 : 0;
            noiseFilter.Q.value = 0.7;

            sourceNode.connect(noiseFilter);
            noiseFilter.connect(analyser);

            drawVisualizer();
        } catch (err) {
            console.error('Audio init failed:', err);
            showError('Could not access microphone. Please check permissions.');
            stopRecording();
        }
    }

    noiseToggle.addEventListener('change', () => {
        if (noiseFilter) {
            noiseFilter.frequency.value = noiseToggle.checked ? 200 : 0;
        }
        showToast(noiseToggle.checked ? '🔇 Noise filter enabled' : '🔊 Noise filter disabled');
    });

    function drawVisualizer() {
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;

        function resize() {
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
        }
        resize();
        window.addEventListener('resize', resize);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function draw() {
            animFrameId = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);

            const w = canvas.width / dpr;
            const h = canvas.height / dpr;

            // Clear
            ctx.clearRect(0, 0, w, h);

            // Bar visualization
            const barCount = bufferLength;
            const barWidth = w / barCount;
            const gap = 1;

            for (let i = 0; i < barCount; i++) {
                const val = dataArray[i] / 255;
                const barH = val * h * 0.9;

                // Gradient color from cyan → blue → purple
                const hue = 190 + val * 80;
                const sat = 70 + val * 30;
                const light = 45 + val * 20;
                ctx.fillStyle = `hsla(${hue}, ${sat}%, ${light}%, ${0.6 + val * 0.4})`;

                const x = i * barWidth + gap / 2;
                const y = h - barH;
                ctx.fillRect(x, y, barWidth - gap, barH);

                // Mirror on top (subtle)
                ctx.fillStyle = `hsla(${hue}, ${sat}%, ${light}%, ${0.1 + val * 0.1})`;
                ctx.fillRect(x, 0, barWidth - gap, barH * 0.3);
            }
        }

        draw();
    }

    function drawIdleVisualizer() {
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        ctx.clearRect(0, 0, w, h);

        // Flat line
        ctx.strokeStyle = 'rgba(100, 116, 139, 0.3)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();
    }

    function stopAudio() {
        if (animFrameId) cancelAnimationFrame(animFrameId);
        animFrameId = null;
        if (audioCtx) {
            audioCtx.close().catch(() => { });
            audioCtx = null;
        }
        if (micStream) {
            micStream.getTracks().forEach(t => t.stop());
            micStream = null;
        }
        analyser = null;
        sourceNode = null;
        noiseFilter = null;
        drawIdleVisualizer();
    }

    // Draw idle state on load
    drawIdleVisualizer();
    window.addEventListener('resize', () => {
        if (!isRecording) drawIdleVisualizer();
    });

    /* ========================================
       Recording Control
       ======================================== */
    async function startRecording() {
        hideError();
        isRecording = true;
        micBtn.classList.add('recording');
        micWrapper.classList.add('recording');
        micBtn.textContent = '⏹';
        micBtn.title = 'Stop recording';
        micStatus.textContent = 'Listening…';
        micStatus.classList.add('active');

        startTime = Date.now();
        durationInterval = setInterval(updateDuration, 1000);

        await initAudio();

        recognition = createRecognition();
        try {
            recognition.start();
        } catch (e) {
            showError('Failed to start recognition: ' + e.message);
            stopRecording();
        }
    }

    function stopRecording() {
        isRecording = false;
        micBtn.classList.remove('recording');
        micWrapper.classList.remove('recording');
        micBtn.textContent = '🎤';
        micBtn.title = 'Start recording';
        micStatus.textContent = 'Click to start';
        micStatus.classList.remove('active');

        clearInterval(durationInterval);
        durationInterval = null;

        if (recognition) {
            try { recognition.stop(); } catch (_) { }
            recognition = null;
        }
        stopAudio();
    }

    micBtn.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    /* ========================================
       Transcript Rendering
       ======================================== */
    function renderTranscript() {
        // Show/hide placeholder
        if (segments.length === 0 && !interimText) {
            placeholder.style.display = 'flex';
            return;
        }
        placeholder.style.display = 'none';

        // Build HTML
        let html = '';
        segments.forEach((seg) => {
            const confPercent = Math.round(seg.confidence * 100);
            const confColor = confPercent >= 80 ? 'var(--accent-green)'
                : confPercent >= 50 ? 'var(--accent-amber)'
                    : 'var(--accent-red)';
            html += `
        <div class="transcript-segment">
          <div class="segment-time">${seg.timestamp}</div>
          <div class="segment-text">${escapeHTML(seg.text)}</div>
          <div class="segment-confidence" style="color:${confColor}">
            ● ${confPercent}% confidence
          </div>
        </div>`;
        });

        if (interimText) {
            html += `
        <div class="transcript-segment">
          <div class="segment-time">${elapsedTime()}</div>
          <div class="segment-text interim">${escapeHTML(interimText)}</div>
        </div>`;
        }

        transcriptBody.innerHTML = html;
        // Auto-scroll to bottom
        transcriptBody.scrollTop = transcriptBody.scrollHeight;
    }

    /* ========================================
       Stats
       ======================================== */
    function updateStats() {
        const allText = segments.map(s => s.text).join(' ');
        const words = allText.trim() ? allText.trim().split(/\s+/).length : 0;
        wordCountEl.textContent = words;

        if (confidenceCount > 0) {
            avgConfEl.textContent = Math.round((confidenceSum / confidenceCount) * 100) + '%';
        }
    }

    function updateDuration() {
        if (!startTime) return;
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const secs = String(elapsed % 60).padStart(2, '0');
        durationEl.textContent = `${mins}:${secs}`;
    }

    function elapsedTime() {
        if (!startTime) return '00:00';
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const secs = String(elapsed % 60).padStart(2, '0');
        return `${mins}:${secs}`;
    }

    /* ========================================
       Actions
       ======================================== */
    copyBtn.addEventListener('click', () => {
        const text = getTranscriptText();
        if (!text) return showToast('Nothing to copy');
        navigator.clipboard.writeText(text).then(() => {
            showToast('📋 Transcript copied!');
            copyBtn.classList.add('copied');
            setTimeout(() => copyBtn.classList.remove('copied'), 1500);
        });
    });

    downloadBtn.addEventListener('click', () => {
        const text = getTranscriptText();
        if (!text) return showToast('Nothing to download');

        const header = `VoxScribe Transcript\nLanguage: ${dialectSelect.value}\nDate: ${new Date().toLocaleString()}\n${'─'.repeat(40)}\n\n`;
        const blob = new Blob([header + text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `voxscribe-transcript-${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('💾 Transcript downloaded!');
    });

    clearBtn.addEventListener('click', () => {
        if (segments.length === 0 && !interimText) return;
        segments.length = 0;
        interimText = '';
        confidenceSum = 0;
        confidenceCount = 0;
        wordCountEl.textContent = '0';
        avgConfEl.textContent = '—';
        renderTranscript();
        showToast('🗑️ Transcript cleared');
    });

    function getTranscriptText() {
        return segments.map(s => `[${s.timestamp}] ${s.text}`).join('\n');
    }

    /* ========================================
       Toast & Error
       ======================================== */
    let toastTimer = null;
    function showToast(msg) {
        toast.textContent = msg;
        toast.classList.add('show');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => toast.classList.remove('show'), 2500);
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.classList.add('visible');
    }

    function hideError() {
        errorBanner.classList.remove('visible');
    }

    errorClose.addEventListener('click', hideError);

    /* ========================================
       Helpers
       ======================================== */
    function escapeHTML(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    /* ========================================
       Keyboard Shortcut
       ======================================== */
    document.addEventListener('keydown', (e) => {
        // Spacebar toggles recording (only when not typing in inputs)
        if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA' && e.target.tagName !== 'SELECT') {
            e.preventDefault();
            micBtn.click();
        }
    });

})();
