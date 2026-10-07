/**
 * DINAMIKA TRACER STUDY NLP - CLIENT-SIDE INTERACTION ENGINE
 * Menangani routing tab, live prediction, batch upload, visualisasi probability bar,
 * WordCloud switcher, dan Active Learning Human-in-the-Loop review.
 */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initHealthCheck();
    initLiveInference();
    initBatchUpload();
    initAnalytics();
    initActiveLearning();
});

// ==========================================
// 1. TAB ROUTING & NAVIGATION
// ==========================================
function initTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");

            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const activeContent = document.getElementById(targetTab);
            if (activeContent) {
                activeContent.classList.add("active");
            }

            // Trigger data refresh on tab switch
            if (targetTab === "tab-analytics") {
                loadAnalyticsData();
            } else if (targetTab === "tab-active-learning") {
                loadActiveLearningQueue();
            }
        });
    });
}

// ==========================================
// 2. HEALTH CHECK & SYSTEM STATUS
// ==========================================
async function initHealthCheck() {
    try {
        const res = await fetch("/health");
        if (!res.ok) throw new Error("Health check failed");
        const data = await res.json();

        const statusText = document.getElementById("system-status-text");
        const deviceName = document.getElementById("device-name");

        if (statusText) {
            statusText.textContent = `IndoBERT Online (${data.server_uptime})`;
        }
        if (deviceName) {
            deviceName.textContent = data.cuda_available ? `GPU: ${data.cuda_device_name}` : `CPU (${data.device})`;
        }
    } catch (err) {
        console.warn("Server offline atau belum siap:", err);
        const statusText = document.getElementById("system-status-text");
        if (statusText) statusText.textContent = "Offline Mode / Disconnected";
    }
}

// ==========================================
// 3. LIVE INFERENCE STUDIO
// ==========================================
function initLiveInference() {
    const inputText = document.getElementById("input-text");
    const charCounter = document.getElementById("char-counter");
    const btnPredict = document.getElementById("btn-predict");
    const btnClear = document.getElementById("btn-clear");
    const thresholdInput = document.getElementById("conf-threshold");
    const sampleChips = document.querySelectorAll(".sample-chip");

    // Results elements
    const emptyState = document.getElementById("result-empty-state");
    const loadingState = document.getElementById("result-loading-state");
    const resultContent = document.getElementById("result-content");
    const latencyTag = document.getElementById("latency-tag");

    const cleanedTextDisplay = document.getElementById("cleaned-text-display");
    const topicLabel = document.getElementById("topic-label-display");
    const topicConfBadge = document.getElementById("topic-conf-badge");
    const sentimentLabel = document.getElementById("sentiment-label-display");
    const sentimentConfBadge = document.getElementById("sentiment-conf-badge");
    const alAlert = document.getElementById("al-alert");
    const topicProbBars = document.getElementById("topic-prob-bars");
    const sentimentProbBars = document.getElementById("sentiment-prob-bars");

    // Char counter
    inputText.addEventListener("input", () => {
        charCounter.textContent = `${inputText.value.length} karakter`;
    });

    // Sample chip click
    sampleChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const sampleText = chip.getAttribute("data-sample");
            inputText.value = sampleText;
            charCounter.textContent = `${sampleText.length} karakter`;
            btnPredict.click();
        });
    });

    // Clear button
    btnClear.addEventListener("click", () => {
        inputText.value = "";
        charCounter.textContent = "0 karakter";
        emptyState.classList.remove("hidden");
        loadingState.classList.add("hidden");
        resultContent.classList.add("hidden");
        latencyTag.textContent = "Latency: -- ms";
    });

    // Predict button
    btnPredict.addEventListener("click", async () => {
        const text = inputText.value.trim();
        if (!text) {
            alert("Silakan masukkan teks kuesioner terlebih dahulu!");
            return;
        }

        const threshold = parseFloat(thresholdInput.value) || 0.70;

        // UI Transition to Loading
        emptyState.classList.add("hidden");
        resultContent.classList.add("hidden");
        loadingState.classList.remove("hidden");

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, threshold })
            });

            if (!res.ok) throw new Error("Gagal melakukan prediksi");
            const data = await res.json();

            // Render Results
            loadingState.classList.add("hidden");
            resultContent.classList.remove("hidden");

            latencyTag.textContent = `Latency: ${data.latency_ms} ms`;
            cleanedTextDisplay.textContent = data.teks_bersih || data.teks_asli;

            // Topic Data
            if (data.topic) {
                topicLabel.textContent = data.topic.label_name;
                topicConfBadge.textContent = data.topic.confidence_percentage;
                renderTopicBars(data.topic.probabilities, data.topic.label_name, topicProbBars);
            }

            // Sentiment Data
            if (data.sentiment) {
                sentimentLabel.textContent = data.sentiment.label_name;
                sentimentConfBadge.textContent = data.sentiment.confidence_percentage;
                renderSentimentBars(data.sentiment.probabilities, data.sentiment.label_name, sentimentProbBars);
            }

            // Active Learning Warning
            if (data.requires_human_review) {
                alAlert.classList.remove("hidden");
                updateQueueBadgeCount();
            } else {
                alAlert.classList.add("hidden");
            }

        } catch (err) {
            loadingState.classList.add("hidden");
            emptyState.classList.remove("hidden");
            alert(`Error saat menghubungi server inferensi: ${err.message}`);
        }
    });
}

function renderTopicBars(probs, activeLabel, container) {
    container.innerHTML = "";
    for (const [label, score] of Object.entries(probs)) {
        const percentage = (score * 100).toFixed(1);
        const isWinning = label === activeLabel;

        const row = document.createElement("div");
        row.className = "prob-bar-row";
        row.innerHTML = `
            <div class="prob-bar-label-row">
                <span style="${isWinning ? 'font-weight: 700; color: #67e8f9;' : ''}">${label}</span>
                <span>${percentage}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: ${percentage}%;"></div>
            </div>
        `;
        container.appendChild(row);
    }
}

function renderSentimentBars(probs, activeLabel, container) {
    container.innerHTML = "";
    for (const [label, score] of Object.entries(probs)) {
        const percentage = (score * 100).toFixed(1);
        const isPos = label.toLowerCase().includes("positif");
        const fillClass = isPos ? "sentiment-pos" : "sentiment-neg";

        const row = document.createElement("div");
        row.className = "prob-bar-row";
        row.innerHTML = `
            <div class="prob-bar-label-row">
                <span style="font-weight: ${label === activeLabel ? '700' : '400'}; color: ${isPos ? '#34d399' : '#fb7185'}">${label}</span>
                <span>${percentage}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill ${fillClass}" style="width: ${percentage}%;"></div>
            </div>
        `;
        container.appendChild(row);
    }
}

// ==========================================
// 4. BATCH FILE ANALYZER
// ==========================================
function initBatchUpload() {
    const dropzone = document.getElementById("upload-dropzone");
    const fileInput = document.getElementById("batch-file-input");
    const btnBrowse = document.getElementById("btn-browse-file");
    const fileInfo = document.getElementById("selected-file-info");
    const fileNameText = document.getElementById("file-name-text");
    const btnStart = document.getElementById("btn-start-batch");
    const colInput = document.getElementById("column-name-input");

    const progressContainer = document.getElementById("batch-progress-container");
    const progressFill = document.getElementById("batch-progress-fill");
    const percentText = document.getElementById("batch-percent-text");
    const summaryPanel = document.getElementById("batch-summary-panel");

    let selectedFile = null;

    btnBrowse.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("click", (e) => {
        if (e.target !== btnBrowse && !btnBrowse.contains(e.target)) {
            fileInput.click();
        }
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        fileNameText.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        fileInfo.classList.remove("hidden");
        btnStart.removeAttribute("disabled");
    }

    // Start Batch Processing
    btnStart.addEventListener("click", async () => {
        if (!selectedFile) return;

        btnStart.setAttribute("disabled", "true");
        progressContainer.classList.remove("hidden");
        summaryPanel.classList.add("hidden");
        progressFill.style.width = "20%";
        percentText.textContent = "20%";

        const formData = new FormData();
        formData.append("file", selectedFile);
        if (colInput.value.trim()) {
            formData.append("text_column", colInput.value.trim());
        }

        try {
            // Simulated progress steps
            const progressInterval = setInterval(() => {
                let cur = parseInt(progressFill.style.width);
                if (cur < 85) {
                    cur += 15;
                    progressFill.style.width = `${cur}%`;
                    percentText.textContent = `${cur}%`;
                }
            }, 500);

            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            clearInterval(progressInterval);

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Upload batch gagal");
            }

            const data = await res.json();

            progressFill.style.width = "100%";
            percentText.textContent = "100% Selesai";

            setTimeout(() => {
                progressContainer.classList.add("hidden");
                renderBatchSummary(data);
                btnStart.removeAttribute("disabled");
            }, 600);

        } catch (err) {
            progressContainer.classList.add("hidden");
            btnStart.removeAttribute("disabled");
            alert(`Gagal memproses batch: ${err.message}`);
        }
    });
}

function renderBatchSummary(data) {
    const summaryPanel = document.getElementById("batch-summary-panel");
    const statTotal = document.getElementById("stat-total-rows");
    const statFlagged = document.getElementById("stat-flagged-rows");
    const statTopTopic = document.getElementById("stat-top-topic");
    const statSentiment = document.getElementById("stat-sentiment-ratio");
    const btnDownload = document.getElementById("btn-download-annotated");
    const tableBody = document.getElementById("batch-table-body");

    summaryPanel.classList.remove("hidden");

    statTotal.textContent = data.total_rows_processed;
    statFlagged.textContent = data.total_requires_review;

    // Dominant topic
    let maxTopic = "--";
    let maxTopicCount = 0;
    for (const [t, cnt] of Object.entries(data.topic_summary || {})) {
        if (cnt > maxTopicCount) {
            maxTopicCount = cnt;
            maxTopic = `${t} (${cnt})`;
        }
    }
    statTopTopic.textContent = maxTopic;

    // Sentiment breakdown
    const posCount = data.sentiment_summary["Positif"] || 0;
    const negCount = data.sentiment_summary["Negatif"] || 0;
    statSentiment.textContent = `Pos: ${posCount} | Neg: ${negCount}`;

    // Download button link
    btnDownload.href = data.download_url;

    // Render Preview Table
    tableBody.innerHTML = "";
    (data.sample_preview || []).forEach(row => {
        const tr = document.createElement("tr");
        const statusBadge = row.requires_review
            ? '<span class="badge badge-warning">Review Needed</span>'
            : '<span class="badge badge-success">High Conf.</span>';

        tr.innerHTML = `
            <td>${row.row_index + 1}</td>
            <td style="max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${row.text}">${row.text}</td>
            <td><strong>${row.topic}</strong></td>
            <td>${(row.topic_confidence * 100).toFixed(1)}%</td>
            <td><span class="badge ${row.sentiment === 'Positif' ? 'badge-success' : 'badge-danger'}">${row.sentiment}</span></td>
            <td>${(row.sentiment_confidence * 100).toFixed(1)}%</td>
            <td>${statusBadge}</td>
        `;
        tableBody.appendChild(tr);
    });
}

// ==========================================
// 5. ANALYTICS & WORDCLOUD
// ==========================================
async function initAnalytics() {
    // Confusion Matrix Switcher
    const btnCmTopic = document.getElementById("btn-cm-topic");
    const btnCmSentiment = document.getElementById("btn-cm-sentiment");
    const cmImage = document.getElementById("cm-image");

    if (btnCmTopic && btnCmSentiment && cmImage) {
        btnCmTopic.addEventListener("click", () => {
            btnCmTopic.classList.add("active");
            btnCmSentiment.classList.remove("active");
            cmImage.src = "/reports/figures/confusion_matrix_topic.png";
        });

        btnCmSentiment.addEventListener("click", () => {
            btnCmSentiment.classList.add("active");
            btnCmTopic.classList.remove("active");
            cmImage.src = "/reports/figures/confusion_matrix_sentiment.png";
        });
    }

    loadAnalyticsData();
}

async function loadAnalyticsData() {
    try {
        const res = await fetch("/api/analytics/summary");
        if (!res.ok) return;
        const data = await res.json();

        // Update metrics
        if (data.metrics) {
            const topicMetrics = data.metrics.topic_model;
            const sentimentMetrics = data.metrics.sentiment_model;

            if (topicMetrics) {
                const f1 = document.getElementById("metric-topic-f1");
                const acc = document.getElementById("metric-topic-acc");
                if (f1) f1.textContent = `${(topicMetrics.macro_f1 * 100).toFixed(1)}%`;
                if (acc) acc.textContent = `${(topicMetrics.accuracy * 100).toFixed(1)}%`;
            }

            if (sentimentMetrics) {
                const f1 = document.getElementById("metric-sentiment-f1");
                const acc = document.getElementById("metric-sentiment-acc");
                if (f1) f1.textContent = `${(sentimentMetrics.macro_f1 * 100).toFixed(1)}%`;
                if (acc) acc.textContent = `${(sentimentMetrics.accuracy * 100).toFixed(1)}%`;
            }
        }

        // Populate WordCloud Gallery Tabs
        const wcNav = document.getElementById("wordcloud-nav");
        const activeWcImg = document.getElementById("active-wordcloud-img");

        if (wcNav && data.wordclouds && data.wordclouds.length > 0) {
            wcNav.innerHTML = "";
            data.wordclouds.forEach((wc, idx) => {
                const chip = document.createElement("button");
                chip.className = `btn btn-sm btn-outline ${idx === 0 ? 'active' : ''}`;
                chip.textContent = wc.title;
                chip.addEventListener("click", () => {
                    wcNav.querySelectorAll("button").forEach(b => b.classList.remove("active"));
                    chip.classList.add("active");
                    activeWcImg.src = wc.url;
                });
                wcNav.appendChild(chip);
            });
        }

    } catch (err) {
        console.warn("Gagal memuat ringkasan analitik:", err);
    }
}

// ==========================================
// 6. ACTIVE LEARNING & RETRAINING
// ==========================================
function initActiveLearning() {
    const btnRefresh = document.getElementById("btn-refresh-queue");
    const btnTriggerRetrain = document.getElementById("btn-trigger-retrain");

    if (btnRefresh) {
        btnRefresh.addEventListener("click", loadActiveLearningQueue);
    }

    if (btnTriggerRetrain) {
        btnTriggerRetrain.addEventListener("click", triggerRetraining);
    }

    loadActiveLearningQueue();
}

async function updateQueueBadgeCount() {
    try {
        const res = await fetch("/api/active-learning/queue");
        if (!res.ok) return;
        const data = await res.json();
        const badge = document.getElementById("queue-badge-count");
        if (badge) badge.textContent = data.total_in_queue;
    } catch (e) {
        // silent
    }
}

async function loadActiveLearningQueue() {
    const queueList = document.getElementById("review-cards-list");
    const emptyState = document.getElementById("queue-empty-state");
    const verifiedCount = document.getElementById("verified-samples-count");
    const badge = document.getElementById("queue-badge-count");

    try {
        // 1. Get queue
        const res = await fetch("/api/active-learning/queue");
        if (!res.ok) return;
        const data = await res.json();

        if (badge) badge.textContent = data.total_in_queue;

        // 2. Get verified count from analytics summary
        const summaryRes = await fetch("/api/analytics/summary");
        if (summaryRes.ok) {
            const sumData = await summaryRes.json();
            if (verifiedCount && sumData.active_learning_stats) {
                verifiedCount.textContent = `${sumData.active_learning_stats.total_human_verified_samples} Sampel`;
            }
        }

        if (data.total_in_queue === 0) {
            emptyState.classList.remove("hidden");
            queueList.innerHTML = "";
            return;
        }

        emptyState.classList.add("hidden");
        queueList.innerHTML = "";

        data.queue.forEach(item => {
            const card = document.createElement("div");
            card.className = "review-card";
            card.id = `review-card-${item.id}`;

            const curTopicId = item.predicted_topic ? item.predicted_topic.label_id : 0;
            const curSentimentId = item.predicted_sentiment ? item.predicted_sentiment.label_id : 0;

            card.innerHTML = `
                <div class="review-text-quote">"${item.text}"</div>
                <div class="review-meta" style="font-size: 11px; color: #94a3b8;">
                    <span>Alasan: <strong style="color: #f59e0b;">${item.reason}</strong></span> • 
                    <span>Waktu: ${item.timestamp}</span>
                </div>
                <div class="review-controls">
                    <div>
                        <label style="font-size: 11px; color: #94a3b8;">Koreksi Topik:</label>
                        <select class="review-select" id="sel-topic-${item.id}">
                            <option value="0" ${curTopicId === 0 ? 'selected' : ''}>0: Kurikulum & Pembelajaran</option>
                            <option value="1" ${curTopicId === 1 ? 'selected' : ''}>1: Fasilitas & Sarana Prasarana</option>
                            <option value="2" ${curTopicId === 2 ? 'selected' : ''}>2: Dosen & Tenaga Pengajar</option>
                            <option value="3" ${curTopicId === 3 ? 'selected' : ''}>3: Karier, Magang & Kesiapan Kerja</option>
                            <option value="4" ${curTopicId === 4 ? 'selected' : ''}>4: Layanan Kemahasiswaan</option>
                        </select>
                    </div>
                    <div>
                        <label style="font-size: 11px; color: #94a3b8;">Koreksi Sentimen:</label>
                        <select class="review-select" id="sel-sentiment-${item.id}">
                            <option value="0" ${curSentimentId === 0 ? 'selected' : ''}>0: Negatif (Keluhan)</option>
                            <option value="1" ${curSentimentId === 1 ? 'selected' : ''}>1: Positif (Apresiasi)</option>
                        </select>
                    </div>
                </div>
                <button class="btn btn-sm btn-primary" onclick="submitReview('${item.id}', '${encodeURIComponent(item.text)}')">
                    <i class="fa-solid fa-check"></i> Simpan Koreksi & Masukkan Buffer Pelatihan
                </button>
            `;
            queueList.appendChild(card);
        });

    } catch (err) {
        console.warn("Gagal memuat queue active learning:", err);
    }
}

// Global window function for inline onclick
window.submitReview = async function(queueId, encodedText) {
    const text = decodeURIComponent(encodedText);
    const selTopic = document.getElementById(`sel-topic-${queueId}`);
    const selSentiment = document.getElementById(`sel-sentiment-${queueId}`);

    const payload = {
        queue_id: queueId,
        original_text: text,
        corrected_topic_id: parseInt(selTopic.value),
        corrected_sentiment_id: parseInt(selSentiment.value),
        reviewer_notes: "Diverifikasi via Web Dashboard Human-in-the-Loop"
    };

    try {
        const res = await fetch("/api/active-learning/review", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Gagal mengirim review");
        const resData = await res.json();

        // Animate card removal
        const card = document.getElementById(`review-card-${queueId}`);
        if (card) {
            card.style.opacity = "0";
            card.style.transform = "translateX(30px)";
            setTimeout(() => {
                card.remove();
                loadActiveLearningQueue();
            }, 300);
        }

    } catch (err) {
        alert(`Gagal menyimpan review: ${err.message}`);
    }
};

async function triggerRetraining() {
    const btn = document.getElementById("btn-trigger-retrain");
    const logBox = document.getElementById("retrain-log-output");

    if (!confirm("Apakah Anda yakin ingin memicu siklus Continuous Retraining sekarang?")) return;

    btn.setAttribute("disabled", "true");
    logBox.innerHTML += `<br><span style="color: #eab308;">[${new Date().toLocaleTimeString()}] Memulai pipeline retraining di server background...</span>`;

    try {
        const res = await fetch("/api/retrain/trigger", {
            method: "POST"
        });

        if (!res.ok) throw new Error("Gagal memicu retraining");
        const data = await res.json();

        logBox.innerHTML += `<br><span style="color: #22c55e;">[${new Date().toLocaleTimeString()}] ✅ ${data.message}</span>`;
        logBox.innerHTML += `<br><span style="color: #38bdf8;">[STATUS] Silakan tunggu beberapa saat atau pantau terminal console server.</span>`;
        logBox.scrollTop = logBox.scrollHeight;

        setTimeout(() => {
            btn.removeAttribute("disabled");
        }, 5000);

    } catch (err) {
        btn.removeAttribute("disabled");
        logBox.innerHTML += `<br><span style="color: #ef4444;">[ERROR] ${err.message}</span>`;
    }
}
