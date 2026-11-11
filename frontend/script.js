
const API_URL = "http://127.0.0.1:5000";

// ================= NAVIGATION =================

function showDashboard() {
    document.getElementById("dashboard-page").style.display = "block";
    document.getElementById("new-scan-page").style.display = "none";
    document.getElementById("history-page").style.display = "none";

    updateActiveMenu(0);
    loadStats();
    loadRecentScans();
}

function showNewScan() {
    document.getElementById("dashboard-page").style.display = "none";
    document.getElementById("new-scan-page").style.display = "block";
    document.getElementById("history-page").style.display = "none";

    updateActiveMenu(1);
}

function showHistory() {
    document.getElementById("dashboard-page").style.display = "none";
    document.getElementById("new-scan-page").style.display = "none";
    document.getElementById("history-page").style.display = "block";

    updateActiveMenu(2);
    loadHistory();
}

function updateActiveMenu(index) {
    const links = document.querySelectorAll(".sidebar nav a");

    links.forEach(link => link.classList.remove("active"));

    if (links[index]) {
        links[index].classList.add("active");
    }
}


// ================= IMAGE PREVIEW =================

const scanInput = document.getElementById("scanInput");
const analyzeButton = document.getElementById("analyze-btn");

scanInput.addEventListener("change", function () {
    const file = this.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
        alert("Please select a valid image.");
        this.value = "";
        return;
    }

    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const resultContainer = document.getElementById("result-container");

    imagePreview.src = URL.createObjectURL(file);
    previewContainer.style.display = "block";

    // Hide previous result when a new image is selected
    resultContainer.style.display = "none";

    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze X-ray";
});


// ================= X-RAY ANALYSIS =================

analyzeButton.addEventListener("click", async function () {
    const file = scanInput.files[0];

    if (!file) {
        alert("Please select an X-ray image first.");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Analysis failed");
        }

        const prediction = String(data.prediction || "").toLowerCase();
        const confidence = Number(data.confidence);

        if (!["normal", "pneumonia"].includes(prediction)) {
            throw new Error("Unexpected prediction received.");
        }

        if (!Number.isFinite(confidence) || confidence < 0 || confidence > 100) {
            throw new Error("Invalid confidence value received.");
        }

        // Show result
        document.getElementById("result-container").style.display = "block";

        const predictionElement =
            document.getElementById("prediction-result");

        const confidenceElement =
            document.getElementById("confidence-result");

        const confidenceFill =
            document.getElementById("confidence-fill");

        predictionElement.textContent =
            "Prediction: " + prediction.toUpperCase();

        confidenceElement.textContent =
            "Model confidence: " + confidence.toFixed(2) + "%";

        // Apply result colours
        predictionElement.className = prediction;
        confidenceFill.className = prediction;

        // Update confidence bar
        confidenceFill.style.width = confidence + "%";

        // Refresh dashboard data
        loadStats();
        loadRecentScans();

    } catch (error) {
        alert("Error: " + error.message);
        console.error("Analysis error:", error);

    } finally {
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze X-ray";
    }
});


// ================= SCAN HISTORY =================

async function loadHistory() {
    const tableBody = document.getElementById("history-table-body");

    tableBody.innerHTML =
        `<tr><td colspan="5">Loading...</td></tr>`;

    try {
        const response = await fetch(`${API_URL}/history`);

        if (!response.ok) {
            throw new Error("Unable to load scan history");
        }

        const records = await response.json();

        tableBody.innerHTML = "";

        if (records.length === 0) {
            tableBody.innerHTML =
                `<tr><td colspan="5">No scans available.</td></tr>`;
            return;
        }

        records.forEach(record => {
            const row = document.createElement("tr");

            const confidence = Number(record.confidence);

            const values = [
                record.id,
                record.filename,
                record.date,
                record.prediction
                    ? record.prediction.toUpperCase()
                    : "Not available",
                record.confidence !== null &&
                record.confidence !== undefined &&
                Number.isFinite(confidence)
                    ? confidence.toFixed(2) + "%"
                    : "Not available"
            ];

            values.forEach((value, index) => {
                const cell = document.createElement("td");
                cell.textContent = value;

                if (index === 3) {
                    if (record.prediction === "normal") {
                        cell.className = "normal-result";
                    } else if (record.prediction === "pneumonia") {
                        cell.className = "pneumonia-result";
                    } else {
                        cell.className = "unavailable";
                    }
                }

                row.appendChild(cell);
            });

            tableBody.appendChild(row);
        });

    } catch (error) {
        tableBody.innerHTML =
            `<tr><td colspan="5">Unable to load scan history.</td></tr>`;

        console.error("History error:", error);
    }
}


// ================= DASHBOARD STATISTICS =================

async function loadStats() {
    try {
        const response = await fetch(`${API_URL}/stats`);

        if (!response.ok) {
            throw new Error("Unable to load statistics");
        }

        const data = await response.json();

        document.getElementById("total-scans").textContent =
            data.total_scans;

        document.getElementById("normal-count").textContent =
            data.normal_count;

        document.getElementById("pneumonia-count").textContent =
            data.pneumonia_count;

    } catch (error) {
        console.error("Statistics error:", error);
    }
}


// ================= RECENT SCANS =================

async function loadRecentScans() {
    const tableBody = document.getElementById("recent-scans-body");

    try {
        const response = await fetch(`${API_URL}/history`);

        if (!response.ok) {
            throw new Error("Unable to load recent scans");
        }

        const records = await response.json();

        tableBody.innerHTML = "";

        if (records.length === 0) {
            tableBody.innerHTML =
                `<tr><td colspan="4">No scans available.</td></tr>`;
            return;
        }

        records.slice(0, 5).forEach(record => {
            const row = document.createElement("tr");

            const confidence = Number(record.confidence);

            const values = [
                record.filename,
                record.date,
                record.prediction
                    ? record.prediction.toUpperCase()
                    : "Not available",
                record.confidence !== null &&
                record.confidence !== undefined &&
                Number.isFinite(confidence)
                    ? confidence.toFixed(2) + "%"
                    : "Not available"
            ];

            values.forEach((value, index) => {
                const cell = document.createElement("td");
                cell.textContent = value;

                if (index === 2) {
                    if (record.prediction === "normal") {
                        cell.className = "normal-result";
                    } else if (record.prediction === "pneumonia") {
                        cell.className = "pneumonia-result";
                    }
                }

                row.appendChild(cell);
            });

            tableBody.appendChild(row);
        });

    } catch (error) {
        tableBody.innerHTML =
            `<tr><td colspan="4">Unable to load recent scans.</td></tr>`;

        console.error("Recent scans error:", error);
    }
}


// ================= INITIAL LOAD =================

loadStats();
loadRecentScans();


