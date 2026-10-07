const uploadForm = document.getElementById("uploadForm");
const videoInput = document.getElementById("videoFile");
const dropZone = document.getElementById("dropZone");
const browseBtn = document.getElementById("browseBtn");
const fileInfo = document.getElementById("fileInfo");

const loading = document.getElementById("loadingSection");
const result = document.getElementById("resultSection");

const predictionLabel = document.getElementById("predictionLabel");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");

const fakeProbability = document.getElementById("fakeProbability");
const realProbability = document.getElementById("realProbability");
const framesContainer = document.getElementById("framesContainer");

const progressBar = document.getElementById("progressBar");


// ======================================================
// FILE DISPLAY
// ======================================================

function showSelectedFile(file) {

    if (!file) return;

    fileInfo.innerHTML = `
        <div class="selected-file">
            <h4>✅ ${escapeHtml(file.name)}</h4>
            <p>${(file.size / 1024 / 1024).toFixed(2)} MB</p>
        </div>
    `;
}


// ======================================================
// HTML ESCAPE
// ======================================================

function escapeHtml(text) {

    const div = document.createElement("div");
    div.textContent = text;

    return div.innerHTML;
}


// ======================================================
// BROWSE BUTTON
// ======================================================

browseBtn.addEventListener("click", (e) => {

    e.stopPropagation();

    videoInput.click();

});


// ======================================================
// DROP ZONE CLICK
// ======================================================

dropZone.addEventListener("click", (e) => {

    if (e.target === browseBtn) return;

    videoInput.click();

});


// ======================================================
// FILE SELECTED
// ======================================================

videoInput.addEventListener("change", () => {

    if (videoInput.files.length > 0) {

        showSelectedFile(videoInput.files[0]);

        result.style.display = "none";

    }

});


// ======================================================
// DRAG OVER
// ======================================================

dropZone.addEventListener("dragover", (e) => {

    e.preventDefault();

    dropZone.classList.add("drag");

});


// ======================================================
// DRAG LEAVE
// ======================================================

dropZone.addEventListener("dragleave", () => {

    dropZone.classList.remove("drag");

});


// ======================================================
// DROP
// ======================================================

dropZone.addEventListener("drop", (e) => {

    e.preventDefault();

    dropZone.classList.remove("drag");

    if (e.dataTransfer.files.length > 0) {

        const file = e.dataTransfer.files[0];

        const allowedExtensions = [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm"
        ];

        const filename = file.name.toLowerCase();

        const valid = allowedExtensions.some(
            extension => filename.endsWith(extension)
        );

        if (!valid) {

            alert(
                "Unsupported video format.\n\n" +
                "Please upload MP4, AVI, MOV, MKV or WEBM."
            );

            return;

        }

        try {

            videoInput.files = e.dataTransfer.files;

        } catch (error) {

            console.log("Could not assign dropped file:", error);

        }

        showSelectedFile(file);

        result.style.display = "none";

    }

});


// ======================================================
// PROGRESS ANIMATION
// ======================================================

function startProgress() {

    let progress = 0;

    progressBar.style.width = "0%";
    progressBar.innerHTML = "0%";

    window.progressTimer = setInterval(() => {

        if (progress < 90) {

            progress += Math.random() * 8;

            if (progress > 90) {
                progress = 90;
            }

            progressBar.style.width = `${progress}%`;
            progressBar.innerHTML = `${Math.floor(progress)}%`;

        }

    }, 500);

}


function finishProgress() {

    clearInterval(window.progressTimer);

    progressBar.style.width = "100%";
    progressBar.innerHTML = "100%";

}


// ======================================================
// DISPLAY EXTRACTED FRAMES
// ======================================================

function displayFrames(frames) {

    framesContainer.innerHTML = "";

    if (!frames || frames.length === 0) {

        framesContainer.innerHTML = `
            <p class="text-muted">
                No frame previews available.
            </p>
        `;

        return;

    }

    frames.forEach((frame, index) => {

        const frameWrapper = document.createElement("div");

        frameWrapper.className = "frame-item";

        frameWrapper.innerHTML = `
            <img
                src="data:image/jpeg;base64,${frame}"
                alt="Extracted frame ${index + 1}"
                loading="lazy"
            >

            <p>
                Frame ${index + 1}
            </p>
        `;

        framesContainer.appendChild(frameWrapper);

    });

}


// ======================================================
// DISPLAY RESULT
// ======================================================

function displayResult(data) {

    predictionLabel.innerHTML = data.label;

    confidenceValue.innerHTML =
        `${Number(data.confidence).toFixed(2)} %`;

    confidenceBar.style.width =
        `${data.confidence}%`;

    confidenceBar.innerHTML =
        `${Number(data.confidence).toFixed(2)}%`;


    // -------------------------------
    // Probabilities
    // -------------------------------

    fakeProbability.innerHTML =
        `${Number(data.fake_probability).toFixed(2)}%`;

    realProbability.innerHTML =
        `${Number(data.real_probability).toFixed(2)}%`;


    // -------------------------------
    // Prediction styling
    // -------------------------------

    confidenceBar.classList.remove(
        "bg-success",
        "bg-danger"
    );

    if (data.label === "FAKE") {

        predictionLabel.style.color = "#ff3b3b";

        confidenceBar.classList.add("bg-danger");

    } else {

        predictionLabel.style.color = "#00ff95";

        confidenceBar.classList.add("bg-success");

    }


    // -------------------------------
    // Extracted frames
    // -------------------------------

    displayFrames(data.frames);


    // -------------------------------
    // Show result
    // -------------------------------

    result.style.display = "block";

    setTimeout(() => {

        result.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }, 100);

}


// ======================================================
// FORM SUBMISSION
// ======================================================

uploadForm.addEventListener("submit", async (e) => {

    e.preventDefault();


    // -------------------------------
    // Validate file
    // -------------------------------

    if (!videoInput.files || videoInput.files.length === 0) {

        alert("Please choose a video.");

        return;

    }


    const file = videoInput.files[0];


    // -------------------------------
    // Create FormData
    // -------------------------------

    const formData = new FormData();

    formData.append("file", file);


    // -------------------------------
    // UI state
    // -------------------------------

    loading.style.display = "block";

    result.style.display = "none";

    startProgress();


    try {

        const response = await fetch(
            "/api/predict",
            {
                method: "POST",
                body: formData
            }
        );


        // -------------------------------
        // Read response safely
        // -------------------------------

        let data;

        try {

            data = await response.json();

        } catch (jsonError) {

            throw new Error(
                `Server returned an invalid response (HTTP ${response.status}).`
            );

        }


        finishProgress();

        loading.style.display = "none";


        // -------------------------------
        // Backend error
        // -------------------------------

        if (!response.ok || data.success === false || data.error) {

            throw new Error(
                data.error ||
                `Prediction failed (HTTP ${response.status}).`
            );

        }


        // -------------------------------
        // Display result
        // -------------------------------

        displayResult(data);


    } catch (error) {

        clearInterval(window.progressTimer);

        loading.style.display = "none";

        progressBar.style.width = "0%";
        progressBar.innerHTML = "0%";

        console.error("Prediction error:", error);

        alert(
            "Prediction failed.\n\n" +
            error.message
        );

    }

});