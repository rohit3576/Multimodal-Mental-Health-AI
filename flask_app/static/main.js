/* =========================================================
   AI Mental Health Analyzer — Frontend Logic
   Auto-refresh when Streamlit sends video result
========================================================= */

/* =========================
   TEXT ANALYSIS LOADER
========================= */
function showTextLoader() {
    const loader = document.getElementById("textLoader");
    if (loader) loader.style.display = "block";
}

/* =========================
   POLL FLASK FOR VIDEO RESULT
========================= */
function pollForVideoResult() {
    fetch("/api/status")
        .then(res => res.json())
        .then(data => {
            if (data.has_video) {
                console.log("✅ Video result detected — refreshing UI");
                window.location.reload();
            } else {
                // Retry after 2 seconds
                setTimeout(pollForVideoResult, 2000);
            }
        })
        .catch(err => {
            console.warn("⚠ Status check failed, retrying...", err);
            setTimeout(pollForVideoResult, 3000);
        });
}

/* =========================
   START POLLING ON PAGE LOAD
========================= */
window.addEventListener("load", () => {
    const textLoader = document.getElementById("textLoader");
    if (textLoader) textLoader.style.display = "none";

    // Only poll if video result is not already visible
    const videoSection = document.querySelector(".video-result");
    if (!videoSection) {
        pollForVideoResult();
    }
});
