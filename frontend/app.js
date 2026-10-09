/* AI Plant Disease Detector — frontend logic */

(function () {
  "use strict";

  var API_BASE_URL = (window.APP_CONFIG && window.APP_CONFIG.API_BASE_URL) || "";
  if (API_BASE_URL && API_BASE_URL.slice(-1) === "/") {
    API_BASE_URL = API_BASE_URL.slice(0, -1);
  }
  var PREDICT_ENDPOINT = API_BASE_URL + "/api/predict";
  var MAX_FILE_BYTES = 10 * 1024 * 1024; // 10 MB (backend enforces this too)
  var ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

  // ---------- DOM ----------

  var dropzone = document.getElementById("dropzone");
  var fileInput = document.getElementById("file-input");
  var browseBtn = document.getElementById("browse-btn");
  var previewArea = document.getElementById("preview-area");
  var previewImage = document.getElementById("preview-image");
  var fileName = document.getElementById("file-name");
  var detectBtn = document.getElementById("detect-btn");
  var resetBtn = document.getElementById("reset-btn");
  var loading = document.getElementById("loading");
  var errorArea = document.getElementById("error-area");
  var resultSection = document.getElementById("result-section");
  var predictedClass = document.getElementById("predicted-class");
  var confidenceValue = document.getElementById("confidence-value");
  var confidenceBarFill = document.getElementById("confidence-bar-fill");
  var topPredictions = document.getElementById("top-predictions");

  var selectedFile = null;
  var isSubmitting = false;
  var previewObjectUrl = null;

  // ---------- Helpers ----------

  function showError(message) {
    errorArea.textContent = message;
    errorArea.hidden = false;
  }

  function clearError() {
    errorArea.textContent = "";
    errorArea.hidden = true;
  }

  function setLoading(isLoading) {
    loading.hidden = !isLoading;
    detectBtn.disabled = isLoading || !selectedFile;
    resetBtn.hidden = isLoading;
  }

  function formatClass(rawName) {
    return String(rawName)
      .replace(/___/g, " — ")
      .replace(/_/g, " ");
  }

  function formatPercent(value) {
    return (value * 100).toFixed(1) + "%";
  }

  // ---------- File selection ----------

  function handleFile(file) {
    clearError();

    if (!file) {
      return;
    }

    if (ALLOWED_TYPES.indexOf(file.type) === -1) {
      showError(
        "Unsupported file type (" + (file.type || "unknown") + "). " +
        "Please upload a JPG, PNG, WebP or BMP image."
      );
      return;
    }

    if (file.size === 0) {
      showError("The selected file is empty. Please choose another image.");
      return;
    }

    if (file.size > MAX_FILE_BYTES) {
      showError(
        "The file is too large (" + (file.size / (1024 * 1024)).toFixed(1) +
        " MB). Maximum size is 10 MB."
      );
      return;
    }

    selectedFile = file;

    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
    }
    previewObjectUrl = URL.createObjectURL(file);
    previewImage.src = previewObjectUrl;
    fileName.textContent = file.name + " (" + (file.size / 1024).toFixed(0) + " KB)";
    previewArea.hidden = false;

    detectBtn.disabled = false;
    resultSection.hidden = true;
  }

  function resetForm() {
    selectedFile = null;
    isSubmitting = false;
    fileInput.value = "";
    previewArea.hidden = true;
    previewImage.src = "";
    fileName.textContent = "";
    detectBtn.disabled = true;
    resetBtn.hidden = true;
    setLoading(false);
    clearError();
    resultSection.hidden = true;
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }
  }

  // ---------- API ----------

  function renderResult(data) {
    predictedClass.textContent = formatClass(data.predicted_class);
    confidenceValue.textContent = formatPercent(data.confidence);
    resultSection.hidden = false;

    // Animate the confidence bar after the browser paints the section.
    requestAnimationFrame(function () {
      confidenceBarFill.style.width = (data.confidence * 100).toFixed(1) + "%";
    });

    topPredictions.innerHTML = "";
    (data.top_predictions || []).forEach(function (item, index) {
      var li = document.createElement("li");

      var rank = document.createElement("span");
      rank.className = "rank";
      rank.textContent = index + 1;

      var name = document.createElement("span");
      name.className = "top-class";
      name.textContent = formatClass(item.class_name);

      var conf = document.createElement("span");
      conf.className = "top-confidence";
      conf.textContent = formatPercent(item.confidence);

      li.appendChild(rank);
      li.appendChild(name);
      li.appendChild(conf);
      topPredictions.appendChild(li);
    });

    resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function runPrediction() {
    if (!selectedFile || isSubmitting) {
      return;
    }

    isSubmitting = true;
    clearError();
    setLoading(true);

    var formData = new FormData();
    // Do NOT set Content-Type manually: the browser sets the
    // correct multipart boundary when FormData is passed to fetch.
    formData.append("file", selectedFile, selectedFile.name);

    var request = new Request(PREDICT_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    fetch(request)
      .then(function (response) {
        return response.json().then(function (body) {
          return { status: response.status, body: body };
        });
      })
      .then(function (result) {
        if (result.status === 200 && result.body.success) {
          renderResult(result.body);
          resetBtn.hidden = false;
        } else if (result.status === 413) {
          showError("The image is too large. Maximum size is 10 MB.");
        } else if (result.status === 400) {
          showError("Invalid image: " + (result.body.error || "please upload a clear leaf photo."));
        } else if (result.status === 503) {
          showError(
            "The AI model is not available on the server right now. " +
            "It may still be starting up (cold start) — please try again in a moment."
          );
        } else if (result.status >= 500) {
          showError("The server encountered an error. Please try again later.");
        } else {
          showError(result.body.error || "Prediction failed. Please try another image.");
        }
      })
      .catch(function () {
        showError(
          "Could not reach the AI backend. Check your connection, make sure the " +
          "backend URL is configured correctly (config.js), and verify the backend " +
          "is running and allows requests from this site (CORS)."
        );
      })
      .finally(function () {
        isSubmitting = false;
        setLoading(false);
      });
  }

  // ---------- Events ----------

  browseBtn.addEventListener("click", function (event) {
    event.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("click", function () {
    fileInput.click();
  });

  dropzone.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", function () {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach(function (eventName) {
    dropzone.addEventListener(eventName, function (event) {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(function (eventName) {
    dropzone.addEventListener(eventName, function (event) {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", function (event) {
    var files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  });

  detectBtn.addEventListener("click", runPrediction);
  resetBtn.addEventListener("click", resetForm);
})();
