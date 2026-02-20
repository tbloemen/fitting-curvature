/**
 * Main application: config management, WebSocket, UI state.
 */

(function () {
  "use strict";

  // --- Toast notifications ---
  let toastContainer = document.getElementById("toast-container");
  if (!toastContainer) {
    toastContainer = document.createElement("div");
    toastContainer.id = "toast-container";
    document.body.appendChild(toastContainer);
  }

  function toast(message, type) {
    type = type || "info";
    const el = document.createElement("div");
    el.className = "toast " + type;
    el.textContent = message;
    toastContainer.appendChild(el);
    setTimeout(function () {
      el.remove();
    }, 4000);
  }

  // --- Config form helpers ---
  const fields = {
    dataset: document.getElementById("cfg-dataset"),
    n_samples: document.getElementById("cfg-n-samples"),
    embed_dim: document.getElementById("cfg-embed-dim"),
    n_iterations: document.getElementById("cfg-n-iterations"),
    init_method: document.getElementById("cfg-init-method"),
    perplexity: document.getElementById("cfg-perplexity"),
    early_exag_iters: document.getElementById("cfg-early-exag-iters"),
    early_exag_factor: document.getElementById("cfg-early-exag-factor"),
    momentum_early: document.getElementById("cfg-momentum-early"),
    momentum_main: document.getElementById("cfg-momentum-main"),
    learning_rate: document.getElementById("cfg-learning-rate"),
    init_scale: document.getElementById("cfg-init-scale"),
    curvature: document.getElementById("cfg-curvature"),
    projection: document.getElementById("cfg-projection"),
  };

  function populateForm(config) {
    fields.dataset.value = config.data.dataset;
    fields.n_samples.value = config.data.n_samples;
    fields.embed_dim.value = config.embedding.embed_dim;
    fields.n_iterations.value = config.embedding.n_iterations;
    fields.init_method.value = config.embedding.init_method;
    fields.perplexity.value = config.embedding.perplexity;
    fields.early_exag_iters.value =
      config.embedding.early_exaggeration_iterations;
    fields.early_exag_factor.value = config.embedding.early_exaggeration_factor;
    fields.momentum_early.value = config.embedding.momentum_early;
    fields.momentum_main.value = config.embedding.momentum_main;

    var lr = config.hyperparameters.learning_rates;
    fields.learning_rate.value = lr.k || lr[Object.keys(lr)[0]] || 200;
    fields.init_scale.value = config.hyperparameters.init_scale;

    var curvatures = config.experiments.curvatures;
    fields.curvature.value = curvatures.length > 0 ? curvatures[0] : 0;

    if (config.visualization && config.visualization.spherical_projection) {
      fields.projection.value = config.visualization.spherical_projection;
    }
  }

  function buildConfig() {
    var initScale = fields.init_scale.value.trim();
    if (initScale !== "auto") {
      initScale = parseFloat(initScale);
    }
    return {
      data: {
        dataset: fields.dataset.value,
        n_samples: parseInt(fields.n_samples.value),
      },
      embedding: {
        embed_dim: parseInt(fields.embed_dim.value),
        n_iterations: parseInt(fields.n_iterations.value),
        init_method: fields.init_method.value,
        perplexity: parseFloat(fields.perplexity.value),
        early_exaggeration_iterations: parseInt(fields.early_exag_iters.value),
        early_exaggeration_factor: parseFloat(fields.early_exag_factor.value),
        momentum_early: parseFloat(fields.momentum_early.value),
        momentum_main: parseFloat(fields.momentum_main.value),
      },
      hyperparameters: {
        learning_rates: {
          k: parseFloat(fields.learning_rate.value),
        },
        init_scale: initScale,
      },
      experiments: {
        curvatures: [parseFloat(fields.curvature.value)],
      },
      evaluation: { n_neighbors: 5 },
      visualization: {
        spherical_projection: fields.projection.value,
      },
    };
  }

  // --- Load datasets into select ---
  fetch("/api/datasets")
    .then(function (r) {
      return r.json();
    })
    .then(function (data) {
      var select = fields.dataset;
      var current = select.value;
      select.innerHTML = "";
      data.datasets.forEach(function (d) {
        var opt = document.createElement("option");
        opt.value = d;
        opt.textContent = d;
        select.appendChild(opt);
      });
      if (current) select.value = current;
    });

  // --- Load config on page load ---
  fetch("/api/config")
    .then(function (r) {
      return r.json();
    })
    .then(function (config) {
      populateForm(config);
    });

  // --- Save / Reset buttons ---
  document.getElementById("btn-save").addEventListener("click", function () {
    var config = buildConfig();
    fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    })
      .then(function (r) {
        return r.json();
      })
      .then(function (result) {
        if (result.ok) {
          toast("Configuration saved", "success");
        } else {
          toast("Invalid config: " + result.error, "error");
        }
      });
  });

  document.getElementById("btn-reset").addEventListener("click", function () {
    fetch("/api/config/reset", { method: "POST" })
      .then(function (r) {
        return r.json();
      })
      .then(function (result) {
        if (result.ok) {
          populateForm(result.config);
          toast("Config reset to defaults", "info");
        }
      });
  });

  // --- Training controls ---
  var btnStart = document.getElementById("btn-start");
  var btnStop = document.getElementById("btn-stop");
  var statusLabel = document.getElementById("status-label");
  var iterationLabel = document.getElementById("iteration-label");
  var phaseLabel = document.getElementById("phase-label");
  var progressBar = document.getElementById("progress-bar");
  var lossLabel = document.getElementById("loss-label");

  btnStart.addEventListener("click", function () {
    var config = buildConfig();
    LossChart.reset();

    fetch("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    })
      .then(function (r) {
        return r.json();
      })
      .then(function (result) {
        if (result.ok) {
          btnStart.disabled = true;
          btnStop.disabled = false;
          setStatus("running", "Running");
          toast("Training started", "success");
        } else {
          toast("Error: " + result.error, "error");
        }
      });
  });

  btnStop.addEventListener("click", function () {
    fetch("/api/training/stop", { method: "POST" });
    btnStop.disabled = true;
    toast("Stopping training...", "info");
  });

  function setStatus(cls, text) {
    statusLabel.className = "status " + cls;
    statusLabel.textContent = text;
  }

  // --- WebSocket ---
  var ws;
  var currentCurvature = 0;

  function connectWebSocket() {
    var protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(protocol + "//" + location.host + "/ws");
    ws.binaryType = "arraybuffer";

    ws.onmessage = function (event) {
      if (typeof event.data === "string") {
        handleJsonMessage(JSON.parse(event.data));
      } else {
        handleBinaryMessage(event.data);
      }
    };

    ws.onclose = function () {
      // Reconnect after 2 seconds
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = function () {
      ws.close();
    };
  }

  function handleJsonMessage(msg) {
    if (msg.type === "update") {
      currentCurvature = msg.curvature;
      var iter = msg.iteration + 1;
      iterationLabel.textContent = iter + " / " + msg.max_iterations;
      var pct = msg.max_iterations > 0 ? (iter / msg.max_iterations) * 100 : 0;
      progressBar.style.width = pct + "%";

      if (msg.loss > 0) {
        lossLabel.textContent = msg.loss.toFixed(6);
      }

      // Phase
      if (msg.phase === "early") {
        phaseLabel.textContent = "(Early Exaggeration)";
        phaseLabel.style.color = "orange";
      } else if (msg.phase === "main") {
        phaseLabel.textContent = "(Main Phase)";
        phaseLabel.style.color = "green";
      } else if (msg.phase === "precomputing") {
        phaseLabel.textContent = "(Computing affinities)";
        phaseLabel.style.color = "purple";
      } else {
        phaseLabel.textContent = "";
      }

      setStatus("running", "Running");

      // Loss chart
      LossChart.addPoint(msg.iteration, msg.loss);
    } else if (msg.type === "status") {
      var statusMap = {
        idle: ["idle", "Idle"],
        precomputing: ["precomputing", "Precomputing..."],
        running: ["running", "Running"],
        completed: ["completed", "Completed"],
        stopped: ["stopped", "Stopped"],
        error: ["error", "Error"],
      };
      var s = statusMap[msg.status] || ["idle", "Idle"];
      setStatus(s[0], s[1]);

      if (
        msg.status === "completed" ||
        msg.status === "stopped" ||
        msg.status === "error"
      ) {
        btnStart.disabled = false;
        btnStop.disabled = true;
      }

      if (msg.status === "error" && msg.message) {
        toast("Training error: " + msg.message, "error");
      }
    } else if (msg.type === "boundary") {
      currentCurvature = msg.curvature;
      if (msg.points) {
        ThreeJSPlot.setBoundary(new Float32Array(msg.points));
      } else {
        ThreeJSPlot.hideBoundary();
      }
    }
  }

  function handleBinaryMessage(buffer) {
    var data = new Float32Array(buffer);
    var n = data.length / 5;

    // Extract positions and colors
    var positions = new Float32Array(n * 2);
    var colors = new Float32Array(n * 3);

    for (var i = 0; i < n; i++) {
      positions[i * 2] = data[i * 5];
      positions[i * 2 + 1] = data[i * 5 + 1];
      colors[i * 3] = data[i * 5 + 2];
      colors[i * 3 + 1] = data[i * 5 + 3];
      colors[i * 3 + 2] = data[i * 5 + 4];
    }

    // Build title
    var geometry =
      currentCurvature === 0
        ? "Euclidean"
        : currentCurvature > 0
          ? "Spherical"
          : "Hyperbolic";
    var title =
      "Embedding Visualization - " +
      geometry +
      " (k=" +
      currentCurvature.toFixed(4) +
      ")\n" +
      iterationLabel.textContent;

    ThreeJSPlot.updatePlot(positions, colors, null, title);
  }

  // --- Initialize visualizations ---
  ThreeJSPlot.initPlot("embedding-container");
  LossChart.init("loss-chart-container");
  connectWebSocket();
})();
