/**
 * CO5085 — Deep Learning & Applications in Computer Vision (HCMUT HK261)
 * Shared Client-Side Controller for GitHub Pages
 * Strictly adheres to plans/00_conventions.md:
 * - Official mode displays only verified `run_mode == "full"` artifacts.
 * - Preview mode allows inspecting interactive charts & tables with explicit [Preview Demo] labels.
 */

(function () {
  "use strict";

  const icon = (name, className = "ui-icon") =>
    window.CO5085Icons ? window.CO5085Icons.render(name, className) : "";

  const STORAGE_THEME_KEY = "co5085_theme";
  const STORAGE_PREVIEW_KEY = "co5085_preview_mode";

  const state = {
    theme: localStorage.getItem(STORAGE_THEME_KEY) || "light",
    previewMode: localStorage.getItem(STORAGE_PREVIEW_KEY) === "true",
    siteConfig: null,
    manifest: null,
    charts: {},
    a1FilterFamily: "all",
    a1FilterStrategy: "all",
    a1InspectorFamily: "resnet18",
    a1InspectorStrategy: "partial",
    e2FilterBackend: "all",
    e2FilterTokenizer: "all",
    a2SelectedStepIdx: 8,
  };

  // Apply theme immediately
  document.documentElement.setAttribute("data-theme", state.theme);

  async function fetchJsonWithFallback(url, fallbackObj) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (_err) {
      return fallbackObj;
    }
  }

  function formatPct(val) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    return (Number(val) * 100).toFixed(2) + "%";
  }

  function formatNum(val, decimals = 2) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    return Number(val).toFixed(decimals);
  }

  function formatInt(val) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    return Number(val).toLocaleString("en-US");
  }

  function getMetric(exp, key) {
    if (exp.has_full_metrics && exp.metrics && exp.metrics[key] !== undefined) {
      return { value: exp.metrics[key], source: "full" };
    }
    if (state.previewMode && exp.preview && exp.preview[key] !== undefined) {
      return { value: exp.preview[key], source: "preview" };
    }
    return { value: null, source: "pending" };
  }

  function renderMetricCell(exp, key, formatter) {
    const res = getMetric(exp, key);
    if (res.value === null) {
      return `<span class="badge badge-pending" title="Chưa có artifact full run tại artifacts/*/full/">Pending Full Run</span>`;
    }
    const formatted = formatter(res.value);
    if (res.source === "preview") {
      return `<span class="mono" title="Số liệu minh họa bố cục (Preview Layout)">${formatted} <sup style="color:var(--accent-purple);font-weight:700;">[Demo]</sup></span>`;
    }
    return `<span class="mono" style="font-weight:600;">${formatted}</span>`;
  }

  function destroyChart(id) {
    if (state.charts[id]) {
      state.charts[id].destroy();
      delete state.charts[id];
    }
  }

  function setupHeaderControls() {
    const themeBtn = document.getElementById("theme-toggle-btn");
    if (themeBtn) {
      themeBtn.innerHTML = icon(state.theme === "dark" ? "sun" : "moon");
      themeBtn.addEventListener("click", () => {
        state.theme = state.theme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", state.theme);
        localStorage.setItem(STORAGE_THEME_KEY, state.theme);
        themeBtn.innerHTML = icon(state.theme === "dark" ? "sun" : "moon");
        renderAllDynamicSections();
      });
    }

    const previewBtns = document.querySelectorAll(".js-preview-toggle");
    const updatePreviewUI = () => {
      previewBtns.forEach((btn) => {
        btn.setAttribute("data-preview", state.previewMode ? "true" : "false");
        btn.innerHTML = state.previewMode
          ? `${icon("eye")}<span>Chế độ: Xem thử biểu đồ (Preview ON)</span>`
          : `${icon("lock")}<span>Chế độ: Chính thức (Official Full-Run)</span>`;
      });
      const banner = document.getElementById("preview-mode-banner");
      if (banner) {
        banner.classList.toggle("visible", state.previewMode);
      }
    };

    previewBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        state.previewMode = !state.previewMode;
        localStorage.setItem(STORAGE_PREVIEW_KEY, state.previewMode ? "true" : "false");
        updatePreviewUI();
        renderAllDynamicSections();
      });
    });

    updatePreviewUI();
  }

  function injectSharedMetadata() {
    const cfg = state.siteConfig;
    if (!cfg) return;

    document.querySelectorAll("[data-bind='group-name']").forEach((el) => {
      el.textContent = cfg.team.group_name;
    });
    document.querySelectorAll("[data-bind='course-semester']").forEach((el) => {
      el.textContent = cfg.course.semester;
    });
    document.querySelectorAll("[data-bind='repo-link']").forEach((el) => {
      el.setAttribute("href", cfg.repository.repo_url);
    });

    // Bind specific repo subpaths
    document.querySelectorAll("[data-repo-path]").forEach((el) => {
      const subpath = el.getAttribute("data-repo-path");
      el.setAttribute("href", `${cfg.repository.repo_url}/blob/${cfg.repository.branch}/${subpath}`);
    });

    // Bind Colab links
    document.querySelectorAll("[data-colab-notebook]").forEach((el) => {
      const nb = el.getAttribute("data-colab-notebook");
      el.setAttribute("href", `${cfg.repository.colab_base_url}/${nb}`);
    });

    // Team table or cards on index.html
    const teamContainer = document.getElementById("team-members-container");
    if (teamContainer && cfg.team.members) {
      teamContainer.innerHTML = cfg.team.members
        .map(
          (m, idx) => `
        <div class="meta-card">
          <div class="meta-label">Thành viên 0${idx + 1}${m.student_id ? ` • MSSV: <span class="mono">${m.student_id}</span>` : ""}</div>
          <div class="meta-value" style="margin-bottom:0.25rem;">${m.name}</div>
          <div style="font-size:0.82rem;color:var(--text-secondary);">${m.role}</div>
        </div>
      `
        )
        .join("");
    }

    // Instructors container
    const instContainer = document.getElementById("instructors-container");
    if (instContainer && cfg.course.instructors) {
      instContainer.innerHTML = cfg.course.instructors
        .map(
          (inst) => `
        <div class="meta-card">
          <div class="meta-label">${inst.role}</div>
          <div class="meta-value">${inst.title} ${inst.name}</div>
          <div style="font-size:0.82rem;color:var(--text-secondary);">${cfg.course.faculty}</div>
        </div>
      `
        )
        .join("");
    }

    // Timeline on index.html
    const timelineEl = document.getElementById("timeline-container");
    if (timelineEl && cfg.timeline) {
      const statusBadgeMap = {
        completed: `<span class="badge badge-completed">${icon("check")}Đã hoàn thành</span>`,
        ready: `<span class="badge badge-ready">${icon("circle")}Sẵn sàng nộp</span>`,
        in_progress: `<span class="badge badge-pending">${icon("clock")}Đang thực nghiệm</span>`,
        pending: `<span class="badge badge-pending">${icon("circle")}Chưa đến hạn</span>`,
      };
      timelineEl.innerHTML = cfg.timeline
        .map(
          (item) => `
        <div class="timeline-item ${item.status}">
          <div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
              <span class="mono" style="font-weight:700;font-size:0.85rem;color:var(--brand-primary);">${item.week} • ${item.date_display}</span>
              ${statusBadgeMap[item.status] || ""}
            </div>
            <h3 style="font-size:1.02rem;margin-bottom:0.35rem;">${item.title}</h3>
            <p style="font-size:0.85rem;color:var(--text-secondary);margin:0 0 0.85rem;">${item.deliverable}</p>
          </div>
          <a href="${item.target_page}" class="btn btn-outline btn-sm" style="align-self:flex-start;">Xem chi tiết ${icon("arrowRight")}</a>
        </div>
      `
        )
        .join("");
    }

    // Global progress summary on index.html
    if (state.manifest) {
      const totalDone = state.manifest.total_completed_full_experiments || 0;
      const totalExp = state.manifest.total_expected_experiments || 23;
      document.querySelectorAll("[data-bind='artifact-progress']").forEach((el) => {
        el.textContent = `${totalDone} / ${totalExp} cấu hình Full-Run đã đồng bộ`;
      });
    }
  }

  /* ========================================================================
     EXERCISES PAGE (E1, E2, E3) RENDERING
     ======================================================================== */
  function renderExercisesPage() {
    if (!state.manifest || !state.manifest.assignments) return;
    const e1 = state.manifest.assignments.E1;
    const e2 = state.manifest.assignments.E2;
    const e3 = state.manifest.assignments.E3;

    // 1. E1 Table & Chart
    const e1Tbody = document.getElementById("e1-table-body");
    if (e1Tbody && e1) {
      e1Tbody.innerHTML = e1.experiments
        .map((exp) => {
          const params = exp.metrics?.parameters || exp.expected_params;
          const sizeMb = exp.metrics?.weight_size_mb || exp.expected_weight_mb;
          return `
            <tr>
              <td><code style="font-weight:700;">${exp.id}</code><div style="font-size:0.78rem;color:var(--text-muted);">${exp.name}</div></td>
              <td style="font-size:0.82rem;max-width:300px;"><code>${exp.architecture}</code></td>
              <td class="num-col">${formatInt(params)}</td>
              <td class="num-col">${formatNum(sizeMb, 3)} MB</td>
              <td class="num-col">${renderMetricCell(exp, "val_accuracy", formatPct)}</td>
              <td class="num-col">${renderMetricCell(exp, "test_accuracy", formatPct)}</td>
              <td class="num-col">${renderMetricCell(exp, "test_loss", (v) => formatNum(v, 4))}</td>
              <td class="num-col">${renderMetricCell(exp, "latency_ms_per_image", (v) => formatNum(v, 3) + " ms")}</td>
            </tr>
          `;
        })
        .join("");
      renderE1Chart(e1.experiments);
      renderArtifactGallery("e1-artifact-gallery", "E1", e1.experiments, [
        { key: "curves", label: "Learning Curves (Loss & Accuracy)" },
        { key: "confusion_matrix", label: "Confusion Matrix (10 lớp)" },
        { key: "failures", label: "High-Confidence Failure Cases" },
      ]);
    }

    // 2. E2 Table & Filter & Chart
    const e2Tbody = document.getElementById("e2-table-body");
    if (e2Tbody && e2) {
      const filteredE2 = e2.experiments.filter((exp) => {
        const matchB = state.e2FilterBackend === "all" || exp.backend === state.e2FilterBackend;
        const matchT = state.e2FilterTokenizer === "all" || exp.tokenizer === state.e2FilterTokenizer;
        return matchB && matchT;
      });

      e2Tbody.innerHTML = filteredE2
        .map((exp) => {
          const params = exp.metrics?.parameters || exp.expected_params;
          return `
            <tr>
              <td><code style="font-weight:700;">${exp.id}</code></td>
              <td><span class="badge ${exp.backend === "scratch" ? "badge-ready" : "badge-completed"}">${exp.backend === "scratch" ? "ScratchMSA (Tự viết)" : "TorchMSA (PyTorch)"}</span></td>
              <td><code>${exp.tokenizer}</code> <span class="mono" style="font-size:0.78rem;color:var(--text-muted);">(${exp.token_count} tokens)</span></td>
              <td class="num-col">${formatInt(params)}</td>
              <td class="num-col">${renderMetricCell(exp, "val_accuracy", formatPct)}</td>
              <td class="num-col">${renderMetricCell(exp, "test_accuracy", formatPct)}</td>
              <td class="num-col">${renderMetricCell(exp, "latency_ms_per_image", (v) => formatNum(v, 3) + " ms")}</td>
              <td class="num-col">${renderMetricCell(exp, "peak_gpu_memory_mb", (v) => formatNum(v, 1) + " MB")}</td>
            </tr>
          `;
        })
        .join("");
      renderE2Chart(e2.experiments);
    }

    // 3. E3 Table (including E1 MLP/CNN provenance baselines) & Chart
    const e3Tbody = document.getElementById("e3-table-body");
    if (e3Tbody && e3 && e1) {
      const mlpBase = e1.experiments.find((x) => x.id === "mlp");
      const cnnBase = e1.experiments.find((x) => x.id === "cnn");
      const rowsHtml = [];

      [mlpBase, cnnBase].forEach((base) => {
        if (!base) return;
        const params = base.metrics?.parameters || base.expected_params;
        rowsHtml.push(`
          <tr style="background:rgba(11, 87, 208, 0.04);">
            <td><code style="font-weight:700;">e1_${base.id}</code> <span class="badge badge-ready">Baseline E1</span></td>
            <td>Non-Recurrent (${base.name})</td>
            <td class="mono">${base.id === "mlp" ? "Vector 784D" : "Grid 28×28 (2D)"}</td>
            <td class="num-col">${formatInt(params)}</td>
            <td class="num-col">${renderMetricCell(base, "val_accuracy", formatPct)}</td>
            <td class="num-col">${renderMetricCell(base, "test_accuracy", formatPct)}</td>
            <td class="num-col">${renderMetricCell(base, "latency_ms_per_image", (v) => formatNum(v, 3) + " ms")}</td>
          </tr>
        `);
      });

      e3.experiments.forEach((exp) => {
        const params = exp.metrics?.parameters || exp.expected_params;
        rowsHtml.push(`
          <tr>
            <td><code style="font-weight:700;">${exp.id}</code></td>
            <td>Recurrent (<code>nn.${exp.recurrent}</code>, 1-layer, hidden=128)</td>
            <td class="mono">${exp.representation} (${exp.sequence_shape})</td>
            <td class="num-col">${formatInt(params)}</td>
            <td class="num-col">${renderMetricCell(exp, "val_accuracy", formatPct)}</td>
            <td class="num-col">${renderMetricCell(exp, "test_accuracy", formatPct)}</td>
            <td class="num-col">${renderMetricCell(exp, "latency_ms_per_image", (v) => formatNum(v, 3) + " ms")}</td>
          </tr>
        `);
      });

      e3Tbody.innerHTML = rowsHtml.join("");
      renderE3Chart(e3.experiments, mlpBase, cnnBase);
    }
  }

  function renderE1Chart(experiments) {
    const canvas = document.getElementById("e1-chart-canvas");
    if (!canvas || typeof Chart === "undefined") return;
    destroyChart("e1");

    const hasData = experiments.some((e) => e.has_full_metrics || state.previewMode);
    const emptyNotice = document.getElementById("e1-chart-empty");
    if (emptyNotice) emptyNotice.style.display = hasData ? "none" : "block";

    const labels = ["Ep 2", "Ep 4", "Ep 6", "Ep 8", "Ep 10", "Ep 12", "Ep 14", "Ep 16", "Ep 18", "Ep 20"];
    const colors = { softmax: "#64748b", mlp: "#0b57d0", cnn: "#059669" };

    const datasets = [];
    if (hasData) {
      experiments.forEach((exp) => {
        let curve = [];
        if (exp.has_full_metrics && exp.history && exp.history.length > 0) {
          curve = exp.history.map((h) => Number(h.val_accuracy) * 100);
        } else if (state.previewMode && exp.preview?.val_curve_acc) {
          curve = exp.preview.val_curve_acc.map((v) => v * 100);
        }
        if (curve.length > 0) {
          datasets.push({
            label: `${exp.name} (Val Acc %)${!exp.has_full_metrics ? " [Demo]" : ""}`,
            data: curve,
            borderColor: colors[exp.id] || "#0b57d0",
            backgroundColor: colors[exp.id] || "#0b57d0",
            tension: 0.28,
            borderWidth: 2.5,
            pointRadius: 3,
          });
        }
      });
    }

    state.charts["e1"] = new Chart(canvas, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          title: {
            display: true,
            text: state.previewMode
              ? "Đường cong Validation Accuracy qua các Epoch (Chế độ Preview Minh họa)"
              : "Đường cong Validation Accuracy qua các Epoch (Full Run)",
          },
        },
        scales: {
          y: { title: { display: true, text: "Validation Accuracy (%)" }, suggestedMin: 75, suggestedMax: 95 },
        },
      },
    });
  }

  function renderE2Chart(experiments) {
    const canvas = document.getElementById("e2-chart-canvas");
    if (!canvas || typeof Chart === "undefined") return;
    destroyChart("e2");

    const tokenizers = ["patch", "row", "cnn"];
    const scratchData = tokenizers.map((t) => {
      const exp = experiments.find((x) => x.backend === "scratch" && x.tokenizer === t);
      const m = exp ? getMetric(exp, "test_accuracy") : { value: null };
      return m.value ? Number((m.value * 100).toFixed(2)) : 0;
    });
    const torchData = tokenizers.map((t) => {
      const exp = experiments.find((x) => x.backend === "torch" && x.tokenizer === t);
      const m = exp ? getMetric(exp, "test_accuracy") : { value: null };
      return m.value ? Number((m.value * 100).toFixed(2)) : 0;
    });

    state.charts["e2"] = new Chart(canvas, {
      type: "bar",
      data: {
        labels: ["Patch (49 tokens)", "Row (28 tokens)", "CNN-Stem (49 tokens)"],
        datasets: [
          {
            label: "ScratchMSA (Tự cài đặt)",
            data: scratchData,
            backgroundColor: "rgba(11, 87, 208, 0.82)",
            borderRadius: 6,
          },
          {
            label: "TorchMSA (nn.MultiheadAttention)",
            data: torchData,
            backgroundColor: "rgba(13, 148, 136, 0.82)",
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          title: {
            display: true,
            text: "Đối chiếu Test Accuracy (%) giữa ScratchMSA và TorchMSA trên 3 chiến lược Tokenization",
          },
        },
        scales: {
          y: { beginAtZero: false, suggestedMin: 80, suggestedMax: 94, title: { display: true, text: "Test Accuracy (%)" } },
        },
      },
    });
  }

  function renderE3Chart(experiments, mlpBase, cnnBase) {
    const canvas = document.getElementById("e3-chart-canvas");
    if (!canvas || typeof Chart === "undefined") return;
    destroyChart("e3");

    const allModels = [];
    if (mlpBase) allModels.push({ label: "E1 MLP Baseline", exp: mlpBase, color: "#64748b" });
    if (cnnBase) allModels.push({ label: "E1 CNN Baseline", exp: cnnBase, color: "#059669" });
    experiments.forEach((e) => {
      allModels.push({
        label: e.id.toUpperCase(),
        exp: e,
        color: e.recurrent === "LSTM" ? "#0b57d0" : "#7c3aed",
      });
    });

    state.charts["e3"] = new Chart(canvas, {
      type: "bar",
      data: {
        labels: allModels.map((m) => m.label),
        datasets: [
          {
            label: "Test Accuracy (%)",
            data: allModels.map((m) => {
              const res = getMetric(m.exp, "test_accuracy");
              return res.value ? Number((res.value * 100).toFixed(2)) : 0;
            }),
            backgroundColor: allModels.map((m) => m.color),
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: { display: true, text: "So sánh Test Accuracy (%): Chuỗi LSTM/GRU (E3) vs Baseline không hồi quy (E1)" },
        },
        scales: {
          y: { beginAtZero: false, suggestedMin: 82, suggestedMax: 94, title: { display: true, text: "Test Accuracy (%)" } },
        },
      },
    });
  }

  /* ========================================================================
     ASSIGNMENT 1 PAGE (A1 — CNN vs Transformer on CIFAR-100)
     ======================================================================== */
  function renderA1Page() {
    if (!state.manifest || !state.manifest.assignments?.A1) return;
    const a1 = state.manifest.assignments.A1;

    renderA1FreezeInspector();

    const tbody = document.getElementById("a1-table-body");
    if (tbody) {
      const filtered = a1.experiments.filter((exp) => {
        const matchF = state.a1FilterFamily === "all" || exp.family === state.a1FilterFamily;
        const matchS = state.a1FilterStrategy === "all" || exp.strategy === state.a1FilterStrategy;
        return matchF && matchS;
      });

      tbody.innerHTML = filtered
        .map((exp) => {
          const totalP = exp.metrics?.parameters || exp.expected_params;
          const trainP = exp.metrics?.trainable_parameters || exp.expected_trainable;
          const ratio = ((trainP / totalP) * 100).toFixed(2);
          return `
            <tr>
              <td><code style="font-weight:700;">${exp.id}</code></td>
              <td><span class="badge ${exp.family === "resnet18" ? "badge-ready" : "badge-preview"}">${exp.family === "resnet18" ? "CNN (ResNet-18)" : "ViT (DeiT-Tiny)"}</span></td>
              <td>${exp.strategy_label}</td>
              <td class="num-col">${formatInt(trainP)} / ${formatInt(totalP)} <span style="font-size:0.74rem;color:var(--text-muted);">(${ratio}%)</span></td>
              <td class="num-col">${renderMetricCell(exp, "val_accuracy", formatPct)}</td>
              <td class="num-col">${renderMetricCell(exp, "test_accuracy", formatPct)}</td>
              <td class="num-col">${renderMetricCell(exp, "macro_f1", (v) => formatNum(v, 4))}</td>
              <td class="num-col">${renderMetricCell(exp, "latency_ms_per_image", (v) => formatNum(v, 2) + " ms")}</td>
              <td class="num-col">${renderMetricCell(exp, "peak_gpu_memory_mb", (v) => formatNum(v, 0) + " MB")}</td>
            </tr>
          `;
        })
        .join("");
    }

    renderA1Charts(a1.experiments);
    renderArtifactGallery("a1-artifact-gallery", "A1", a1.experiments, [
      { key: "curves", label: "Loss & Accuracy Curves" },
      { key: "confusion_matrix", label: "Confusion Matrix (100 lớp)" },
      { key: "failures", label: "Top-16 High-Confidence Failure Cases" },
    ]);
  }

  function renderA1FreezeInspector() {
    const container = document.getElementById("a1-freeze-diagram");
    if (!container) return;

    const fam = state.a1InspectorFamily;
    const strat = state.a1InspectorStrategy;

    const blocks =
      fam === "resnet18"
        ? [
            { name: "conv1 + bn1", sub: "7×7, 64, s=2 (BatchNorm)", active: strat === "scratch" || strat === "full" },
            { name: "layer1", sub: "2× BasicBlock (64 ch)", active: strat === "scratch" || strat === "full" },
            { name: "layer2", sub: "2× BasicBlock (128 ch)", active: strat === "scratch" || strat === "full" },
            { name: "layer3", sub: "2× BasicBlock (256 ch)", active: strat === "scratch" || strat === "full" },
            {
              name: "layer4",
              sub: "2× BasicBlock (512 ch)",
              active: strat === "scratch" || strat === "full" || strat === "partial",
            },
            { name: "fc (Head)", sub: "Linear(512 → 100)", active: true },
          ]
        : [
            {
              name: "patch_embed",
              sub: "16×16 Patch + [CLS] + Pos",
              active: strat === "scratch" || strat === "full",
            },
            {
              name: "blocks.0 .. 9",
              sub: "10× Transformer Block (d=192, h=3)",
              active: strat === "scratch" || strat === "full",
            },
            {
              name: "blocks.10 .. 11",
              sub: "2× Final Transformer Block",
              active: strat === "scratch" || strat === "full" || strat === "partial",
            },
            {
              name: "norm",
              sub: "Final LayerNorm(192)",
              active: strat === "scratch" || strat === "full" || strat === "partial",
            },
            { name: "head", sub: "Linear(192 → 100)", active: true },
          ];

    const lrInfo =
      strat === "scratch"
        ? "AdamW (lr = 3e-4, weight_decay = 0.05) cho toàn bộ mạng"
        : strat === "head_only"
        ? "AdamW (head lr = 1e-3, weight_decay = 0.01); toàn bộ backbone đóng băng & giữ module.eval()"
        : "AdamW (backbone lr = 1e-4, head lr = 1e-3, weight_decay = 0.01); module đóng băng giữ module.eval()";

    container.innerHTML = `
      <div class="arch-pipeline">
        ${blocks
          .map(
            (b, i) => `
          <div class="arch-node ${b.active ? "trainable" : "frozen"}">
            <span>${icon(b.active ? "unlock" : "lock")}${b.name}</span>
            <span class="arch-node-sub">${b.sub}</span>
            <span class="arch-node-sub" style="color:${b.active ? "var(--accent-emerald)" : "var(--text-muted)"};">
              ${b.active ? "requires_grad = True" : "Frozen + eval()"}
            </span>
          </div>
          ${i < blocks.length - 1 ? `<span class="arch-arrow">${icon("arrowRight")}</span>` : ""}
        `
          )
          .join("")}
      </div>
      <div style="font-size:0.83rem;color:var(--text-secondary);display:flex;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">
        <span><strong>Cấu hình Optimizer:</strong> <code>${lrInfo}</code></span>
        <span><strong>Kiểm chứng bất biến:</strong> <code>verify_freeze_invariant()</code> đảm bảo <code>rtol=0, atol=0</code> trên trọng số &amp; BatchNorm buffer đóng băng.</span>
      </div>
    `;
  }

  function renderA1Charts(experiments) {
    const accCanvas = document.getElementById("a1-acc-chart");
    const memCanvas = document.getElementById("a1-mem-chart");
    if (!accCanvas || typeof Chart === "undefined") return;
    destroyChart("a1_acc");
    destroyChart("a1_mem");

    const strategies = ["scratch", "head_only", "partial", "full"];
    const stratLabels = ["Scratch", "Head-Only (Freeze)", "Partial Freeze", "Full Fine-Tune"];

    const getSeries = (family, metricKey, scale = 1) =>
      strategies.map((s) => {
        const exp = experiments.find((x) => x.family === family && x.strategy === s);
        const m = exp ? getMetric(exp, metricKey) : { value: null };
        return m.value !== null ? Number((m.value * scale).toFixed(2)) : 0;
      });

    state.charts["a1_acc"] = new Chart(accCanvas, {
      type: "bar",
      data: {
        labels: stratLabels,
        datasets: [
          {
            label: "ResNet-18 (CNN)",
            data: getSeries("resnet18", "test_accuracy", 100),
            backgroundColor: "rgba(11, 87, 208, 0.85)",
            borderRadius: 6,
          },
          {
            label: "DeiT-Tiny (Vision Transformer)",
            data: getSeries("deit_tiny", "test_accuracy", 100),
            backgroundColor: "rgba(124, 58, 237, 0.85)",
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          title: { display: true, text: "Test Accuracy (%) theo Chiến lược Huấn luyện trên CIFAR-100" },
        },
        scales: { y: { beginAtZero: true, suggestedMax: 90, title: { display: true, text: "Test Accuracy (%)" } } },
      },
    });

    if (memCanvas) {
      state.charts["a1_mem"] = new Chart(memCanvas, {
        type: "bar",
        data: {
          labels: stratLabels,
          datasets: [
            {
              label: "ResNet-18 Peak VRAM (MB)",
              data: getSeries("resnet18", "peak_gpu_memory_mb", 1),
              backgroundColor: "rgba(13, 148, 136, 0.82)",
              borderRadius: 6,
            },
            {
              label: "DeiT-Tiny Peak VRAM (MB)",
              data: getSeries("deit_tiny", "peak_gpu_memory_mb", 1),
              backgroundColor: "rgba(217, 119, 6, 0.82)",
              borderRadius: 6,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "top" },
            title: { display: true, text: "Peak GPU Memory (MB) theo Chiến lược Đóng băng" },
          },
          scales: { y: { beginAtZero: true, title: { display: true, text: "Peak GPU Memory (MB)" } } },
        },
      });
    }
  }

  /* ========================================================================
     ASSIGNMENT 2 PAGE (A2 — DDPM Noise Schedule Ablation & ICML 2021 Paper)
     ======================================================================== */
  function renderA2Page() {
    if (!state.manifest || !state.manifest.assignments?.A2) return;
    const a2 = state.manifest.assignments.A2;

    // 1. Live Mathematical Plot of Linear vs Cosine Schedule (Exact equations!)
    renderA2ScheduleMathChart();

    // 2. Quantitative Comparison Table (Linear vs Cosine)
    const tbody = document.getElementById("a2-table-body");
    if (tbody) {
      tbody.innerHTML = a2.experiments
        .map((exp) => {
          const params = exp.metrics?.parameters || exp.expected_params;
          return `
            <tr>
              <td><code style="font-weight:700;">${exp.id}</code><div style="font-size:0.78rem;color:var(--text-muted);">${exp.name}</div></td>
              <td><code>${exp.schedule_type}</code></td>
              <td class="num-col">${formatInt(params)}</td>
              <td class="num-col">${renderMetricCell(exp, "fid", (v) => formatNum(v, 2) + " ↓")}</td>
              <td class="num-col">${renderMetricCell(exp, "inception_score_mean", (v) => formatNum(v, 2) + " ↑")}</td>
              <td class="num-col">${renderMetricCell(exp, "seconds_per_image", (v) => formatNum(v, 4) + " s")}</td>
              <td class="num-col">${renderMetricCell(exp, "peak_gpu_memory_mb", (v) => formatNum(v, 1) + " MB")}</td>
            </tr>
          `;
        })
        .join("");
    }

    renderArtifactGallery("a2-artifact-gallery", "A2", a2.experiments, [
      { key: "denoising_trace", label: "Reverse Denoising Trace (t = 1000 → 0)" },
      { key: "latest_preview", label: "EMA Generated Sample Grid (4×4)" },
    ]);
  }

  function renderA2ScheduleMathChart() {
    const canvas = document.getElementById("a2-schedule-chart");
    if (!canvas || typeof Chart === "undefined") return;
    destroyChart("a2_schedule");

    // Compute exact alpha_bar_t for T=1000 per Ho et al. (2020) and Nichol & Dhariwal (2021)
    const T = 1000;
    const steps = [];
    const linearAlphaBar = [];
    const cosineAlphaBar = [];

    let prodLinear = 1.0;
    const s = 0.008;
    const f0 = Math.pow(Math.cos(((0 / T + s) / (1 + s)) * (Math.PI / 2)), 2);

    for (let t = 0; t <= T; t++) {
      if (t > 0) {
        const betaLin = 1e-4 + ((t - 1) / (T - 1)) * (0.02 - 1e-4);
        prodLinear *= 1.0 - betaLin;
      }
      const ft = Math.pow(Math.cos(((t / T + s) / (1 + s)) * (Math.PI / 2)), 2);
      const alphaBarCos = Math.min(1.0, Math.max(1e-6, ft / f0));

      if (t % 25 === 0) {
        steps.push(`t=${t}`);
        linearAlphaBar.push(Number(prodLinear.toFixed(4)));
        cosineAlphaBar.push(Number(alphaBarCos.toFixed(4)));
      }
    }

    state.charts["a2_schedule"] = new Chart(canvas, {
      type: "line",
      data: {
        labels: steps,
        datasets: [
          {
            label: "Linear Schedule ᾱ_t (beta: 1e-4 → 0.02)",
            data: linearAlphaBar,
            borderColor: "#e11d48",
            backgroundColor: "rgba(225, 29, 72, 0.1)",
            fill: true,
            tension: 0.25,
            borderWidth: 2.5,
            pointRadius: 2,
          },
          {
            label: "Cosine Schedule ᾱ_t (squaredcos_cap_v2, s=0.008)",
            data: cosineAlphaBar,
            borderColor: "#0b57d0",
            backgroundColor: "rgba(11, 87, 208, 0.1)",
            fill: true,
            tension: 0.25,
            borderWidth: 2.5,
            pointRadius: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          title: {
            display: true,
            text: "Đồ thị toán học Hệ số Tín hiệu tích lũy ᾱ_t theo bước thời gian t ∈ [0, 1000]",
          },
        },
        scales: {
          y: {
            min: 0,
            max: 1.02,
            title: { display: true, text: "Cumulative Signal Weight ᾱ_t" },
          },
          x: {
            title: { display: true, text: "Diffusion Timestep (t)" },
          },
        },
      },
    });
  }

  /* ========================================================================
     SHARED ARTIFACT GALLERY & LIGHTBOX
     ======================================================================== */
  function renderArtifactGallery(containerId, assignment, experiments, plotTypes) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const cards = [];
    experiments.forEach((exp) => {
      plotTypes.forEach((pt) => {
        const imgUrl = exp.images ? exp.images[pt.key] : null;
        const expectedPath = `artifacts/${assignment}/full/${exp.id}/${pt.key === "latest_preview" ? "preview_step_020000.png" : pt.key + ".png"}`;
        if (imgUrl) {
          cards.push(`
            <div class="artifact-card">
              <div class="artifact-thumb" data-lightbox-src="${imgUrl}" data-lightbox-title="${assignment} • ${exp.id} — ${pt.label}">
                <img src="${imgUrl}" alt="${exp.id} ${pt.label}" loading="lazy" />
              </div>
              <div class="artifact-caption">
                <span><code>${exp.id}</code> • ${pt.label}</span>
                <span class="badge badge-completed">Full Artifact</span>
              </div>
            </div>
          `);
        } else {
          cards.push(`
            <div class="artifact-card">
              <div class="artifact-thumb" style="cursor:default;">
                <div class="artifact-placeholder">
                  ${icon("chart")}
                  <div><strong>${exp.id}</strong> — ${pt.label}</div>
                  <div style="margin-top:0.3rem;font-size:0.72rem;opacity:0.8;">Sẽ tự động hiển thị khi đồng bộ từ:<br/><code>${expectedPath}</code></div>
                </div>
              </div>
              <div class="artifact-caption">
                <span><code>${exp.id}</code> • ${pt.label}</span>
                <span class="badge badge-pending">Pending Full Run</span>
              </div>
            </div>
          `);
        }
      });
    });

    container.innerHTML = cards.join("");

    // Bind lightbox clicks
    container.querySelectorAll("[data-lightbox-src]").forEach((el) => {
      el.addEventListener("click", () => {
        openLightbox(el.getAttribute("data-lightbox-src"), el.getAttribute("data-lightbox-title"));
      });
    });
  }

  function openLightbox(src, title) {
    const dialog = document.getElementById("artifact-lightbox");
    const imageContainer = document.getElementById("lightbox-image-container");
    const titleEl = document.getElementById("lightbox-title");
    if (!dialog || !imageContainer || !src) return;
    const img = document.createElement("img");
    img.src = src;
    img.alt = title || "Chi tiết biểu đồ thực nghiệm";
    imageContainer.replaceChildren(img);
    if (titleEl) titleEl.textContent = title || "Chi tiết biểu đồ thực nghiệm";
    dialog.showModal();
  }

  function setupInteractiveFilters() {
    // E2 backend filter
    document.querySelectorAll("[data-e2-backend]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-e2-backend]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.e2FilterBackend = btn.getAttribute("data-e2-backend");
        renderExercisesPage();
      });
    });

    // E2 tokenizer filter
    document.querySelectorAll("[data-e2-tokenizer]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-e2-tokenizer]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.e2FilterTokenizer = btn.getAttribute("data-e2-tokenizer");
        renderExercisesPage();
      });
    });

    // A1 family filter
    document.querySelectorAll("[data-a1-family]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-a1-family]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.a1FilterFamily = btn.getAttribute("data-a1-family");
        renderA1Page();
      });
    });

    // A1 strategy filter
    document.querySelectorAll("[data-a1-strategy]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-a1-strategy]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.a1FilterStrategy = btn.getAttribute("data-a1-strategy");
        renderA1Page();
      });
    });

    // A1 Freeze Architecture Inspector selectors
    document.querySelectorAll("[data-inspector-family]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-inspector-family]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.a1InspectorFamily = btn.getAttribute("data-inspector-family");
        renderA1FreezeInspector();
      });
    });

    document.querySelectorAll("[data-inspector-strategy]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-inspector-strategy]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.a1InspectorStrategy = btn.getAttribute("data-inspector-strategy");
        renderA1FreezeInspector();
      });
    });

    // Lightbox close button
    const closeBtn = document.getElementById("lightbox-close-btn");
    const dialog = document.getElementById("artifact-lightbox");
    if (closeBtn && dialog) {
      closeBtn.addEventListener("click", () => dialog.close());
      dialog.addEventListener("click", (e) => {
        if (e.target === dialog) dialog.close();
      });
    }
  }

  function renderAllDynamicSections() {
    injectSharedMetadata();
    renderExercisesPage();
    renderA1Page();
    renderA2Page();
  }

  async function init() {
    setupHeaderControls();
    setupInteractiveFilters();

    state.siteConfig = await fetchJsonWithFallback("data/site_config.json", window.FALLBACK_SITE_CONFIG || null);
    state.manifest = await fetchJsonWithFallback(
      "data/experiments_manifest.json",
      window.FALLBACK_EXPERIMENTS_MANIFEST || null
    );

    renderAllDynamicSections();

    if (typeof renderMathInElement === "function") {
      renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "$", right: "$", display: false },
        ],
        throwOnError: false,
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

