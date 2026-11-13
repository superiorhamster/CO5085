/**
 * CO5085 — Deep Learning & Applications in Computer Vision (HCMUT HK261)
 * Shared Client-Side Controller for GitHub Pages
 * Strictly adheres to plans/00_conventions.md:
 * - Official mode displays only verified `run_mode == "full"` artifacts.
 */

(function () {
  "use strict";

  const icon = (name, className = "ui-icon") =>
    window.CO5085Icons ? window.CO5085Icons.render(name, className) : "";

  const STORAGE_THEME_KEY = "co5085_theme";

  const state = {
    theme: localStorage.getItem(STORAGE_THEME_KEY) || "light",
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
    if (exp.has_full_metrics && Number.isFinite(exp.metrics?.[key])) {
      return { value: exp.metrics[key], source: "full" };
    }
    return { value: null, source: "pending" };
  }

  function renderMetricCell(exp, key, formatter) {
    const res = getMetric(exp, key);
    if (res.value === null) {
      return `<span class="badge badge-pending" title="Chưa có artifact full run tại artifacts/*/full/">Pending Full Run</span>`;
    }
    const formatted = formatter(res.value);
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

    const controls = document.getElementById("home-controls");
    if (controls && state.manifest) controls.textContent = `${state.manifest.assignments?.A1?.augmentation?.completed_count ?? 0} / 2`;

    // Global progress summary on index.html
    if (state.manifest) {
      const totalDone = state.manifest.total_completed_full_experiments || 0;
      const totalExp = state.manifest.total_expected_experiments || 23;
      document.querySelectorAll("[data-bind='artifact-progress']").forEach((el) => {
        el.textContent = `${totalDone} / ${totalExp}`;
      });
    }
  }

  /* ========================================================================
     EXERCISES PAGE (E1, E2, E3) RENDERING
     ======================================================================== */
  function renderExercisesPage() {
    if (!state.manifest || !state.manifest.assignments) return;
    if (window.Chart?.defaults) {
      const color = state.theme === 'dark' ? '#cbd5e1' : '#334155';
      window.Chart.defaults.color = color;
      window.Chart.defaults.borderColor = state.theme === 'dark' ? '#334155' : '#e2e8f0';
    }
    const e1 = state.manifest.assignments.E1;
    const e2 = state.manifest.assignments.E2;
    const e3 = state.manifest.assignments.E3;

    const summary = document.getElementById('exercises-summary');
    if (summary) summary.innerHTML = [e1,e2,e3].filter(Boolean).flatMap((a,i) => a.experiments.map(e =>
      `<tr><th scope="row">E${i+1} / ${e.id}</th><td>${renderMetricCell(e,'val_accuracy',formatPct)}</td><td>${renderMetricCell(e,'test_accuracy',formatPct)}</td><td>${renderMetricCell(e,'parameters',formatInt)}</td><td>${renderMetricCell(e,'latency_ms_per_image',formatNum)}</td></tr>`)).join('');
    const completion = document.getElementById('exercises-completion');
    if (completion) completion.textContent = `${[e1,e2,e3].filter(Boolean).flatMap(a=>a.experiments).filter(e=>e.has_full_metrics).length}/13 cấu hình đã xác minh`;
    // 1. E1 Table & Chart
    const e1Tbody = document.getElementById("e1-table-body");
    if (e1Tbody && e1) {
      e1Tbody.innerHTML = e1.experiments
        .map((exp) => {
          const params = exp.has_full_metrics ? exp.metrics?.parameters : null;
          const sizeMb = exp.has_full_metrics ? exp.metrics?.weight_size_mb : null;
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
          const params = exp.has_full_metrics ? exp.metrics?.parameters : null;
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
      renderE2Chart(filteredE2);
      renderArtifactGallery('e2-artifact-gallery','E2',filteredE2,[{key:'curves',label:'Learning curves'},{key:'confusion_matrix',label:'Confusion matrix'}]);
    }

    // 3. E3 Table (including E1 MLP/CNN provenance baselines) & Chart
    const e3Tbody = document.getElementById("e3-table-body");
    if (e3Tbody && e3 && e1) {
      const mlpBase = e1.experiments.find((x) => x.id === "mlp");
      const cnnBase = e1.experiments.find((x) => x.id === "cnn");
      const rowsHtml = [];

      [mlpBase, cnnBase].forEach((base) => {
        if (!base) return;
        const params = base.has_full_metrics ? base.metrics?.parameters : null;
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
        const params = exp.has_full_metrics ? exp.metrics?.parameters : null;
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
      renderArtifactGallery('e3-artifact-gallery','E3',e3.experiments,[{key:'curves',label:'Learning curves'},{key:'confusion_matrix',label:'Confusion matrix'}]);
    }
  }

  function renderE1Chart(experiments) {
    const canvas = document.getElementById("e1-chart-canvas");
    if (!canvas || typeof Chart === "undefined") return;
    destroyChart("e1");

    const hasData = experiments.some((e) => e.has_full_metrics && e.history?.some(h => Number.isFinite(Number(h.val_accuracy))));
    const emptyNotice = document.getElementById("e1-chart-empty");
    if (emptyNotice) emptyNotice.style.display = hasData ? "none" : "block";

    const labels = [];
    const colors = { softmax: "#64748b", mlp: "#0b57d0", cnn: "#059669" };

    const datasets = [];
    if (hasData) {
      experiments.forEach((exp) => {
        let curve = [];
        if (exp.has_full_metrics && exp.history && exp.history.length > 0) {
          curve = exp.history.map((h, i) => ({x: Number(h.epoch ?? i) + 1, y: Number(h.val_accuracy) * 100}));

        }
        if (curve.length > 0) {
          datasets.push({
            label: `${exp.name} (Val Acc %)`,
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
            text: "Validation accuracy theo epoch · full runs",
          },
        },
        scales: {
          x: {type: "linear", title: {display:true,text:"Epoch"}, ticks: {precision:0}},
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
      return m.value === null ? null : Number((m.value * 100).toFixed(2));
    });
    const torchData = tokenizers.map((t) => {
      const exp = experiments.find((x) => x.backend === "torch" && x.tokenizer === t);
      const m = exp ? getMetric(exp, "test_accuracy") : { value: null };
      return m.value === null ? null : Number((m.value * 100).toFixed(2));
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
              return res.value === null ? null : Number((res.value * 100).toFixed(2));
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
     ASSIGNMENT 1 PAGE (A1 — ResNet-18 vs DeiT-Tiny on CIFAR-100)
     ======================================================================== */
  function renderA1Page() {
    if (window.renderCifarA1) window.renderCifarA1(state);
  }

  /* ========================================================================
     ASSIGNMENT 2 PAGE (A2 — DCGAN and DiffAugment)
     ======================================================================== */
  function renderA2Page() {
    if(window.renderAnimeA2)window.renderAnimeA2(state);
  }

  function renderArtifactGallery(containerId, assignment, experiments, plotTypes) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const cards = [];
    experiments.forEach((exp) => {
      plotTypes.forEach((pt) => {
        const imgUrl = exp.has_full_metrics && exp.images ? exp.images[pt.key] : null;
        const expectedPath = `artifacts/${assignment}/${assignment==='A2'?'dcgan_diffaugment_v1/':''}full/${exp.id}/${pt.key === "latest_preview" ? "preview_step_020000.png" : pt.key + ".png"}`;
        if (imgUrl) {
          cards.push(`
            <div class="artifact-card">
              <button type="button" class="artifact-thumb" data-lightbox-src="${imgUrl}" data-lightbox-title="${assignment} • ${exp.id} — ${pt.label}">
                <img src="${imgUrl}" alt="${exp.id} ${pt.label}" loading="lazy" />
              </button>
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
        openLightbox(el.getAttribute("data-lightbox-src"), el.getAttribute("data-lightbox-title"), el);
      });
    });
  }

  function openLightbox(src, title, trigger) {
    window.CourseworkUI.openImage(src,title,trigger);
  }

  let exerciseTabs;
  function setupExerciseTabs() {
    if (!document.getElementById('exercises-dashboard')) return;
    if (!exerciseTabs) exerciseTabs = window.CourseworkUI.tabs({
      names:['overview','e1','e2','e3'],
      button:n=>document.getElementById('exercise-tab-'+n),
      panel:n=>document.getElementById(n==='overview'?'exercises-overview':n+'-section'),
      hash:n=>n==='overview'?'#exercises-overview':'#'+n+'-section',
      onActivate:()=>Object.values(state.charts).forEach(c=>c.resize())
    });
    else exerciseTabs.refresh();
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

    window.CourseworkUI.bindLightbox();
  }

  function renderAllDynamicSections() {
    injectSharedMetadata();
    renderExercisesPage();
    setupExerciseTabs();
    renderA1Page();
    renderA2Page();
    document.querySelectorAll('[data-report-pdf]').forEach(link => {
      const report = state.manifest?.reports?.[link.dataset.reportPdf];
      const available = report?.available && !report.is_placeholder;
      link.setAttribute('aria-disabled', String(!available));
      if (available) {
        link.setAttribute('href', report.path);
        link.textContent = `Báo cáo PDF · ${report.pages} trang`;
      } else {
        link.removeAttribute('href');
        link.textContent = 'PDF: chưa có bản build được xác minh';
      }
    });
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

