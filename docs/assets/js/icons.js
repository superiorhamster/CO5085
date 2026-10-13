/** Minimal outline icon set for the CO5085 interface. */
(function () {
  "use strict";

  const paths = {
    app: '<path d="M8.8 11.4 15.2 8.2M8.8 12.6l6.4 3.2"/><circle cx="7" cy="12" r="2"/><circle cx="17" cy="7" r="2"/><circle cx="17" cy="17" r="2"/>',
    lock: '<rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
    unlock: '<rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 7.5-2"/>',
    moon: '<path d="M20 15.2A8.5 8.5 0 0 1 8.8 4 8.5 8.5 0 1 0 20 15.2Z"/>',
    sun: '<circle cx="12" cy="12" r="3.5"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    fileText: '<path d="M6 2.5h8l4 4V21H6Z"/><path d="M14 2.5V7h4M9 12h6M9 16h6"/>',
    externalLink: '<path d="M14 5h5v5M19 5l-8 8"/><path d="M18 13v6H5V6h6"/>',
    arrowRight: '<path d="M5 12h14M14 7l5 5-5 5"/>',
    eye: '<path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.5"/>',
    shieldCheck: '<path d="M12 3 5 6v5c0 4.6 2.7 7.8 7 10 4.3-2.2 7-5.4 7-10V6Z"/><path d="m9 12 2 2 4-4"/>',
    bookOpen: '<path d="M4 5.5h5a3 3 0 0 1 3 3V20a3 3 0 0 0-3-3H4ZM20 5.5h-5a3 3 0 0 0-3 3V20a3 3 0 0 1 3-3h5Z"/>',
    lightbulb: '<path d="M8.5 15.5A7 7 0 1 1 15.5 15.5c-1 .7-1.5 1.5-1.5 2.5h-4c0-1-.5-1.8-1.5-2.5Z"/><path d="M10 21h4"/>',
    settings: '<path d="M4 7h6M14 7h6M4 17h10M18 17h2"/><circle cx="12" cy="7" r="2"/><circle cx="16" cy="17" r="2"/>',
    microphone: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M6 11a6 6 0 0 0 12 0M12 17v4M9 21h6"/>',
    video: '<rect x="3" y="5" width="14" height="14" rx="2"/><path d="m17 10 4-2v8l-4-2Z"/>',
    link: '<path d="M10 13a5 5 0 0 0 7.5.5l2-2a5 5 0 0 0-7-7l-1.2 1.2"/><path d="M14 11a5 5 0 0 0-7.5-.5l-2 2a5 5 0 0 0 7 7l1.2-1.2"/>',
    save: '<path d="M5 3h11l3 3v15H5Z"/><path d="M8 3v6h7V3M8 21v-7h8v7"/>',
    flask: '<path d="M9 3h6M10 3v6l-5 8.5A2.3 2.3 0 0 0 7 21h10a2.3 2.3 0 0 0 2-3.5L14 9V3"/><path d="M7.5 15h9"/>',
    chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    x: '<path d="m6 6 12 12M18 6 6 18"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    circle: '<circle cx="12" cy="12" r="8"/>',
    github: '<path d="M15 21v-3.5c.1-1-.4-1.8-1-2.3 3.3-.4 6.8-1.6 6.8-7.3A5.7 5.7 0 0 0 19.3 4 5.3 5.3 0 0 0 19.1.2S17.9-.2 15 1.7a13.4 13.4 0 0 0-6 0C6.1-.2 4.9.2 4.9.2A5.3 5.3 0 0 0 4.7 4a5.7 5.7 0 0 0-1.5 3.9c0 5.7 3.5 6.9 6.8 7.3-.4.4-.8 1.1-.9 2.1-.8.4-2.8 1.1-4-1.2 0 0-.7-1.3-2.1-1.4 0 0-1.3 0-.1.8 0 0 .9.4 1.5 1.7 0 0 .8 2.6 4.6 1.8V21"/>',
  };

  function render(name, className = "ui-icon") {
    const path = paths[name];
    if (!path) return "";
    return `<svg class="${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${path}</svg>`;
  }

  function mount(root = document) {
    root.querySelectorAll("[data-icon]").forEach((slot) => {
      slot.innerHTML = render(slot.getAttribute("data-icon"));
    });
  }

  window.CO5085Icons = { render, mount };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => mount());
  } else {
    mount();
  }
})();
