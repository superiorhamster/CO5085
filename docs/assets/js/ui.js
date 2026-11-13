/* Shared interaction primitives for the static coursework dashboards. */
(() => {
  'use strict';
  const later = fn => window.requestAnimationFrame ? window.requestAnimationFrame(fn) : fn();
  function tabs({names, button, panel, hash, aliases = {}, onActivate = () => {}}) {
    let selected = names[0];
    const resolve = () => {
      if (aliases[location.hash]) return aliases[location.hash];
      const canonical = names.find(n => hash(n) === location.hash);
      if (canonical) return canonical;
      let target;
      try { target = document.getElementById(decodeURIComponent(location.hash.slice(1))); } catch (_) { return null; }
      return target ? names.find(n => panel(n) === target || panel(n)?.contains?.(target)) : null;
    };
    function select(name, {update = true, focus = false} = {}) {
      if (!names.includes(name)) name = names[0];
      selected = name;
      names.forEach(n => {
        const b = button(n), p = panel(n), active = n === name;
        if (b) { b.setAttribute('aria-selected', String(active)); b.tabIndex = active ? 0 : -1; }
        if (p) p.hidden = !active;
      });
      if (update && location.hash !== hash(name)) history.pushState(null, '', hash(name));
      if (focus) button(name)?.focus({preventScroll: true});
      later(() => {
        if (update || focus) button(name)?.scrollIntoView?.({block: 'nearest', inline: 'nearest'});
        onActivate(name);
      });
    }
    const readLocation = () => {
      const name = resolve();
      select(name || names[0], {update: false});
      if (name && location.hash !== hash(name)) later(() => {
        const target = document.getElementById(location.hash.slice(1));
        target?.scrollIntoView?.({block: 'start'});
      });
    };
    names.forEach((n, i) => {
      const b = button(n); if (!b) return;
      b.addEventListener('click', () => select(n));
      b.addEventListener('keydown', event => {
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? names.length-1 :
          event.key === 'ArrowRight' ? (i+1)%names.length : event.key === 'ArrowLeft' ? (i+names.length-1)%names.length : null;
        if (next === null) return;
        event.preventDefault(); select(names[next], {focus: true});
      });
    });
    window.addEventListener('hashchange', readLocation);
    window.addEventListener('popstate', readLocation);
    readLocation();
    return {select, refresh: () => select(selected, {update: false}), get selected() { return selected; }};
  }
  let lightboxTrigger, lightboxBound = false;
  function bindLightbox() {
    const d = document.getElementById('artifact-lightbox');
    if (!d || lightboxBound) return;
    lightboxBound = true;
    document.getElementById('lightbox-close-btn')?.addEventListener('click', () => d.close());
    d.addEventListener('click', e => { if (e.target === d) d.close(); });
    d.addEventListener('close', () => { if (lightboxTrigger?.isConnected !== false) lightboxTrigger?.focus(); });
  }
  function openImage(src, title, trigger) {
    const d = document.getElementById('artifact-lightbox'), container = document.getElementById('lightbox-image-container');
    if (!src || !d || !container) return;
    bindLightbox(); lightboxTrigger = trigger || document.activeElement;
    const image = document.createElement('img'); image.src = src; image.alt = title || 'Chi tiết hình thực nghiệm';
    container.replaceChildren(image);
    const heading = document.getElementById('lightbox-title'); if (heading) heading.textContent = image.alt;
    d.showModal();
  }
  window.CourseworkUI = {tabs, bindLightbox, openImage};
})();
