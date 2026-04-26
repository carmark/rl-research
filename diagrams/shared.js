/* ===== TrainingFlowAnimator - Shared Animation Engine ===== */

/**
 * TrainingFlowAnimator: Drives the animated training‐flow diagrams.
 *
 * Usage:
 *   const animator = new TrainingFlowAnimator(svgElement, stages, options);
 *   animator.mount(controlsContainer);
 *
 * Each `stage` has:
 *   { id, label, category, duration, components[], particles[] }
 *
 * Each `particle` in a stage has:
 *   { pathId, category, count, stagger }
 */
class TrainingFlowAnimator {
  constructor(svg, stages, opts = {}) {
    this.svg = svg;
    this.stages = stages;
    this.opts = Object.assign({
      speed: 1,
      loop: true,
      particleRadius: 3,
    }, opts);

    this.currentStage = 0;
    this.playing = false;
    this.speed = this.opts.speed;
    this.particles = [];
    this._raf = null;
    this._stageTimer = null;
    this._lastTime = 0;
    this._elapsed = 0;
    this._controls = null;
  }

  /* ---- Public API ---- */

  mount(container) {
    this._controls = container;
    this._buildControls();
    this._updateStageDisplay();
    this._highlightStage(0);
  }

  play() {
    if (this.playing) return;
    this.playing = true;
    this._lastTime = performance.now();
    this._tick();
    this._updatePlayBtn();
  }

  pause() {
    this.playing = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    this._updatePlayBtn();
  }

  toggle() {
    this.playing ? this.pause() : this.play();
  }

  step() {
    this.pause();
    this._advanceStage();
  }

  reset() {
    this.pause();
    this.currentStage = 0;
    this._elapsed = 0;
    this._clearParticles();
    this._highlightStage(0);
    this._updateTimeline(0);
    this._updateStageDisplay();
  }

  setSpeed(s) {
    this.speed = s;
    if (this._speedLabel) this._speedLabel.textContent = s.toFixed(1) + 'x';
  }

  goToStage(idx) {
    if (idx < 0 || idx >= this.stages.length) return;
    this.currentStage = idx;
    this._elapsed = 0;
    this._clearParticles();
    this._highlightStage(idx);
    this._updateStageDisplay();
    this._spawnParticles(this.stages[idx]);
  }

  /* ---- Controls UI ---- */

  _buildControls() {
    const c = this._controls;
    c.classList.add('controls');
    c.innerHTML = '';

    // Play / Pause
    const playBtn = this._el('button', 'btn', '▶');
    playBtn.title = 'Play / Pause';
    playBtn.addEventListener('click', () => this.toggle());
    this._playBtn = playBtn;

    // Step
    const stepBtn = this._el('button', 'btn', '⏭');
    stepBtn.title = 'Next stage';
    stepBtn.addEventListener('click', () => this.step());

    // Reset
    const resetBtn = this._el('button', 'btn', '⏮');
    resetBtn.title = 'Reset';
    resetBtn.addEventListener('click', () => this.reset());

    // Speed
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'speed-slider';
    slider.min = '0.2';
    slider.max = '3';
    slider.step = '0.1';
    slider.value = String(this.speed);
    slider.addEventListener('input', () => {
      this.setSpeed(parseFloat(slider.value));
    });

    const speedLabel = this._el('span', 'speed-label', this.speed.toFixed(1) + 'x');
    this._speedLabel = speedLabel;

    // Stage badges
    const stageDiv = document.createElement('div');
    stageDiv.className = 'stage-indicators';
    this.stages.forEach((st, i) => {
      const badge = this._el('span', 'stage-badge ' + st.category, st.label);
      badge.dataset.idx = i;
      badge.style.cursor = 'pointer';
      badge.addEventListener('click', () => this.goToStage(i));
      stageDiv.appendChild(badge);
    });
    this._stageBadges = stageDiv;

    c.append(resetBtn, playBtn, stepBtn, slider, speedLabel, stageDiv);

    // Timeline
    const tl = document.createElement('div');
    tl.className = 'timeline';
    const tlFill = document.createElement('div');
    tlFill.className = 'timeline-fill';
    tlFill.style.width = '0%';
    tl.appendChild(tlFill);
    tl.addEventListener('click', (e) => {
      const rect = tl.getBoundingClientRect();
      const pct = (e.clientX - rect.left) / rect.width;
      const idx = Math.floor(pct * this.stages.length);
      this.goToStage(Math.min(idx, this.stages.length - 1));
    });
    this._tlFill = tlFill;

    c.parentNode.insertBefore(tl, c.nextSibling);
  }

  _el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  }

  _updatePlayBtn() {
    if (this._playBtn) {
      this._playBtn.textContent = this.playing ? '⏸' : '▶';
      this._playBtn.classList.toggle('active', this.playing);
    }
  }

  _updateStageDisplay() {
    if (!this._stageBadges) return;
    const badges = this._stageBadges.children;
    for (let i = 0; i < badges.length; i++) {
      badges[i].classList.toggle('active', i === this.currentStage);
    }
  }

  _updateTimeline(pct) {
    if (this._tlFill) this._tlFill.style.width = (pct * 100) + '%';
  }

  /* ---- Animation Loop ---- */

  _tick() {
    if (!this.playing) return;
    const now = performance.now();
    const dt = (now - this._lastTime) * this.speed;
    this._lastTime = now;
    this._elapsed += dt;

    const stage = this.stages[this.currentStage];
    const dur = stage.duration || 2000;

    // progress within stage
    const stagePct = Math.min(this._elapsed / dur, 1);

    // overall progress
    const overallPct = (this.currentStage + stagePct) / this.stages.length;
    this._updateTimeline(overallPct);

    // move particles
    this._moveParticles(stagePct);

    if (this._elapsed >= dur) {
      this._advanceStage();
    }

    this._raf = requestAnimationFrame(() => this._tick());
  }

  _advanceStage() {
    this._clearParticles();
    this.currentStage++;
    this._elapsed = 0;

    if (this.currentStage >= this.stages.length) {
      if (this.opts.loop) {
        this.currentStage = 0;
      } else {
        this.currentStage = this.stages.length - 1;
        this.pause();
        this._updateTimeline(1);
        return;
      }
    }

    this._highlightStage(this.currentStage);
    this._updateStageDisplay();
    this._spawnParticles(this.stages[this.currentStage]);
  }

  /* ---- Stage Highlighting ---- */

  _highlightStage(idx) {
    const stage = this.stages[idx];
    // Dim all boxes, then activate the relevant ones
    this.svg.querySelectorAll('.comp-box').forEach(b => {
      b.classList.remove('active');
      b.classList.add('dimmed');
    });
    this.svg.querySelectorAll('.flow-arrow').forEach(a => {
      a.classList.add('dimmed');
    });

    if (stage.components) {
      stage.components.forEach(id => {
        const el = this.svg.querySelector('#' + id);
        if (el) {
          el.classList.remove('dimmed');
          el.classList.add('active');
        }
      });
    }

    if (stage.arrows) {
      stage.arrows.forEach(id => {
        const el = this.svg.querySelector('#' + id);
        if (el) {
          el.classList.remove('dimmed');
        }
      });
    }
  }

  /* ---- Particle System ---- */

  _spawnParticles(stage) {
    if (!stage.particles) return;

    stage.particles.forEach(pDef => {
      const path = this.svg.querySelector('#' + pDef.pathId);
      if (!path) return;
      const pathLen = path.getTotalLength();
      const count = pDef.count || 3;
      const stagger = pDef.stagger || 0.15;

      for (let i = 0; i < count; i++) {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('r', this.opts.particleRadius);
        circle.classList.add('particle', pDef.category, 'active');
        this.svg.appendChild(circle);
        this.particles.push({
          el: circle,
          path: path,
          pathLen: pathLen,
          offset: i * stagger,   // delay as fraction
          category: pDef.category,
        });
      }
    });
  }

  _moveParticles(stagePct) {
    this.particles.forEach(p => {
      let t = stagePct - p.offset;
      if (t < 0) { p.el.style.opacity = 0; return; }
      if (t > 1) t = 1;
      p.el.style.opacity = 1 - t * 0.3;

      const pt = p.path.getPointAtLength(t * p.pathLen);
      p.el.setAttribute('cx', pt.x);
      p.el.setAttribute('cy', pt.y);
    });
  }

  _clearParticles() {
    this.particles.forEach(p => p.el.remove());
    this.particles = [];
  }
}

/* ---- Tooltip helper ---- */
class DiagramTooltip {
  constructor() {
    this.el = document.createElement('div');
    this.el.className = 'tooltip';
    document.body.appendChild(this.el);
  }

  show(target, html) {
    this.el.innerHTML = html;
    this.el.classList.add('visible');
    const r = target.getBoundingClientRect();
    this.el.style.left = (r.left + r.width / 2 - this.el.offsetWidth / 2) + 'px';
    this.el.style.top = (r.top - this.el.offsetHeight - 8 + window.scrollY) + 'px';
  }

  hide() {
    this.el.classList.remove('visible');
  }

  attachTo(selector, contentFn) {
    document.querySelectorAll(selector).forEach(el => {
      el.addEventListener('mouseenter', () => this.show(el, contentFn(el)));
      el.addEventListener('mouseleave', () => this.hide());
    });
  }
}

/* ---- SVG Arrow marker injector ---- */
function injectArrowMarkers(svg) {
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');

  const categories = [
    { id: 'arrowhead', color: '#8892a6' },
    { id: 'arrowhead-rollout', color: '#4a9eff' },
    { id: 'arrowhead-training', color: '#34d399' },
    { id: 'arrowhead-data', color: '#fb923c' },
    { id: 'arrowhead-io', color: '#a78bfa' },
  ];

  categories.forEach(c => {
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    marker.setAttribute('id', c.id);
    marker.setAttribute('viewBox', '0 0 10 7');
    marker.setAttribute('refX', '10');
    marker.setAttribute('refY', '3.5');
    marker.setAttribute('markerWidth', '8');
    marker.setAttribute('markerHeight', '6');
    marker.setAttribute('orient', 'auto-start-reverse');
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', '0 0, 10 3.5, 0 7');
    poly.setAttribute('fill', c.color);
    marker.appendChild(poly);
    defs.appendChild(marker);
  });

  svg.prepend(defs);
}

/* ---- Helper: build legend ---- */
function buildLegend(container) {
  const items = [
    { cls: 'rollout',  label: 'Rollout / Generation' },
    { cls: 'training', label: 'Training / Optimization' },
    { cls: 'data',     label: 'Data Processing' },
    { cls: 'io',       label: 'I/O & Communication' },
  ];
  const div = document.createElement('div');
  div.className = 'legend';
  items.forEach(it => {
    const item = document.createElement('span');
    item.className = 'legend-item';
    item.innerHTML = `<span class="legend-dot ${it.cls}"></span>${it.label}`;
    div.appendChild(item);
  });
  container.appendChild(div);
}
