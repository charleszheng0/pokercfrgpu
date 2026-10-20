(() => {
  const $ = id => document.getElementById(id);
  const fields = [
    ['learning_rate', 'Learning rate', 'number', '0.0001'],
    ['batch_size', 'Batch size', 'number', '1'],
    ['hands_per_run', 'Self-play hands / run', 'number', '1'],
    ['checkpoint_every', 'Checkpoint every hands', 'number', '1'],
    ['evaluation_hands', 'Evaluation hands / opponent', 'number', '1'],
    ['interval_seconds', 'Seconds between runs', 'number', '1'],
    ['hidden_layers', 'Hidden layers, comma separated', 'text', ''],
    ['epsilon', 'DQN exploration ε', 'number', '0.01'],
    ['eta', 'NFSP anticipatory η', 'number', '0.01'],
    ['validation_fraction', 'Held-out supervised fraction', 'number', '0.01'],
    ['target_update_interval', 'Target sync updates', 'number', '1'],
    ['rl_capacity', 'DQN replay capacity', 'number', '1'],
    ['sl_capacity', 'Reservoir capacity', 'number', '1'],
    ['seed', 'Base seed', 'number', '1'],
    ['cfr_iterations', 'River cache CFR+ iterations', 'number', '1'],
  ];
  const integerFields = new Set(['batch_size', 'hands_per_run', 'checkpoint_every',
    'evaluation_hands', 'interval_seconds', 'target_update_interval',
    'rl_capacity', 'sl_capacity', 'seed', 'cfr_iterations']);
  const html = fields.map(([key, label, type, step]) =>
    `<label>${label}<input id="train-${key}" type="${type}" ${step ? `step="${step}"` : ''}></label>`).join('');
  $('training-fields').innerHTML = html;
  let loaded = false;
  function render(data) {
    const state = data.status || {};
    $('training-state').textContent = state.state || 'idle';
    $('training-state').title = state.error || '';
    $('training-start').disabled = ['running', 'waiting'].includes(state.state);
    $('training-stop').disabled = !['running', 'waiting'].includes(state.state);
    $('training-progress').textContent = state.progress
      ? `${state.progress.hands} hands · ${state.progress.elapsed_seconds.toFixed(1)} s · policy loss ${state.progress.policy_loss?.toFixed(4) ?? 'pending'} · held-out loss ${state.progress.validation_policy_loss?.toFixed(4) ?? 'pending'} · Q loss ${state.progress.q_loss?.toFixed(4) ?? 'pending'}`
      : state.error || (state.last_run ? `Latest completed: ${state.last_run}` : 'No run yet.');
    const blueprint = data.blueprint;
    $('blueprint-status').textContent = blueprint
      ? `Cached river demo: NashConv ${blueprint.nash_conv_chips.toFixed(4)} chips · ${blueprint.iterations} CFR+ iterations · ${blueprint.solve_seconds.toFixed(2)} s`
      : 'No promoted river cache yet.';
    if (!loaded) {
      fields.forEach(([key]) => {
        const value = data.config[key];
        $(`train-${key}`).value = Array.isArray(value) ? value.join(', ') : value;
      });
      loaded = true;
    }
    const history = [...(data.history || [])].reverse();
    const latest = history[0];
    const summary = $('training-summary'); summary.replaceChildren();
    const metrics = [
      ['Self-play', state.progress ? `${state.progress.hands} hands` : latest ? `${latest.training_hands} hands` : '—'],
      ['Vs heuristic', latest ? `${latest.evaluation.find(e => e.opponent === 'heuristic')?.mean_chips_per_hand.toFixed(2) ?? '—'} chips/hand` : '—'],
      ['Active version', data.active?.run_id ? data.active.run_id.replace(/^run-/, '').slice(0, 13) : 'Preset model'],
    ];
    metrics.forEach(([label, value]) => {
      const item = document.createElement('div'); item.className = 'metric';
      const name = document.createElement('span'); name.textContent = label;
      const number = document.createElement('strong'); number.textContent = value;
      item.append(name, number); summary.append(item);
    });
    const box = $('training-history'); box.replaceChildren();
    if (!history.length) box.textContent = 'No completed runs yet.';
    history.forEach(run => {
      const node = document.createElement('div'); node.className = 'training-run';
      const title = document.createElement('strong'); title.textContent = `${run.run_id} · ${run.promotion.replaceAll('_', ' ')}${data.active?.run_id === run.run_id ? ' · ACTIVE' : ''}`; node.append(title);
      const info = document.createElement('div'); info.className = 'training-metrics';
      const metrics = [`${run.training_hands} hands`, `${run.training_seconds.toFixed(1)} s`,
        ...run.evaluation.map(e => `${e.opponent}: ${e.mean_chips_per_hand.toFixed(2)} ± ${e.standard_error?.toFixed(2) ?? '—'} chips/hand`)];
      if (run.delta_95ci_independent) metrics.push(`Δ vs incumbent 95% CI [${run.delta_95ci_independent.map(x => x.toFixed(2)).join(', ')}] chips`);
      metrics.forEach(value => { const chip = document.createElement('span'); chip.textContent = value; info.append(chip); });
      node.append(info); box.append(node);
    });
    const curve = $('training-curve'); curve.replaceChildren();
    const points = data.current_run?.curves || history[0]?.curves || [];
    if (!points.length) {
      curve.textContent = 'Curves appear after the first checkpoint.';
    } else {
      const series = [
        ['Policy loss', 'policy_loss', '#72cfa9'],
        ['Held-out loss', 'validation_policy_loss', '#e4c37e'],
        ['Q loss', 'q_loss', '#8bb8ed'],
      ];
      const numbers = series.flatMap(([, key]) => points.map(p => p[key]).filter(Number.isFinite));
      if (!numbers.length) { curve.textContent = 'Losses pending'; return; }
      const min = Math.min(...numbers), max = Math.max(...numbers);
      const span = Math.max(max - min, .01), width = 720, height = 150;
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.setAttribute('preserveAspectRatio', 'none');
      svg.setAttribute('aria-hidden', 'true');
      for (let step = 0; step <= 2; step++) {
        const line = document.createElementNS(svg.namespaceURI, 'line');
        const y = 12 + step * 62;
        for (const [key, value] of Object.entries({x1: 10, x2: 710, y1: y, y2: y, stroke: '#38465b', 'stroke-width': 1})) line.setAttribute(key, value);
        svg.append(line);
      }
      for (const [, key, color] of series) {
        const valid = points.map((point, index) => [index, point[key]]).filter(([, value]) => Number.isFinite(value));
        if (!valid.length) continue;
        const polyline = document.createElementNS(svg.namespaceURI, 'polyline');
        const x = index => 10 + index * 700 / Math.max(1, points.length - 1);
        const y = value => 137 - (value - min) * 124 / span;
        polyline.setAttribute('points', valid.map(([index, value]) => `${x(index)},${y(value)}`).join(' '));
        polyline.setAttribute('fill', 'none'); polyline.setAttribute('stroke', color);
        polyline.setAttribute('stroke-width', '2.6'); polyline.setAttribute('stroke-linecap', 'round');
        polyline.setAttribute('stroke-linejoin', 'round'); svg.append(polyline);
        for (const [index, value] of valid) {
          const dot = document.createElementNS(svg.namespaceURI, 'circle');
          dot.setAttribute('cx', x(index)); dot.setAttribute('cy', y(value));
          dot.setAttribute('r', '3.5'); dot.setAttribute('fill', color); svg.append(dot);
        }
      }
      curve.append(svg);
      const legend = document.createElement('div'); legend.className = 'training-legend';
      series.forEach(([label, key, color]) => {
        if (!points.some(point => Number.isFinite(point[key]))) return;
        const item = document.createElement('span'); item.style.setProperty('--series-color', color);
        const last = [...points].reverse().find(point => Number.isFinite(point[key]));
        item.textContent = `${label} ${last[key].toFixed(3)}`; legend.append(item);
      });
      curve.append(legend);
    }
  }
  async function refresh() {
    try { const response = await fetch('/api/training/status'); render(await response.json()); }
    catch (error) { $('training-state').textContent = `Unavailable: ${error.message}`; }
  }
  async function command(path, payload = {}) {
    const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw Error(data.error || `HTTP ${response.status}`);
    render(data);
  }
  $('training-tab').onclick = () => {
    $('game-mode').hidden = true; $('analysis-mode').hidden = true; $('training-mode').hidden = false;
    ['play-tab','analysis-tab'].forEach(id => $(id).classList.remove('active'));
    $('training-tab').classList.add('active'); $('status').textContent = 'Offline training';
    refresh();
  };
  $('training-form').onsubmit = async event => {
    event.preventDefault();
    const values = {};
    fields.forEach(([key]) => {
      const raw = $(`train-${key}`).value.trim();
      values[key] = key === 'hidden_layers' ? raw.split(',').map(x => Number(x.trim()))
        : integerFields.has(key) ? Number.parseInt(raw, 10) : Number(raw);
    });
    try { await command('/api/training/config', values); $('training-save-status').textContent = 'Saved for next run'; }
    catch (error) { $('training-save-status').textContent = error.message; }
  };
  $('training-start').onclick = async () => {
    try { await command('/api/training/start'); } catch (error) { $('training-save-status').textContent = error.message; }
  };
  $('training-stop').onclick = async () => {
    try { await command('/api/training/stop'); $('training-save-status').textContent = 'Stop requested after current hand'; }
    catch (error) { $('training-save-status').textContent = error.message; }
  };
  refresh();
  setInterval(() => { if (!$('training-mode').hidden) refresh(); }, 3000);
})();
