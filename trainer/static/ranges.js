/* Hand-matrix editor for Solver Lab. The server still accepts explicit JSON. */
(() => {
  const ranks = 'AKQJT98765432';
  const suits = 'cdhs';
  const $ = id => document.getElementById(id);
  const state = {
    hero: {weights: new Map(), selected: null, preset: null},
    opponent: {weights: new Map(), selected: null, preset: null},
  };
  let catalog = [];
  let onChange = () => {};

  function canonical(cards) {
    if (cards.length !== 2 || cards.some(card => !ranks.includes(card?.[0]) || !suits.includes(card?.[1]) || card.length !== 2) || cards[0] === cards[1]) {
      throw new Error('Enter two distinct cards such as Jc Tc.');
    }
    return [...cards].sort((a, b) => ranks.indexOf(a[0]) - ranks.indexOf(b[0]) || suits.indexOf(a[1]) - suits.indexOf(b[1]));
  }
  const comboKey = cards => canonical(cards).join(' ');
  function labelFor(row, col) {
    if (row === col) return ranks[row] + ranks[col];
    return ranks[Math.min(row, col)] + ranks[Math.max(row, col)] + (row < col ? 's' : 'o');
  }
  function combosFor(label) {
    const a = label[0], b = label[1];
    if (a === b) {
      const combos = [];
      for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) combos.push(comboKey([a + suits[i], a + suits[j]]));
      return combos;
    }
    if (label[2] === 's') return [...suits].map(suit => comboKey([a + suit, b + suit]));
    const combos = [];
    for (const first of suits) for (const second of suits) if (first !== second) combos.push(comboKey([a + first, b + second]));
    return combos;
  }
  function blockedCards(seat) {
    const blocked = new Set($('river-board').value.trim().split(/\s+/).filter(Boolean));
    if (seat === 'opponent') for (const card of $('river-hero').value.trim().split(/\s+/).filter(Boolean)) blocked.add(card);
    return blocked;
  }
  function isBlocked(combo, blockers) {
    return combo.split(' ').some(card => blockers.has(card));
  }
  function available(seat) {
    const blockers = blockedCards(seat);
    return [...state[seat].weights].filter(([combo, weight]) => weight > 0 && !isBlocked(combo, blockers));
  }
  function solverEntries(seat) {
    const boardOnly = blockedCards('hero');
    return [...state[seat].weights].filter(([combo, weight]) => weight > 0 && !isBlocked(combo, boardOnly));
  }
  function renderSummary(seat) {
    const active = available(seat);
    const sent = solverEntries(seat);
    const all = [...state[seat].weights].filter(([, weight]) => weight > 0).length;
    const blocked = all - active.length;
    const limit = sent.length > 8 ? ' · narrow to 8 or fewer for this CPU river solve' : '';
    $(seat + '-range-summary').textContent = `${sent.length} solver-range combos · ${active.length} possible with the entered hand${blocked ? ` · ${blocked} blocked in this view` : ''}${limit}`;
    $(seat + '-range-summary').classList.toggle('warning', sent.length > 8);
  }
  function renderMatrix(seat) {
    const holder = $(seat + '-matrix');
    holder.replaceChildren();
    const blockers = blockedCards(seat);
    for (let row = 0; row < 13; row++) for (let col = 0; col < 13; col++) {
      const label = labelFor(row, col);
      const combos = combosFor(label).filter(combo => !isBlocked(combo, blockers));
      const frequency = combos.length ? combos.reduce((total, combo) => total + (state[seat].weights.get(combo) || 0), 0) / combos.length : 0;
      const cell = document.createElement('button');
      cell.type = 'button'; cell.className = `matrix-cell ${row === col ? 'pair' : row < col ? 'suited' : 'offsuited'}`;
      if (state[seat].selected === label) cell.classList.add('selected');
      if (!combos.length) cell.classList.add('blocked');
      cell.disabled = !combos.length;
      cell.textContent = label;
      cell.setAttribute('aria-label', `${label}, ${Math.round(frequency * 100)} percent average, ${combos.length} available combinations`);
      if (combos.length) cell.style.background = `linear-gradient(0deg, #327454 ${Math.round(frequency * 100)}%, #293330 ${Math.round(frequency * 100)}%)`;
      cell.onclick = () => { state[seat].selected = label; renderMatrix(seat); renderEditor(seat); };
      holder.append(cell);
    }
    renderSummary(seat);
  }
  function markCustom(seat) {
    state[seat].preset = 'custom';
    $(seat + '-preset').value = 'custom';
    $(seat + '-preset-note').textContent = 'Custom frequencies entered here. They are assumptions, not inferred from betting history.';
    onChange();
  }
  function renderEditor(seat) {
    const holder = $(seat + '-combos');
    holder.replaceChildren();
    const label = state[seat].selected;
    if (!label) { holder.textContent = 'Select a hand class to edit its suit combinations.'; return; }
    const heading = document.createElement('strong'); heading.textContent = `${label} suit combinations`; holder.append(heading);
    const controls = document.createElement('div'); controls.className = 'class-controls';
    const field = document.createElement('label'); field.textContent = 'Set all available combos (%)';
    const input = document.createElement('input'); input.type = 'number'; input.min = '0'; input.max = '100'; input.step = '1'; input.value = '100';
    field.append(input);
    const apply = document.createElement('button'); apply.type = 'button'; apply.textContent = 'Apply to class';
    apply.onclick = () => {
      const value = Number(input.value);
      if (!Number.isFinite(value) || value < 0 || value > 100) return;
      const blockers = blockedCards(seat);
      for (const combo of combosFor(label)) if (!isBlocked(combo, blockers)) {
        if (value) state[seat].weights.set(combo, value / 100);
        else state[seat].weights.delete(combo);
      }
      markCustom(seat); renderMatrix(seat); renderEditor(seat);
    };
    controls.append(field, apply); holder.append(controls);
    const blockers = blockedCards(seat);
    for (const combo of combosFor(label)) {
      const row = document.createElement('div'); row.className = 'combo-row';
      const name = document.createElement('span'); name.textContent = combo.replace(' ', ' ');
      const slider = document.createElement('input'); slider.type = 'range'; slider.min = '0'; slider.max = '100'; slider.step = '1';
      const blocked = isBlocked(combo, blockers);
      slider.disabled = blocked;
      slider.value = String(Math.round((state[seat].weights.get(combo) || 0) * 100));
      const amount = document.createElement('output'); amount.textContent = blocked ? 'blocked' : `${slider.value}%`;
      slider.oninput = () => {
        const value = Number(slider.value);
        if (value) state[seat].weights.set(combo, value / 100);
        else state[seat].weights.delete(combo);
        amount.textContent = `${value}%`;
        markCustom(seat); renderMatrix(seat);
      };
      row.append(name, slider, amount); holder.append(row);
    }
  }
  function applyPreset(seat, preset) {
    const weights = new Map();
    if (preset.classes) {
      for (let row = 0; row < 13; row++) for (let col = 0; col < 13; col++) {
        const label = labelFor(row, col);
        const weight = preset.classes[label] || 0;
        if (weight) for (const combo of combosFor(label)) weights.set(combo, weight);
      }
    }
    if (preset.combos) for (const [combo, weight] of Object.entries(preset.combos)) {
      if (weight > 0) weights.set(comboKey(combo.split(' ')), weight);
    }
    state[seat].weights = weights;
    state[seat].preset = preset.id;
    state[seat].selected = null;
    $(seat + '-preset-note').textContent = preset.description;
    renderMatrix(seat); renderEditor(seat); onChange();
  }
  function usePreset(seat, id) {
    const preset = catalog.find(item => item.id === id);
    if (!preset || !state[seat]) throw new Error('Requested range preset is unavailable.');
    $(seat + '-preset').value = id;
    applyPreset(seat, preset);
  }
  function currentPreset(seat) { return state[seat]?.preset; }
  function refresh() {
    for (const seat of ['hero', 'opponent']) { renderMatrix(seat); renderEditor(seat); }
    onChange();
  }
  async function init(changed) {
    onChange = changed || (() => {});
    const response = await fetch('/api/range-presets');
    if (!response.ok) throw new Error('Range presets are unavailable.');
    catalog = (await response.json()).presets;
    for (const seat of ['hero', 'opponent']) {
      const select = $(seat + '-preset');
      const custom = document.createElement('option'); custom.value = 'custom'; custom.textContent = 'Custom edited range'; select.append(custom);
      for (const preset of catalog) {
        const option = document.createElement('option'); option.value = preset.id; option.textContent = preset.label; select.append(option);
      }
      select.onchange = () => {
        const preset = catalog.find(item => item.id === select.value);
        if (preset) applyPreset(seat, preset);
      };
      const initial = catalog.find(item => item.id === (seat === 'hero' ? 'river_demo_hero' : 'river_demo_opponent'));
      if (!initial) throw new Error('River example range is missing.');
      select.value = initial.id;
      applyPreset(seat, initial);
      $(seat + '-clear-range').onclick = () => {
        state[seat].weights.clear();
        markCustom(seat); renderMatrix(seat); renderEditor(seat);
      };
    }
    $('include-hero-hand').onclick = () => {
      try {
        const cards = canonical($('river-hero').value.trim().split(/\s+/));
        const combo = comboKey(cards);
        if (isBlocked(combo, blockedCards('hero'))) throw new Error('Hero hand overlaps the board.');
        state.hero.weights.set(combo, 1);
        state.hero.selected = cards[0][0] + cards[1][0] + (cards[0][0] === cards[1][0] ? '' : cards[0][1] === cards[1][1] ? 's' : 'o');
        markCustom('hero'); renderMatrix('hero'); renderEditor('hero');
      } catch (error) { $('analysis-status').textContent = error.message; $('analysis-status').className = 'analysis-error'; }
    };
    return true;
  }
  function payloadRanges() {
    const ranges = [];
    for (const seat of ['hero', 'opponent']) {
      const entries = solverEntries(seat);
      if (!entries.length) throw new Error(`${seat === 'hero' ? 'Hero' : 'Opponent'} range has no available hands after card removal.`);
      if (entries.length > 8) throw new Error(`${seat === 'hero' ? 'Hero' : 'Opponent'} range has ${entries.length} combos. Narrow it to 8 or fewer for this restricted CPU solver.`);
      ranges.push({hands: entries.map(([combo]) => combo.split(' ')), weights: entries.map(([, weight]) => weight)});
    }
    const hero = comboKey($('river-hero').value.trim().split(/\s+/));
    if (!ranges[0].hands.some(cards => comboKey(cards) === hero)) throw new Error('Include the entered hero hand in the hero range before solving.');
    return ranges;
  }
  function renderStrategyMatrix(holder, perHand, action, accent = '#327454') {
    holder.replaceChildren();
    const byCombo = new Map(perHand.map(row => [comboKey(row.hand), row]));
    for (let row = 0; row < 13; row++) for (let col = 0; col < 13; col++) {
      const label = labelFor(row, col);
      let mass = 0, weightedFrequency = 0, count = 0;
      for (const combo of combosFor(label)) {
        const strategy = byCombo.get(combo);
        if (!strategy) continue;
        const estimate = strategy.actions.find(item => item.kind === action.kind && item.amount === action.amount);
        if (!estimate) continue;
        mass += strategy.marginal_probability;
        weightedFrequency += strategy.marginal_probability * estimate.frequency;
        count++;
      }
      const frequency = mass ? weightedFrequency / mass : 0;
      const cell = document.createElement('span');
      cell.className = `matrix-cell strategy-cell ${count ? '' : 'blocked'}`;
      cell.textContent = count ? `${label} ${Math.round(frequency * 100)}%` : label;
      cell.setAttribute('aria-label', `${label}, ${count} supported combos, ${Math.round(frequency * 100)} percent ${action.kind}`);
      cell.title = count ? `${label} · ${Math.round(frequency * 100)}% ${action.kind} · ${count} supported combo${count === 1 ? '' : 's'}` : `${label} · outside supported range`;
      if (count) cell.style.background = `linear-gradient(0deg, ${accent} ${Math.round(frequency * 100)}%, #293330 ${Math.round(frequency * 100)}%)`;
      holder.append(cell);
    }
  }
  window.RangeEditor = {init, refresh, payloadRanges, combosFor, labelFor,
                        renderStrategyMatrix, usePreset, currentPreset};
})();
