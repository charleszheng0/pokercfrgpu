/* Dependency-free DOM smoke for the Solver Lab hand matrix. Run with node. */
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

class Element {
  constructor() {
    this.children = [];
    this.style = {};
    this.classList = {add() {}, toggle() {}};
    this.value = '';
    this.textContent = '';
    this.disabled = false;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren() { this.children = []; }
  setAttribute() {}
}

const elements = new Map();
for (const id of [
  'river-board', 'river-hero', 'hero-matrix', 'opponent-matrix',
  'hero-range-summary', 'opponent-range-summary', 'hero-combos',
  'opponent-combos', 'hero-preset', 'opponent-preset',
  'hero-preset-note', 'opponent-preset-note', 'include-hero-hand',
  'hero-clear-range', 'opponent-clear-range', 'analysis-status',
]) elements.set(id, new Element());
elements.get('river-board').value = 'As Kh Qd 7c 2s';
elements.get('river-hero').value = 'Jc Tc';

const presets = [
  {id: 'river_demo_hero', label: 'Hero demo', description: 'assumption', combos: {'Jc Tc': 1, '9c 8c': 1, '4c 4d': 1}},
  {id: 'river_demo_opponent', label: 'Opponent demo', description: 'assumption', combos: {'Ac Ad': 0.5, 'Jd Jh': 1, '8h 8s': 0.5}},
  {id: 'overlap_opponent', label: 'Blocker test', description: 'assumption', combos: {'Jc 9d': 1, 'Ac Ad': 1}},
];
const context = {
  window: {},
  document: {
    getElementById: id => elements.get(id),
    createElement: () => new Element(),
  },
  fetch: async () => ({ok: true, json: async () => ({presets})}),
};
vm.createContext(context);
vm.runInContext(fs.readFileSync('trainer/static/ranges.js', 'utf8'), context);
const editor = context.window.RangeEditor;

(async () => {
  await editor.init();
  assert.strictEqual(elements.get('hero-matrix').children.length, 169);
  assert.strictEqual(elements.get('opponent-matrix').children.length, 169);
  const defaultRanges = editor.payloadRanges();
  assert.strictEqual(defaultRanges[0].hands.length, 3);
  assert.strictEqual(defaultRanges[1].hands.length, 3);
  const selectClass = label => elements.get('hero-matrix').children.find(cell => cell.textContent === label).onclick();
  selectClass('AKs');
  selectClass('AQs');
  assert.deepStrictEqual(Array.from(editor.selectedClasses('hero')), ['AKs', 'AQs']);
  elements.get('hero-combos').children[1].children[2].onclick(); // 50% for both selected classes
  assert(elements.get('hero-range-summary').textContent.includes('7 combos'));
  elements.get('hero-combos').children[1].children[0].onclick(); // clear both without changing other classes
  assert.strictEqual(editor.payloadRanges()[0].hands.length, 3);
  selectClass('AKs');
  assert.deepStrictEqual(Array.from(editor.selectedClasses('hero')), ['AQs']);
  elements.get('river-board').value = 'As Kh Qd 7c Ac';
  editor.refresh();
  assert.strictEqual(editor.payloadRanges()[1].hands.length, 2);
  elements.get('river-board').value = 'As Kh Qd 7c 2s';
  elements.get('opponent-preset').value = 'overlap_opponent';
  elements.get('opponent-preset').onchange();
  assert.strictEqual(editor.payloadRanges()[1].hands.length, 1);
  assert.deepStrictEqual(Array.from(editor.payloadRanges()[1].hands[0]), ['Ac', 'Ad']);
  assert(elements.get('opponent-range-summary').textContent.includes('1 combos'));

  // Exact suit-combo editing is exposed for one selected class and serialized
  // as concrete cards with its own weight, while multi-select remains available.
  elements.get('hero-clear-range').onclick();
  elements.get('include-hero-hand').onclick();
  selectClass('JTs'); // switch from the auto-selected hero hand to one class
  selectClass('99');
  const heroEditor = elements.get('hero-combos');
  const suitDetails = heroEditor.children[2];
  assert.strictEqual(suitDetails.open, true);
  assert(suitDetails.children[0].textContent.includes('Specific combos'));
  const exactRow = suitDetails.children.find(child => child.children?.[0]?.textContent === '9c 9d');
  assert(exactRow, 'pair class should expose concrete card combinations');
  exactRow.children[1].value = '37';
  exactRow.children[1].oninput();
  const exactRanges = editor.payloadRanges();
  const exactIndex = exactRanges[0].hands.findIndex(cards => Array.from(cards).join(' ') === '9c 9d');
  assert(exactIndex >= 0);
  assert.strictEqual(exactRanges[0].weights[exactIndex], 0.37);
  const seen = new Set();
  for (let row = 0; row < 13; row++) for (let col = 0; col < 13; col++) {
    const combos = editor.combosFor(editor.labelFor(row, col));
    assert.strictEqual(combos.length, row === col ? 6 : row < col ? 4 : 12);
    for (const combo of combos) { assert(!seen.has(combo)); seen.add(combo); }
  }
  assert.strictEqual(seen.size, 1326);
  const strategyHolder = new Element();
  editor.renderStrategyMatrix(strategyHolder, [{
    hand: ['Jc', 'Tc'], marginal_probability: 1,
    actions: [{kind: 'check', amount: 0, frequency: 0.25}],
  }], {kind: 'check', amount: 0});
  assert.strictEqual(strategyHolder.children.length, 169);
  assert(strategyHolder.children.some(cell => cell.textContent === 'JTs 25%'));
  editor.renderStrategyMatrix(strategyHolder, [
    {hand: ['Jc', 'Tc'], marginal_probability: 0.25,
     actions: [{kind: 'check', amount: 0, frequency: 0.2}]},
    {hand: ['Jd', 'Td'], marginal_probability: 0.75,
     actions: [{kind: 'check', amount: 0, frequency: 0.6}]},
  ], {kind: 'check', amount: 0});
  assert(strategyHolder.children.some(cell => cell.textContent === 'JTs 50%'));
  console.log('Range editor: 169 cells, 1326 unique combos, default payload and card removal passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
