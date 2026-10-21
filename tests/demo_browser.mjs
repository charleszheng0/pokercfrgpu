import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { get } from 'node:http';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

function signedEV(value, places = 3) {
  const rounded = Math.abs(value) < 0.5 * 10 ** -places ? 0 : value;
  return `${rounded > 0 ? '+' : ''}${rounded.toFixed(places)}`;
}

async function freePort() {
  const server = createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

const port = await freePort();
const base = `http://127.0.0.1:${port}`;
const server = spawn('python3', ['-m', 'tests.review_browser_server', String(port)], {
  cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'],
});
let serverErrors = '';
server.stderr.on('data', chunk => { serverErrors += chunk.toString(); });
let browser;
function serverReady(url) {
  return new Promise(resolve => {
    const request = get(url, response => { response.resume(); resolve(response.statusCode === 200); });
    request.on('error', () => resolve(false));
    request.setTimeout(1000, () => { request.destroy(); resolve(false); });
  });
}
try {
  let ready = false;
  for (let attempt = 0; attempt < 100; attempt++) {
    if (await serverReady(base + '/')) { ready = true; break; }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.ok(ready, `Trainer server did not start: ${serverErrors}`);
  browser = await chromium.launch({channel: 'chrome', headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  let solveRequests = 0;
  page.on('request', request => {
    if (request.url().endsWith('/api/demo-analysis')) solveRequests++;
  });
  await page.goto(base + '/', {waitUntil: 'networkidle'});
  assert.equal(await page.locator('#analysis-mode').isVisible(), true);
  assert.equal(await page.locator('#play-layout').isVisible(), false);
  assert.equal(await page.locator('#analysis-tab').innerText(), 'Scenario Lab');
  assert.equal(await page.locator('#hero-preset').isVisible(), true);
  assert.equal(await page.locator('#river-output').isVisible(), true);
  await page.locator('.mode-overflow > summary').click();
  await page.locator('#training-tab').click();
  assert.equal(await page.locator('#training-mode').isVisible(), true);
  assert.equal(await page.locator('#train-learning_rate').inputValue(), '0.001');
  await page.locator('.mode-overflow > summary').click();
  await page.locator('.mode-overflow > summary').click();
  await page.locator('#play-tab').click();
  assert.equal(await page.locator('#play-layout').isVisible(), true);
  const responsePromise = page.waitForResponse(response =>
    response.url().endsWith('/api/demo-analysis') && response.request().method() === 'POST');
  await page.getByRole('button', {name: 'Try Solver Demo'}).click();
  const response = await responsePromise;
  assert.equal(response.status(), 200, await response.text());
  const data = await response.json();
  await page.locator('#demo-actions button').first().waitFor();
  const report = data.analysis;
  assert.equal(solveRequests, 1);
  assert.equal(await page.locator('#demo-mode').isVisible(), true);
  assert.equal(await page.locator('#play-layout').isVisible(), false);
  assert.equal(await page.locator('#demo-board .card').count(), 5);
  assert.equal(await page.locator('#demo-hero-cards .card').count(), 2);
  assert.equal(await page.locator('#demo-opponent-cards .back').count(), 2);
  assert.equal(await page.locator('#demo-actions button').count(), report.action_evs.length);
  assert.equal(await page.locator('#demo-action-values .demo-action-row').count(), report.action_evs.length);
  assert.equal(await page.locator('#demo-range-matrix .matrix-cell').count(), 169);
  assert.equal(await page.locator('.setup').isVisible(), false);
  assert.equal(await page.locator('#demo-action-values .demo-action-row.best').count(), 1);
  const filledBar = await page.locator('#demo-action-values .frequency-fill').first().evaluate(node => node.getBoundingClientRect().width);
  assert.ok(filledBar > 10, 'Frequency bars must render with visible width');
  const diagnostics = await page.locator('#demo-diagnostics').innerText();
  assert.ok(diagnostics.includes(report.nash_conv.toFixed(4)));
  assert.ok(diagnostics.includes(report.iterations.toLocaleString()));
  assert.ok(diagnostics.includes(report.latency_ms.toFixed(1)));
  await page.locator('#demo-sample-cached').click();
  await page.locator('#demo-choice:not([hidden])').waitFor();
  assert.equal(await page.locator('#demo-actions button.selected').count(), 1);
  assert.equal(await page.locator('.demo-details').evaluate(node => node.open), false);
  await page.locator('.demo-details summary').click();
  const ranges = await page.locator('#demo-range-assumptions').innerText();
  assert.ok(ranges.includes('illustrative'));
  assert.ok(ranges.includes('50%'));
  await page.locator('.demo-details summary').click();
  for (let index = 0; index < report.action_evs.length; index++) {
    const text = await page.locator('#demo-action-values .demo-action-row').nth(index).innerText();
    assert.ok(text.includes(signedEV(report.action_evs[index].decision_ev)));
    assert.ok(text.includes(signedEV(report.action_evs[index].decision_ev_bb)));
    assert.ok(text.includes((report.frequencies[index].frequency * 100).toFixed(1)));
  }
  await page.locator('#demo-actions button').first().click();
  const firstChoice = await page.locator('#demo-choice').innerText();
  assert.ok(firstChoice.includes(report.action_evs[0].ev_loss.toFixed(3)));
  const lastIndex = report.action_evs.length - 1;
  await page.locator('#demo-actions button').nth(lastIndex).click();
  const lastChoice = await page.locator('#demo-choice').innerText();
  assert.ok(lastChoice.includes(signedEV(report.action_evs[lastIndex].decision_ev)));
  assert.ok(lastChoice.includes(report.action_evs[lastIndex].ev_loss_bb.toFixed(3)));
  assert.notEqual(firstChoice, lastChoice);
  assert.equal(solveRequests, 1, 'Changing the chosen action must not rerun CFR');
  assert.match(await page.locator('#demo-matrix-title').textContent(), /Bet to/i);
  if (process.env.SAVE_DEMO_SCREENSHOT === '1') {
    await mkdir('artifacts', {recursive: true});
    await page.screenshot({path: 'artifacts/solver_demo.png', fullPage: true});
  }
  await page.setViewportSize({width: 390, height: 844});
  assert.equal(await page.locator('#demo-mode').isVisible(), true);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), true);
  await page.setViewportSize({width: 1440, height: 1000});
  await page.getByRole('button', {name: 'Back to game'}).click();
  assert.equal(await page.locator('#play-layout').isVisible(), true);
  await page.getByRole('button', {name: 'New hand'}).click();
  await page.locator('#your-cards .card').first().waitFor();
  assert.equal(await page.locator('#your-cards .card').count(), 2);
  let state = await page.evaluate(async () => (await fetch('/api/state')).json());
  assert.equal(state.street, 0);
  const estimateResponse = page.waitForResponse(response => response.url().endsWith('/api/live-estimate'));
  await page.getByRole('button', {name: 'Estimate action EVs'}).click();
  const estimate = await (await estimateResponse).json();
  assert.equal(estimate.status, 'experimental_rollout');
  assert.equal(estimate.street, 'preflop');
  assert.equal(estimate.actions.length, state.legal_actions.length);
  await page.locator('#live-estimate-results .review-sub').first().waitFor();
  assert.ok((await page.locator('#live-estimate-results').innerText()).includes('not GTO'));
  assert.equal(await page.locator('#opponent-cards .back').count(), 2);
  await page.locator('#open-lab').click();
  assert.equal(await page.locator('#lab-street').inputValue(), 'preflop');
  assert.equal(await page.locator('#run-river-analysis').isDisabled(), true);
  await page.locator('#lab-quick-position').selectOption('river_example');
  await page.locator('#play-tab').click();
  for (let turn = 0; turn < 10 && !state.done; turn++) {
    assert.equal(state.solver_feedback.status, 'unavailable');
    await page.waitForFunction(reason => document.querySelector('#ev-box').title === reason, state.solver_feedback.reason);
    const index = state.legal_actions.findIndex(action => action.kind === 'call' || action.kind === 'check');
    assert.ok(index >= 0);
    const actionResponse = page.waitForResponse(response => response.url().endsWith('/api/action'));
    await page.locator('#actions button').nth(index).click();
    state = await (await actionResponse).json();
  }
  assert.equal(state.done, true);
  const riverReview = state.solver_reviews.find(review => review.street === 'river');
  assert.ok(riverReview);
  await page.locator('#solver-review-drawer summary').click();
  const reviewResponse = page.waitForResponse(response => response.url().endsWith('/api/review-decision'));
  await page.locator('#solver-review-list .review-item').filter({hasText: 'river ·'}).getByRole('button', {name: 'Analyze with illustrative ranges'}).click();
  const reviewData = await (await reviewResponse).json();
  assert.equal(reviewData.analysis.status, 'experimental_assumed_range');
  await page.getByText(`Your action EV: ${signedEV(reviewData.analysis.selected_action_decision_ev_bb, 2)} BB`, {exact: false}).waitFor();
  const reviewText = await page.locator('#solver-review-list').innerText();
  assert.ok(reviewText.includes(reviewData.analysis.nash_conv_chips.toFixed(3)));
  assert.ok(reviewText.includes(signedEV(reviewData.analysis.actions[0].decision_ev_chips, 2)));
  assert.ok((await page.locator('#solver-review-list .review-item').filter({hasText: 'river ·'}).locator('.study-note').first().getAttribute('title')).includes('not inferred from betting history'));
  const earlyReviews = {};
  for (const street of ['turn', 'flop']) {
    const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/review-decision'));
    const reviewItem = page.locator('#solver-review-list .review-item').filter({hasText: new RegExp(`^${street} ·`)});
    await reviewItem.getByRole('button', {name: 'Analyze with illustrative ranges'}).click();
    const responseData = await (await responsePromise).json();
    assert.equal(responseData.analysis.status, 'experimental_assumed_range');
    assert.ok(responseData.analysis.solver_scope_note.includes('no later betting'));
    await reviewItem.getByText('Your action EV:', {exact: false}).waitFor();
    const itemText = await reviewItem.innerText();
    assert.ok(itemText.includes(signedEV(responseData.analysis.actions[0].decision_ev_bb, 2)));
    assert.ok(itemText.includes(`NashConv ${responseData.analysis.nash_conv_chips.toFixed(3)}`));
    earlyReviews[street] = {nash_conv_chips: responseData.analysis.nash_conv_chips,
      latency_ms: responseData.analysis.latency_ms};
  }
  assert.ok((await page.locator('#solver-review-list').innerText()).includes('Unsupported: preflop'));
  await page.getByRole('button', {name: 'Scenario Lab', exact: true}).click();
  assert.equal(await page.locator('#lab-advanced').evaluate(node => node.open), false);
  assert.equal(await page.locator('#lab-quick-position').isVisible(), true);
  assert.equal(await page.locator('#lab-effective-stack').inputValue(), '100');
  assert.equal(await page.locator('#river-stacks').isVisible(), false);
  assert.ok((await page.locator('#lab-range-source').innerText()).includes('Assumed ranges'));
  await page.locator('#lab-advanced summary').click();
  await page.locator('#hero-matrix .matrix-cell').filter({hasText: /^AKs$/}).click();
  await page.locator('#hero-matrix .matrix-cell').filter({hasText: /^AQs$/}).click();
  assert.equal(await page.locator('#hero-matrix .matrix-cell[aria-pressed="true"]').count(), 2);
  await page.locator('#hero-combos .class-controls button').filter({hasText: '50%'}).click();
  assert.ok((await page.locator('#hero-range-summary').innerText()).includes('7 combos'));
  await page.locator('#hero-combos .class-controls button').filter({hasText: /^0%$/}).click();
  await page.locator('#hero-matrix .matrix-cell').filter({hasText: /^AKs$/}).click();
  await page.locator('#hero-combos .suit-details summary').click();
  assert.equal(await page.locator('#hero-combos .combo-row').count(), 4);
  await page.locator('#range-view-opponent').click();
  assert.equal(await page.locator('#opponent-range-card').isVisible(), true);
  assert.equal(await page.locator('#hero-range-card').isVisible(), false);
  await page.locator('#range-view-hero').click();
  await page.locator('#river-tree').selectOption('shark_2_0_inspired');
  const labResponse = page.waitForResponse(response => response.url().endsWith('/api/solve'));
  await page.locator('#run-river-analysis').click();
  const labData = await (await labResponse).json();
  assert.equal(labData.betting_tree, 'shark_2_0_inspired');
  assert.ok(labData.label.includes('Shark-inspired (2.0)'));
  assert.deepEqual(labData.action_evs.filter(row => row.action.kind === 'bet')
    .map(row => row.action.amount), [6, 13, 20, 100]);
  await page.locator('#river-output .lab-action-row').first().waitFor();
  assert.equal(await page.locator('#river-output .lab-action-row').count(), labData.action_evs.length);
  assert.equal(await page.locator('#river-output .strategy-cell').count(), 169);
  assert.equal(await page.locator('#river-output .lab-range-result').isVisible(), true);
  assert.ok((await page.locator('#river-output .strategy-mix-cell:not(.blocked)').count()) > 0);
  await page.locator('#river-output button.strategy-mix-cell').first().click();
  assert.ok((await page.locator('#river-output .lab-hand-detail').innerText()).includes('%'));
  const labRows = await page.locator('#river-output .lab-action-row').allInnerTexts();
  labData.action_evs.forEach((estimate, index) => {
    assert.ok(labRows[index].includes(signedEV(estimate.decision_ev_bb)));
    assert.ok(labRows[index].includes(signedEV(estimate.decision_ev)));
    assert.ok(labRows[index].includes((labData.frequencies[index].frequency * 100).toFixed(1)));
  });
  const lastLab = labData.action_evs.length - 1;
  await page.locator('#river-output .lab-action-row').nth(lastLab).click();
  assert.equal(await page.locator('#river-chosen-action').inputValue(), JSON.stringify(labData.action_evs[lastLab].action));
  assert.ok((await page.locator('#river-output .analysis-diagnostics').last().innerText()).includes(signedEV(labData.action_evs[lastLab].decision_ev_bb)));
  assert.equal(await page.locator('#river-output .lab-solver-info').evaluate(node => node.open), false);
  await page.locator('#lab-quick-position').selectOption('turn_example');
  assert.equal(await page.locator('#lab-street').inputValue(), 'turn');
  assert.equal(await page.locator('#hero-preset').inputValue(), 'turn_mechanics_hero');
  const turnLabResponse = page.waitForResponse(response => response.url().endsWith('/api/solve'));
  await page.locator('#run-river-analysis').click();
  const turnLabData = await (await turnLabResponse).json();
  assert.equal(turnLabData.status, 'restricted_turn_analysis');
  assert.equal(turnLabData.range_source, 'illustrative_preset');
  assert.equal(await page.locator('#river-output .strategy-mix-cell').count(), 169);
  await page.locator('#lab-quick-position').selectOption('fixed_demo');
  assert.equal(await page.locator('#river-hero').inputValue(), '8h 8s');
  assert.ok((await page.locator('#lab-range-source').innerText()).includes('Saved assumptions'));
  const fixedLabResponse = page.waitForResponse(response => response.url().endsWith('/api/solve'));
  await page.locator('#run-river-analysis').click();
  const fixedLabData = await (await fixedLabResponse).json();
  assert.equal(fixedLabData.status, 'restricted_river_analysis');
  assert.equal(fixedLabData.range_source, 'fixed_demo_inputs');
  assert.equal(fixedLabData.joint_deals, 4);
  assert.equal(await page.locator('#river-output .strategy-mix-cell').count(), 169);
  if (process.env.SAVE_DEMO_SCREENSHOT === '1') {
    await page.screenshot({path: 'artifacts/solver_lab.png', fullPage: true});
  }
  await page.locator('#lab-street').selectOption('flop');
  assert.equal(await page.locator('#run-river-analysis').isDisabled(), true);
  assert.ok((await page.locator('#lab-street-support').innerText()).includes('unavailable'));
  await page.locator('#lab-street').selectOption('preflop');
  assert.equal(await page.locator('#run-river-analysis').isDisabled(), true);
  assert.deepEqual(pageErrors, []);
  console.log(JSON.stringify({browser: 'Chrome headless', demo_nash_conv_chips: report.nash_conv,
    iterations: report.iterations, solve_latency_ms: report.latency_ms,
    action_count: report.action_evs.length, matrix_cells: 169,
    solve_requests_after_two_choices: solveRequests, game_resumed: true,
    gameplay_review_nash_conv_chips: reviewData.analysis.nash_conv_chips,
    gameplay_review_latency_ms: reviewData.analysis.latency_ms,
    early_street_reviews: earlyReviews,
    lab_nash_conv_chips: labData.nash_conv}));
} finally {
  if (browser) await browser.close();
  server.kill('SIGTERM');
}
