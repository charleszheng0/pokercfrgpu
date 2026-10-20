import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createServer} from 'node:net';
import {get} from 'node:http';
import {chromium} from 'playwright';

const socket = createServer();
await new Promise(resolve => socket.listen(0, '127.0.0.1', resolve));
const port = socket.address().port;
await new Promise(resolve => socket.close(resolve));
const base = `http://127.0.0.1:${port}`;
const server = spawn('python3', ['-m', 'trainer.server', '--port', String(port), '--no-training'], {
  cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'],
});
let browser;
try {
  let ready = false;
  for (let attempt = 0; attempt < 100; attempt++) {
    ready = await new Promise(resolve => {
      const req = get(base + '/', response => { response.resume(); resolve(response.statusCode === 200); });
      req.on('error', () => resolve(false));
    });
    if (ready) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.ok(ready, 'Server did not start');
  browser = await chromium.launch({channel: 'chrome', headless: true});
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base);
  const river = page.waitForResponse(response => response.url().endsWith('/api/demo-analysis'));
  await page.locator('#try-solver-demo').click();
  const riverResult = await (await river).json();
  assert.ok(riverResult.analysis.action_evs.length > 0);
  await page.locator('#demo-action-values .demo-action-row').first().waitFor();
  const started = page.waitForResponse(response => response.url().endsWith('/api/showcase/start'));
  await page.locator('#demo-play-hands').click();
  let state = await (await started).json();
  assert.equal(await page.locator('#play-layout').isVisible(), true);
  assert.equal(await page.locator('#demo-mode').isVisible(), false);
  const raiseIndex = state.legal_actions.findIndex(action => action.kind === 'raise');
  assert.ok(raiseIndex >= 0, 'First scenario should offer a real raise choice');
  const raised = page.waitForResponse(response => response.url().endsWith('/api/action'));
  await page.locator('#actions .action-button').nth(raiseIndex).click();
  const raisedState = await (await raised).json();
  assert.ok(raisedState.history.some(event => event[2] === 'raise'));
  const riverAgain = page.waitForResponse(response => response.url().endsWith('/api/demo-analysis'));
  await page.locator('#try-solver-demo').click();
  await riverAgain;
  const restarted = page.waitForResponse(response => response.url().endsWith('/api/showcase/start'));
  await page.locator('#demo-play-hands').click();
  state = await (await restarted).json();
  const expectedBoards = [
    ['As', '7c', '2d', '9h', '3s'],
    ['9h', '8h', '2s', '3c', 'Ah'],
    ['Qh', '7d', '2s', '9c', '8d'],
  ];
  for (let index = 0; index < 3; index++) {
    assert.equal(state.showcase.index, index + 1);
    assert.equal(state.street, 0);
    assert.equal(state.solver_feedback.solver_api_status, 'unsupported');
    assert.match(await page.locator('#ev-box').innerText(), /No matching CFR solve/);
    assert.equal(await page.locator('#estimate-live').isVisible(), false);
    for (let moves = 0; !state.done && moves < 20; moves++) {
      const action = state.legal_actions.find(row => row.kind === 'check' || row.kind === 'call');
      assert.ok(action);
      const actionIndex = state.legal_actions.indexOf(action);
      const response = page.waitForResponse(item => item.url().endsWith('/api/action'));
      await page.locator('#actions .action-button').nth(actionIndex).click();
      state = await (await response).json();
    }
    assert.equal(state.done, true);
    assert.deepEqual(state.board, expectedBoards[index]);
    assert.equal(state.opponent_hole.length, 2);
    assert.equal(state.rewards[0] + state.rewards[1], 0);
    if (index < 2) {
      const response = page.waitForResponse(item => item.url().endsWith('/api/showcase/next'));
      await page.locator('#showcase-next').click();
      state = await (await response).json();
    }
  }
  assert.equal(await page.locator('#showcase-next').isVisible(), false);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({browser: 'Chrome headless', showcase_hands: 3,
    showdown_boards: expectedBoards, strategy_status: 'unsupported; no fabricated CFR'}));
} finally {
  if (browser) await browser.close();
  server.kill('SIGTERM');
}
