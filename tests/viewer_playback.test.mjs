// Exercise the shipped player with a tiny in-memory DOM and a controlled clock.
// No browser, network server, or frontend dependencies needed.
import test from 'node:test';
import assert from 'node:assert/strict';

const context = new Proxy({}, {
  get: () => () => {}
});
class Element {
  constructor() {
    this.textContent = '';
    this.value = '';
    this.hidden = false;
    this.disabled = false;
    this.dataset = {};
    this.style = {};
    this.listeners = {};
    this.children = new Map();
    this.width = 500;
    this.height = 300;
  }
  addEventListener(name, fn) {
    this.listeners[name] = fn;
  }
  append() {}
  querySelector(name) {
    if (!this.children.has(name)) this.children.set(name, new Element());
    return this.children.get(name);
  }
  getContext() {
    return context;
  }
  getBoundingClientRect() {
    return {
      width: 500,
      height: 300
    };
  }
  async fire(name) {
    return this.listeners[name]?.();
  }
}
const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
const initial = () => ({
  session: 'session',
  cells: [
    [0, 0, '1']
  ],
  head: [0, 0],
  runes: {},
  tape_row: 0,
  status: 'ready'
});
const event = (n, final = false) => ({
  step: n,
  patches: [
    [n, 0, '1']
  ],
  head: [n, 0],
  runes: {},
  status: final ? 'halted' : 'running',
  message: final ? 'Halted' : 'Running',
  read: '1',
  write: '1',
  motion: final ? 'S' : 'R',
  state: n
});
const response = data => ({
  ok: true,
  json: async () => data
});
const flush = () => new Promise(resolve => setImmediate(resolve));
let frame, time = performance.now(),
  pending, calls = 0;

Object.assign(globalThis, {
  document: {
    getElementById: element,
    addEventListener() {}
  },
  matchMedia: () => ({
    matches: true,
    addEventListener() {}
  }),
  ResizeObserver: class {
    observe() {}
  },
  devicePixelRatio: 1,
  requestAnimationFrame: fn => {
    frame = fn;
  },
  fetch: async path => {
    if (path === '/api/reset') return response(initial());
    calls++;
    return new Promise(resolve => {
      pending = data => resolve(response(data));
    });
  }
});
element('bootstrap').textContent = JSON.stringify({
  width: 21,
  height: 9,
  input: '1',
  environment: 'tm',
  tasks: [],
  label: 'Test controller',
  token: 'test'
});

async function tick() {
  time += 16;
  frame(time);
  await flush();
}
await import('../src/differentiable_exact_learning/viewer/static/viewer.mjs');
await flush();

test('buffered halt renders every action; pause, single-step and reset preserve execution order',
  async () => {
    assert.equal(element('counter').textContent, 'Step 0');
    assert.equal(calls, 0, 'preview never runs a controller step');
    element('speed').value = '7';
    await element('speed').fire('input');
    await element('play').fire('click');
    await tick();
    pending({
      events: Array.from({
        length: 16
      }, (_, i) => event(i + 1, i === 15)),
      status: 'halted',
      message: 'Halted'
    });
    await flush();
    await tick();
    assert.equal(element('status').textContent, 'Running');
    for (let n = 0; n < 12; n++) await tick();
    assert.equal(element('counter').textContent, 'Step 16');
    assert.equal(element('action').textContent, '1 → 1   ·   S   ·   state 16');
    assert.equal(element('status').textContent, 'Halted');
    assert.equal(calls, 1, 'never prefetch beyond a queued terminal action');

    await element('reset').fire('click');
    await flush();
    await element('play').fire('click');
    await tick();
    await element('play').fire('click'); // Pause while a batch remains in flight.
    const stepping = element('step').fire('click');
    pending({
      events: [event(1), event(2), event(3)],
      status: 'running'
    });
    await stepping;
    assert.equal(element('counter').textContent, 'Step 1');
    for (let n = 0; n < 5; n++) await tick();
    assert.equal(element('counter').textContent, 'Step 1',
      'pause does not drain buffered actions');
    await element('step').fire('click');
    assert.equal(element('counter').textContent, 'Step 2');

    await element('reset').fire('click');
    await flush();
    const staleStep = element('step').fire('click');
    const staleResponse = pending;
    await element('reset').fire('click');
    await flush();
    staleResponse({
      events: [event(99)],
      status: 'running'
    });
    await staleStep;
    assert.equal(element('counter').textContent, 'Step 0',
      'reset discards stale asynchronous steps');
  });
