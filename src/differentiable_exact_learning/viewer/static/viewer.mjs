import {
  cameraTarget,
  validDimension
} from './camera.mjs';

const boot = JSON.parse(document.getElementById('bootstrap').textContent);
const $ = id => document.getElementById(id);
const canvas = $('grid'),
  ctx = canvas.getContext('2d');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const colors = {
  text: '#dce4ed',
  muted: '#7c91aa',
  grid: '#2b3747',
  head: '#edd19b',
  X: '#7dd3bc',
  Y: '#b9a4ed',
  Z: '#f0a396',
  '0': '#90b9c4',
  '1': '#e3d5b9'
};
const speeds = [1, 2, 4, 8, 16, 32, 64, 128];
let cells = new Map(),
  head = [0, 0],
  runes = {},
  camera = [0, 0],
  target = [0, 0];
let columns = boot.width,
  rows = boot.height,
  tapeRow = 0,
  step = 0;
let session = null,
  queue = [],
  playing = false,
  terminal = false,
  fetching = null;
let epoch = 0,
  resetTimer, rate = 4,
  budget = 0,
  lastTime = 0,
  jumpUntil = 0,
  fadeStart = 0;
let valid = false,
  status = 'ready',
  inspecting = false,
  dirty = true;
const key = (x, y) => `${x},${y}`;
const blank = value => value === '' ? '□' : value;
const formatPosition = p => `(${p[0]}, ${p[1]})`;

$('controller-label').textContent = boot.label;
$('input').value = boot.input;
$('width').value = columns;
$('height').value = rows;
$('mode').textContent = boot.model ? 'Autonomous model execution' : 'Expert execution';
$('bookmarks').hidden = boot.environment !== 'teleporting';
if (boot.tasks.length) {
  $('task-label').hidden = false;
  for (const task of boot.tasks) {
    const option = document.createElement('option');
    option.value = option.textContent = task;
    $('task').append(option);
  }
  $('task').value = boot.task;
}

async function request(path, payload) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Viewer-Token': boot.token
    },
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error ||
    'The local viewer could not complete the request.');
  return data;
}

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
}

function controls() {
  $('play').textContent = playing ? 'Pause' : terminal ? 'Restart' : step ? 'Resume' : 'Start';
  $('play').disabled = !valid;
  $('step').disabled = !valid || playing || terminal;
  for (const r of 'XYZ') $('rune-' + r).disabled = playing || !valid || !runes[r];
}

function pause() {
  playing = false;
  budget = 0;
  $('status').textContent = terminal ? $('status').textContent : step ? 'Paused' : 'Ready';
  controls();
}

async function reset() {
  const generation = ++epoch;
  playing = false;
  fetching = null;
  queue = [];
  valid = false;
  terminal = false;
  step = 0;
  inspecting = false;
  budget = 0;
  jumpUntil = 0;
  $('jump').hidden = true;
  showError('');
  controls();
  $('status').textContent = 'Preparing preview…';
  try {
    const initial = await request('/api/reset', {
      input: $('input').value,
      task: $('task').value || null,
      session
    });
    if (generation !== epoch) return;
    session = initial.session;
    cells = new Map(initial.cells.map(([x, y, s]) => [key(x, y), s]));
    head = initial.head;
    runes = initial.runes;
    tapeRow = initial.tape_row;
    camera = [...head];
    target = [...head];
    valid = true;
    status = 'ready';
    $('status').textContent = 'Ready';
    $('action').textContent = 'Read · write · move';
    refresh();
    controls();
  } catch (error) {
    if (generation !== epoch) return;
    cells.clear();
    runes = {};
    head = [0, 0];
    camera = [0, 0];
    target = [0, 0];
    $('action').textContent = 'Read · write · move';
    $('status').textContent = 'Check input';
    showError(error.message);
    refresh();
    controls();
  }
}

function fillQueue(count) {
  if (fetching) return fetching;
  // A terminal action may still wait in the display queue after the server stops.
  // Render every remaining action before announcing the final state.
  if (terminal || !session || !valid || ['halted', 'limit', 'error'].includes(queue.at(-1)
    ?.status)) {
    return Promise.resolve();
  }
  const generation = epoch;
  fetching = request('/api/step', {
    session,
    count
  }).then(result => {
    if (generation !== epoch) return;
    queue.push(...result.events);
    if (!queue.length && !result.events.length && result.status !== 'running' && result
      .status !== 'ready') {
      terminal = true;
      pause();
      $('status').textContent = result.message;
    }
  }).catch(error => {
    if (generation !== epoch) return;
    pause();
    showError(`Connection interrupted: ${error.message} Reset to reconnect.`);
    valid = false;
    controls();
  }).finally(() => {
    if (generation === epoch) fetching = null;
  });
  return fetching;
}

function moveCamera(jump = false) {
  const next = cameraTarget(target, head, columns, rows, jump);
  target = next.center;
  if (next.cut || reducedMotion.matches) {
    camera = [...target];
    if (next.cut && !reducedMotion.matches) fadeStart = performance.now();
  }
  dirty = true;
}

function apply(event) {
  for (const [x, y, s] of event.patches) {
    if (s === '') cells.delete(key(x, y));
    else cells.set(key(x, y), s);
  }
  head = event.head;
  step = event.step;
  runes = event.runes;
  status = event.status;
  terminal = ['halted', 'limit', 'error'].includes(status);
  if (event.read !== undefined) {
    $('action').textContent = `${blank(event.read)} → ${blank(event.write)}   ·   ${event.motion}` +
      (event.state !== null ? `   ·   state ${event.state}` : '');
  }
  if (event.jump) {
    const {
      rune,
      available
    } = event.jump;
    $('jump').textContent = available ?
      `${rune}  ${formatPosition(event.from)} → ${formatPosition(event.head)}` :
      `${rune} has no bookmark · head stays in place`;
    $('jump').style.color = colors[rune];
    $('jump').hidden = false;
    jumpUntil = performance.now() + 900;
  }
  moveCamera(inspecting || Boolean(event.jump?.available));
  inspecting = false;
  if (terminal) {
    playing = false;
    budget = 0;
    $('status').textContent = event.message;
    if (status === 'error') showError(event.message);
  } else $('status').textContent = playing ? 'Running' : 'Paused';
  refresh();
  controls();
}

async function singleStep() {
  if (playing || terminal || !valid) return;
  const generation = epoch;
  if (!queue.length) await fillQueue(1);
  if (generation === epoch && queue.length && !playing) apply(queue.shift());
}

async function toggle() {
  if (!valid) return;
  if (playing) return pause();
  if (terminal) {
    await reset();
    if (!valid) return;
  }
  moveCamera(inspecting);
  inspecting = false;
  refresh();
  playing = true;
  budget = 1;
  lastTime = performance.now();
  $('status').textContent = 'Running';
  controls();
}

function refresh() {
  $('counter').textContent = `Step ${step.toLocaleString()}`;
  $('coordinates').textContent = `head x ${head[0]} · y ${head[1]}`;
  $('region').textContent = boot.environment === 'tm' ? 'TAPE' :
    boot.environment === 'teleporting' ? (head[1] === tapeRow ? 'TAPE' : 'PROGRAM') : 'GRID';
  if (inspecting) {
    $('region').textContent = `${inspecting} BOOKMARK · ${camera[1] === tapeRow ? 'TAPE' : 'PROGRAM'}`;
  }
  canvas.dataset.head = head.join(',');
  canvas.dataset.step = String(step);
  for (const rune of 'XYZ') {
    const info = runes[rune],
      button = $('rune-' + rune);
    button.querySelector('strong').textContent = info ? formatPosition(info.position) :
      'Not placed';
    button.querySelector('small').textContent = info ? `covers ${blank(info.under)}` : {
      X: 'Program position',
      Y: 'Instruction bookmark',
      Z: 'Tape position'
    } [rune];
    drawBookmark(button.querySelector('canvas'), info, rune);
  }
  dirty = true;
}

function drawBookmark(view, info, rune) {
  const c = view.getContext('2d'),
    w = view.width,
    h = view.height;
  c.clearRect(0, 0, w, h);
  if (!info) return;
  c.font = '15px ui-monospace, monospace';
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  for (let y = -1; y <= 1; y++)
    for (let x = -2; x <= 2; x++) {
      const symbol = cells.get(key(info.position[0] + x, info.position[1] + y));
      c.fillStyle = x === 0 && y === 0 ? colors[rune] : '#8294aa';
      c.fillText(symbol || '·', (x + 2.5) * w / 5, (y + 1.5) * h / 3);
    }
}

function resize() {
  const box = canvas.getBoundingClientRect(),
    ratio = Math.min(devicePixelRatio || 1, 3);
  canvas.width = Math.round(box.width * ratio);
  canvas.height = Math.round(box.height * ratio);
  dirty = true;
}
new ResizeObserver(resize).observe(canvas);

function draw(now) {
  const ratio = canvas.width / canvas.getBoundingClientRect().width;
  const w = canvas.width / ratio,
    h = canvas.height / ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const cell = Math.min((w - 34) / columns, (h - 30) / rows);
  const frameW = columns * cell,
    frameH = rows * cell;
  const left = (w - frameW) / 2,
    top = (h - frameH) / 2;
  const centerX = w / 2,
    centerY = h / 2;
  const px = x => centerX + (x - camera[0]) * cell;
  const py = y => centerY + (y - camera[1]) * cell;
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, frameW, frameH);
  ctx.clip();
  const fade = fadeStart ? Math.min(1, .3 + (now - fadeStart) / 180 * .7) : 1;
  ctx.globalAlpha = fade;
  if (boot.environment === 'tm' || boot.environment === 'teleporting') {
    ctx.fillStyle = '#20303a';
    ctx.fillRect(left, py(tapeRow) - cell / 2, frameW, cell);
  }
  const x0 = Math.floor(camera[0] - columns / 2),
    x1 = Math.ceil(camera[0] + columns / 2);
  const y0 = Math.floor(camera[1] - rows / 2),
    y1 = Math.ceil(camera[1] + rows / 2);
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = .6;
  ctx.beginPath();
  for (let x = x0; x <= x1; x++) {
    ctx.moveTo(px(x) - cell / 2, top);
    ctx.lineTo(px(x) - cell / 2, top + frameH);
  }
  for (let y = y0; y <= y1; y++) {
    ctx.moveTo(left, py(y) - cell / 2);
    ctx.lineTo(left + frameW, py(y) - cell / 2);
  }
  ctx.stroke();
  ctx.font =
    `${Math.max(7,Math.min(23,cell*.49))}px ui-monospace, SFMono-Regular, Consolas, monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) {
      const symbol = cells.get(key(x, y));
      if (!symbol) continue;
      if ('XYZ'.includes(symbol)) {
        ctx.fillStyle = colors[symbol] + '14';
        ctx.fillRect(px(x) - cell / 2 + 2, py(y) - cell / 2 + 2, cell - 4, cell - 4);
      }
      ctx.fillStyle = colors[symbol] || (symbol === '+' || symbol === 'x' ? '#c7ac82' : '#9caec4');
      ctx.fillText(symbol, px(x), py(y) + .5);
    }
  ctx.fillStyle = '#edd19b10';
  ctx.fillRect(px(head[0]) - cell / 2 + 1, py(head[1]) - cell / 2 + 1, cell - 2, cell - 2);
  ctx.strokeStyle = colors.head;
  ctx.lineWidth = 1.7;
  ctx.strokeRect(px(head[0]) - cell / 2 + 1.5, py(head[1]) - cell / 2 + 1.5, cell - 3, cell - 3);
  if (!cells.has(key(...head))) {
    ctx.fillStyle = '#edd19b';
    ctx.fillText('·', px(head[0]), py(head[1]));
  }
  ctx.restore();
  canvas.dataset.camera = camera.map(n => n.toFixed(2)).join(',');
  dirty = fade < 1;
}

function frame(now) {
  const dt = Math.min(100, now - (lastTime || now));
  lastTime = now;
  if (playing) {
    if (queue.length < Math.min(8, rate) && !fetching) fillQueue(Math.min(32, Math.max(1, Math.ceil(
      rate / 3))));
    budget = Math.min(budget + dt / 1000 * rate, Math.max(1, rate / 4));
    while (budget >= 1 && queue.length && playing) {
      apply(queue.shift());
      budget -= 1;
    }
  }
  if (jumpUntil && now > jumpUntil) {
    $('jump').hidden = true;
    jumpUntil = 0;
  }
  const distance = Math.max(...camera.map((v, i) => Math.abs(v - target[i])));
  if (distance > .005) {
    const blend = reducedMotion.matches ? 1 : 1 - Math.exp(-dt / 65);
    camera = camera.map((v, i) => v + (target[i] - v) * blend);
    dirty = true;
  } else if (distance) {
    camera = [...target];
    dirty = true;
  }
  if (dirty) draw(now);
  requestAnimationFrame(frame);
}

$('play').addEventListener('click', toggle);
$('step').addEventListener('click', singleStep);
$('reset').addEventListener('click', () => {
  clearTimeout(resetTimer);
  reset();
});
$('input').addEventListener('input', () => {
  pause();
  valid = false;
  controls();
  clearTimeout(resetTimer);
  // Invalidate an in-flight step request immediately, before the debounced reset.
  ++epoch;
  fetching = null;
  queue = [];
  resetTimer = setTimeout(reset, 160);
});
$('task').addEventListener('change', () => {
  clearTimeout(resetTimer);
  reset();
});
$('speed').addEventListener('input', () => {
  rate = speeds[Number($('speed').value)];
  $('speed-value').textContent = `${rate} steps/s`;
});
for (const id of ['width', 'height']) $(id).addEventListener('change', () => {
  if (!validDimension($(id).value)) {
    $(id).value = id === 'width' ? columns : rows;
    return;
  }
  columns = Number($('width').value);
  rows = Number($('height').value);
  if (!inspecting) moveCamera();
  dirty = true;
});
for (const rune of 'XYZ') $('rune-' + rune).addEventListener('click', () => {
  if (playing || !runes[rune]) return;
  camera = [...runes[rune].position];
  target = [...camera];
  inspecting = rune;
  refresh();
});
document.addEventListener('keydown', event => {
  if (event.target.closest('input,select,textarea,button') || event.ctrlKey || event.metaKey ||
    event.altKey) return;
  if (event.code === 'Space') {
    event.preventDefault();
    toggle();
  }
  if (event.code === 'ArrowRight') {
    event.preventDefault();
    singleStep();
  }
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) pause();
});
reducedMotion.addEventListener('change', () => {
  camera = [...target];
  dirty = true;
});
reset();
requestAnimationFrame(frame);
