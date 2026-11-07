import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cameraTarget,
  validDimension
} from '../src/differentiable_exact_learning/viewer/static/camera.mjs';

test('head motion inside the safe area leaves the camera still', () => {
  assert.deepEqual(cameraTarget([0, 0], [7, 2], 21, 11), {
    center: [0, 0],
    cut: false
  });
});
test('an approaching edge moves the head back inside the safe area', () => {
  const result = cameraTarget([0, 0], [9, 4], 21, 11);
  assert.deepEqual(result, {
    center: [4, 2],
    cut: false
  });
  assert.deepEqual(cameraTarget(result.center, [10, 4], 21, 11).center, result.center);
});
test('distant teleportation cuts directly to the destination', () => {
  assert.deepEqual(cameraTarget([0, 0], [10, 150], 21, 11, true), {
    center: [10, 150],
    cut: true
  });
});
test('nearby jumps and negative coordinates remain bounded', () => {
  assert.equal(cameraTarget([0, 0], [-4, -2], 21, 11, true).cut, false);
  assert.deepEqual(cameraTarget([0, 0], [-9, -4], 21, 11).center, [-4, -2]);
});
test('tiny windows retain the head and dimensions reject invalid values', () => {
  assert.deepEqual(cameraTarget([0, 0], [1, 1], 3, 3).center, [1, 1]);
  for (const v of [3, 21, 100, 101, '11']) assert.ok(validDimension(v));
  for (const v of [0, 2, 102, 1.5, 'x', '']) assert.ok(!validDimension(v));
});
