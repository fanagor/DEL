// Compare a built static viewer with reference executions from export_grid_viewer.py --parity.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [directory, fixturePath] = process.argv.slice(2);
if (!directory || !fixturePath) {
    throw Error("Usage: node tests/check_grid_export.mjs VIEWER_DIRECTORY PARITY_JSON");
}

const { createExecution, inspectStep, applyStep, unpackWeights } =
    await import(pathToFileURL(resolve(directory, "engine.mjs")));
const catalog = JSON.parse(readFileSync(resolve(directory, "data/catalog.json")));
const fixtures = JSON.parse(readFileSync(fixturePath));
assert.equal(new Set(fixtures.map(f => f.run).filter(Boolean)).size, catalog.runs.length);
let count = 0;
for (const fixture of fixtures) {
    let expert = catalog.experts[fixture.task];
    let weights = null;
    if (fixture.run) {
        const run = catalog.runs.find(run => run.source === fixture.run);
        assert.ok(run, fixture.run);
        const bytes = readFileSync(resolve(directory, run.path));
        const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        weights = unpackWeights(expert, run, buffer);
        const table = catalog.programs?.[run.programs?.[fixture.program]];
        if (table) expert = { ...expert, programs: { ...expert.programs, [fixture.program]: table } };
    }
    const state = createExecution(expert, fixture.input, fixture.program, weights);
    for (const expected of fixture.steps) {
        const action = inspectStep(state);
        assert.ok(action);
        applyStep(state, action);
        const where = (fixture.run || fixture.task) + " at step " + state.count;
        assert.deepEqual([action.read, action.write, action.motion], [expected.read, expected.write, expected.motion], where);
        assert.deepEqual([state.x, state.y], expected.head, where);
        assert.equal(state.halted, expected.status === "halted", where);
        if (!weights) assert.equal(state.state, expected.state, where);
        expected.hiddenAfter?.forEach((v, i) => assert.ok(Math.abs(state.hidden[i] - v) < 2e-5, where));
        count++;
    }
    const memory = fixture.cells.map(([x, y, symbol]) => [[x, y].join(","), symbol]);
    assert.deepEqual([...state.cells].sort(), memory.sort());
}
console.log(`Matched ${count} transitions across ${fixtures.length} reference executions.`);
