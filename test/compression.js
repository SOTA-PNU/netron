import assert from 'node:assert/strict';
import { prune, quantize } from '../source/compression.js';

const pruning = prune([0.1, 0.01, -0.4, 0.001], 0.02);
assert.deepEqual(pruning.values, [0.1, 0, -0.4, 0]);
assert.equal(pruning.zeroCount, 2);
assert.equal(pruning.sparsity, 0.5);

for (const bits of [16, 8, 4, 2]) {
    const quantization = quantize([0.1832, -0.0121, 0.4851, 0.0034], bits);
    assert.equal(quantization.bits, bits);
    assert.equal(quantization.precision, `INT${bits}`);
    assert.equal(quantization.values.length, 4);
    assert.equal(quantization.dequantized.length, 4);
    assert.equal(quantization.error.length, 4);
    assert.ok(quantization.scale > 0);
    assert.ok(Number.isInteger(quantization.zeroPoint));
    const qmin = -(2 ** (bits - 1));
    const qmax = (2 ** (bits - 1)) - 1;
    assert.equal(quantization.qmin, qmin);
    assert.equal(quantization.qmax, qmax);
    assert.ok(quantization.values.every((value) => value >= qmin && value <= qmax));
    assert.equal(quantization.weightReduction, 1 - (bits / 32));
}

assert.equal(quantize([1, 2, 3], 16).weightReduction, 0.5);
assert.equal(quantize([1, 2, 3], 8).weightReduction, 0.75);
assert.equal(quantize([1, 2, 3], 4).weightReduction, 0.875);
assert.equal(quantize([1, 2, 3], 2).weightReduction, 0.9375);

const zeros = quantize([0, 0, 0], 8);
assert.deepEqual(zeros.values, [0, 0, 0]);
assert.deepEqual(zeros.dequantized, [0, 0, 0]);

assert.throws(() => quantize([1, 2, 3], 32), /INT16, INT8, INT4, or INT2/);
assert.throws(() => quantize([1, 2, 3], 3), /INT16, INT8, INT4, or INT2/);

console.log('Compression Lab calculation tests passed.');
