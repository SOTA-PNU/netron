import assert from 'node:assert/strict';
import { prune, quantize } from '../source/compression.js';

const pruning = prune([0.1, 0.01, -0.4, 0.001], 0.02);
assert.deepEqual(pruning.values, [0.1, 0, -0.4, 0]);
assert.equal(pruning.zeroCount, 2);
assert.equal(pruning.sparsity, 0.5);

const quantization = quantize([0.1832, -0.0121, 0.4851, 0.0034]);
assert.equal(quantization.values.length, 4);
assert.equal(quantization.dequantized.length, 4);
assert.equal(quantization.error.length, 4);
assert.ok(quantization.scale > 0);
assert.ok(Number.isInteger(quantization.zeroPoint));
assert.ok(quantization.values.every((value) => value >= -128 && value <= 127));

const zeros = quantize([0, 0, 0]);
assert.deepEqual(zeros.values, [0, 0, 0]);
assert.deepEqual(zeros.dequantized, [0, 0, 0]);

console.log('Compression Lab calculation tests passed.');
