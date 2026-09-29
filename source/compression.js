const compression = {};

compression.prune = (values, threshold) => {
    if (!Number.isFinite(threshold) || threshold < 0) {
        throw new Error('Pruning threshold must be a non-negative number.');
    }
    const input = Array.from(values, (value) => Number(value));
    if (!input.every((value) => Number.isFinite(value))) {
        throw new Error('Pruning values must be finite numbers.');
    }
    const output = input.map((value) => Math.abs(value) < threshold ? 0 : value);
    const zeroCount = output.reduce((count, value) => count + (value === 0 ? 1 : 0), 0);
    return {
        values: output,
        zeroCount,
        sparsity: output.length === 0 ? 0 : zeroCount / output.length
    };
};

compression.quantize = (values, qmin = -128, qmax = 127) => {
    const input = Array.from(values, (value) => Number(value));
    if (!Number.isInteger(qmin) || !Number.isInteger(qmax) || qmin >= qmax) {
        throw new Error('Quantization range is invalid.');
    }
    if (input.length === 0) {
        return { values: [], dequantized: [], error: [], scale: 1, zeroPoint: 0, minimum: 0, maximum: 0 };
    }
    if (!input.every((value) => Number.isFinite(value))) {
        throw new Error('Quantization values must be finite numbers.');
    }
    const minimum = Math.min(...input);
    const maximum = Math.max(...input);
    let scale = (maximum - minimum) / (qmax - qmin);
    let zeroPoint = 0;
    if (scale === 0) {
        const magnitude = Math.max(Math.abs(minimum), Math.abs(maximum));
        scale = magnitude === 0 ? 1 : magnitude / Math.max(Math.abs(qmin), Math.abs(qmax));
        zeroPoint = 0;
    } else {
        zeroPoint = Math.round(qmin - minimum / scale);
        zeroPoint = Math.max(qmin, Math.min(qmax, zeroPoint));
    }
    const quantized = input.map((value) => Math.max(qmin, Math.min(qmax, Math.round(value / scale + zeroPoint))));
    const dequantized = quantized.map((value) => (value - zeroPoint) * scale);
    const error = input.map((value, index) => Math.abs(value - dequantized[index]));
    return {
        values: quantized,
        dequantized,
        error,
        scale,
        zeroPoint,
        minimum,
        maximum
    };
};

export const prune = compression.prune;
export const quantize = compression.quantize;
