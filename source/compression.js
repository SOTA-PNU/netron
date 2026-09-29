const compression = {};

compression.prune = (values, threshold) => {
    if (!Number.isFinite(threshold) || threshold < 0) {
        throw new Error('Pruning threshold must be a non-negative number.');
    }
    const input = Array.from(values, (value) => Number(value));
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
    if (input.length === 0) {
        return { values: [], scale: 1, zeroPoint: 0 };
    }
    if (!input.every((value) => Number.isFinite(value))) {
        throw new Error('Quantization values must be finite numbers.');
    }
    const minimum = Math.min(...input);
    const maximum = Math.max(...input);
    const scale = maximum === minimum ? 1 : (maximum - minimum) / (qmax - qmin);
    const zeroPoint = Math.max(qmin, Math.min(qmax, Math.round(qmin - minimum / scale)));
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
