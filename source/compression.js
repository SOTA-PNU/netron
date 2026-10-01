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

compression.quantize = (values, bits = 8) => {
    const supported = new Set([16, 8, 4, 2]);
    if (!supported.has(bits)) {
        throw new Error('Quantization precision must be INT16, INT8, INT4, or INT2.');
    }

    const input = Array.from(values, (value) => Number(value));
    if (!input.every((value) => Number.isFinite(value))) {
        throw new Error('Quantization values must be finite numbers.');
    }

    const qmax = (2 ** (bits - 1)) - 1;
    const qmin = -qmax;
    const weightReduction = 1 - (bits / 32);

    if (input.length === 0) {
        return {
            values: [],
            dequantized: [],
            error: [],
            scale: 1,
            zeroPoint: 0,
            minimum: 0,
            maximum: 0,
            bits,
            precision: `INT${bits}`,
            qmin,
            qmax,
            weightReduction
        };
    }

    const minimum = Math.min(...input);
    const maximum = Math.max(...input);
    const magnitude = Math.max(Math.abs(minimum), Math.abs(maximum));
    const scale = magnitude === 0 ? 1 : magnitude / qmax;
    const zeroPoint = 0;

    const quantized = input.map((value) =>
        Math.max(qmin, Math.min(qmax, Math.round(value / scale)))
    );
    const dequantized = quantized.map((value) => value * scale);
    const error = input.map((value, index) => Math.abs(value - dequantized[index]));

    return {
        values: quantized,
        dequantized,
        error,
        scale,
        zeroPoint,
        minimum,
        maximum,
        bits,
        precision: `INT${bits}`,
        qmin,
        qmax,
        weightReduction
    };
};

compression.installInputGuard = () => {
    const document = globalThis.document;
    if (!document || document.__compressionLabInputGuard) {
        return;
    }
    document.addEventListener('keydown', (event) => {
        const target = event.target;
        const editable = target && typeof target.closest === 'function' && target.closest('.compression-lab') &&
            (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);
        if (editable && (event.key === 'Backspace' || event.keyCode === 8)) {
            event.stopPropagation();
        }
    });
    document.__compressionLabInputGuard = true;
};

compression.installInputGuard();

export const prune = compression.prune;
export const quantize = compression.quantize;
