import * as base from './base.js';
import { prune, quantize } from './compression.js';

const lab = {};

lab.maxValues = 100000;
lab.previewRows = 10;

lab.format = (value) => {
    if (!Number.isFinite(value)) {
        return String(value);
    }
    const absolute = Math.abs(value);
    if ((absolute !== 0 && absolute < 0.0001) || absolute >= 10000) {
        return value.toExponential(5);
    }
    return value.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
};

lab.flatten = (value, output) => {
    if (output.length > lab.maxValues) {
        return;
    }
    if (Array.isArray(value) || ArrayBuffer.isView(value)) {
        for (const item of value) {
            lab.flatten(item, output);
            if (output.length > lab.maxValues) {
                return;
            }
        }
    } else if (typeof value === 'number') {
        if (Number.isFinite(value)) {
            output.push(value);
        }
    } else if (typeof value === 'bigint') {
        const number = Number(value);
        if (Number.isSafeInteger(number)) {
            output.push(number);
        }
    }
};

lab.shape = (entry) => {
    const type = entry.value && entry.value.type ? entry.value.type : entry.initializer.type;
    const shape = type && type.shape && Array.isArray(type.shape.dimensions) ? type.shape.dimensions : [];
    return shape.length > 0 ? shape.map((dimension) => String(dimension)).join(' × ') : '?';
};

lab.weights = (node) => {
    const entries = [];
    for (const argument of Array.isArray(node.inputs) ? node.inputs : []) {
        if (!Array.isArray(argument.value)) {
            continue;
        }
        for (let index = 0; index < argument.value.length; index++) {
            const value = argument.value[index];
            if (value && value.initializer) {
                const name = argument.name || value.name || `weight ${entries.length}`;
                entries.push({ name, value, initializer: value.initializer, index });
            }
        }
    }
    return entries;
};

lab.load = async (entry) => {
    const initializer = entry.initializer;
    if (initializer.peek && !initializer.peek()) {
        await initializer.read();
    }
    const tensor = new base.Tensor(initializer);
    if (tensor.empty) {
        throw new Error('Tensor data is empty.');
    }
    const values = [];
    lab.flatten(tensor.value, values);
    if (values.length > lab.maxValues) {
        throw new Error(`This tensor contains more than ${lab.maxValues.toLocaleString()} values. Select a smaller weight tensor for the classroom preview.`);
    }
    if (values.length === 0) {
        throw new Error('No numeric weight values could be read from this tensor.');
    }
    return values;
};

lab.styles = (document) => {
    if (document.getElementById('compression-lab-style')) {
        return;
    }
    const style = document.createElement('style');
    style.id = 'compression-lab-style';
    style.textContent = `
.compression-lab { margin: 8px 16px 18px 16px; font-size: 12px; line-height: 1.45; }
.compression-lab-note { margin: 6px 0 10px 0; color: #666; }
.compression-lab-row { display: flex; align-items: center; gap: 6px; margin: 7px 0; flex-wrap: wrap; }
.compression-lab-row label { min-width: 76px; font-weight: 600; }
.compression-lab select, .compression-lab input { box-sizing: border-box; min-height: 28px; border: 1px solid #b8b8b8; border-radius: 4px; background: transparent; color: inherit; padding: 3px 6px; font: inherit; }
.compression-lab select { max-width: 100%; min-width: 180px; }
.compression-lab input[type='number'] { width: 110px; }
.compression-lab button { min-height: 28px; border: 1px solid #aaa; border-radius: 4px; background: #f4f4f4; color: #222; padding: 3px 9px; font: inherit; cursor: pointer; }
.compression-lab button:hover { background: #e8e8e8; }
.compression-lab button:disabled { opacity: 0.55; cursor: default; }
.compression-lab-tabs { display: flex; gap: 4px; margin: 10px 0 8px 0; }
.compression-lab-tabs button[aria-selected='true'] { font-weight: 700; border-color: #666; }
.compression-lab-summary { margin: 8px 0; padding: 7px 8px; border: 1px solid #d0d0d0; border-radius: 4px; }
.compression-lab-formula { margin: 7px 0; padding: 7px 8px; border-left: 3px solid #999; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: normal; overflow-wrap: anywhere; }
.compression-lab-table-wrap { overflow-x: auto; margin-top: 8px; }
.compression-lab table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
.compression-lab th, .compression-lab td { border-bottom: 1px solid #ddd; padding: 4px 6px; text-align: right; white-space: nowrap; }
.compression-lab th:first-child, .compression-lab td:first-child { text-align: left; }
.compression-lab-error { color: #b00020; margin-top: 6px; }
@media (prefers-color-scheme: dark) {
    .compression-lab-note { color: #aaa; }
    .compression-lab button { background: #555; color: #eee; border-color: #777; }
    .compression-lab button:hover { background: #666; }
    .compression-lab select, .compression-lab input { border-color: #777; }
    .compression-lab-summary, .compression-lab th, .compression-lab td { border-color: #666; }
}`;
    document.head.appendChild(style);
};

lab.table = (document, headers) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'compression-lab-table-wrap';
    const table = document.createElement('table');
    const head = document.createElement('thead');
    const row = document.createElement('tr');
    for (const header of headers) {
        const cell = document.createElement('th');
        cell.textContent = header;
        row.appendChild(cell);
    }
    head.appendChild(row);
    const body = document.createElement('tbody');
    table.appendChild(head);
    table.appendChild(body);
    wrapper.appendChild(table);
    return { wrapper, body };
};

lab.fillTable = (body, rows) => {
    body.replaceChildren();
    const document = body.ownerDocument;
    for (const values of rows) {
        const row = document.createElement('tr');
        for (const value of values) {
            const cell = document.createElement('td');
            cell.textContent = value;
            row.appendChild(cell);
        }
        body.appendChild(row);
    }
};

lab.attach = (sidebar) => {
    const node = sidebar._node;
    const entries = lab.weights(node);
    if (entries.length === 0 || sidebar.element.querySelector('.compression-lab')) {
        return;
    }

    const document = sidebar._host.document;
    lab.styles(document);
    sidebar.addSection('Compression Lab');

    const root = document.createElement('div');
    root.className = 'compression-lab';
    sidebar.element.appendChild(root);

    const note = document.createElement('div');
    note.className = 'compression-lab-note';
    note.textContent = 'Educational working copy only. The original model file is not overwritten. For class, select the final Dense layer and inspect its kernel weights.';
    root.appendChild(note);

    const tensorRow = document.createElement('div');
    tensorRow.className = 'compression-lab-row';
    const tensorLabel = document.createElement('label');
    tensorLabel.textContent = 'Weight tensor';
    const selector = document.createElement('select');
    selector.setAttribute('aria-label', 'Weight tensor');
    entries.forEach((entry, index) => {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `${entry.name} (${lab.shape(entry)})`;
        selector.appendChild(option);
    });
    const preferred = entries.findIndex((entry) => /kernel|weight/i.test(entry.name));
    selector.selectedIndex = preferred >= 0 ? preferred : 0;
    tensorRow.appendChild(tensorLabel);
    tensorRow.appendChild(selector);
    root.appendChild(tensorRow);

    const info = document.createElement('div');
    info.className = 'compression-lab-summary';
    info.textContent = 'Loading weights…';
    root.appendChild(info);

    const tabs = document.createElement('div');
    tabs.className = 'compression-lab-tabs';
    const pruningTab = document.createElement('button');
    pruningTab.type = 'button';
    pruningTab.textContent = 'Pruning';
    pruningTab.setAttribute('aria-selected', 'true');
    const quantizationTab = document.createElement('button');
    quantizationTab.type = 'button';
    quantizationTab.textContent = 'Quantization';
    quantizationTab.setAttribute('aria-selected', 'false');
    tabs.appendChild(pruningTab);
    tabs.appendChild(quantizationTab);
    root.appendChild(tabs);

    const pruningPanel = document.createElement('div');
    const quantizationPanel = document.createElement('div');
    quantizationPanel.hidden = true;
    root.appendChild(pruningPanel);
    root.appendChild(quantizationPanel);

    const thresholdRow = document.createElement('div');
    thresholdRow.className = 'compression-lab-row';
    const thresholdLabel = document.createElement('label');
    thresholdLabel.textContent = 'Threshold';
    const threshold = document.createElement('input');
    threshold.type = 'number';
    threshold.step = '0.001';
    threshold.min = '0';
    threshold.value = '0.02';
    threshold.setAttribute('aria-label', 'Pruning threshold');
    const applyPruning = document.createElement('button');
    applyPruning.type = 'button';
    applyPruning.textContent = 'Apply to working copy';
    const resetPruning = document.createElement('button');
    resetPruning.type = 'button';
    resetPruning.textContent = 'Reset';
    thresholdRow.appendChild(thresholdLabel);
    thresholdRow.appendChild(threshold);
    thresholdRow.appendChild(applyPruning);
    thresholdRow.appendChild(resetPruning);
    pruningPanel.appendChild(thresholdRow);

    const pruningSummary = document.createElement('div');
    pruningSummary.className = 'compression-lab-summary';
    pruningPanel.appendChild(pruningSummary);
    const pruningTable = lab.table(document, ['Index', 'Before', 'After']);
    pruningPanel.appendChild(pruningTable.wrapper);

    const quantizeRow = document.createElement('div');
    quantizeRow.className = 'compression-lab-row';
    const indexLabel = document.createElement('label');
    indexLabel.textContent = 'Weight index';
    const weightIndex = document.createElement('input');
    weightIndex.type = 'number';
    weightIndex.min = '0';
    weightIndex.step = '1';
    weightIndex.value = '0';
    weightIndex.setAttribute('aria-label', 'Weight index');
    const applyQuantization = document.createElement('button');
    applyQuantization.type = 'button';
    applyQuantization.textContent = 'Quantize working copy';
    const resetQuantization = document.createElement('button');
    resetQuantization.type = 'button';
    resetQuantization.textContent = 'Reset';
    quantizeRow.appendChild(indexLabel);
    quantizeRow.appendChild(weightIndex);
    quantizeRow.appendChild(applyQuantization);
    quantizeRow.appendChild(resetQuantization);
    quantizationPanel.appendChild(quantizeRow);

    const quantizationSummary = document.createElement('div');
    quantizationSummary.className = 'compression-lab-summary';
    quantizationPanel.appendChild(quantizationSummary);
    const formula = document.createElement('div');
    formula.className = 'compression-lab-formula';
    quantizationPanel.appendChild(formula);
    const quantizationTable = lab.table(document, ['Index', 'FP32', 'INT8', 'Dequantized', '|Error|']);
    quantizationPanel.appendChild(quantizationTable.wrapper);
    const quantizationNote = document.createElement('div');
    quantizationNote.className = 'compression-lab-note';
    quantizationNote.textContent = 'This is an educational affine INT8 preview. A real converter may use symmetric or per-channel quantization, so its internal integer weights can differ.';
    quantizationPanel.appendChild(quantizationNote);

    const error = document.createElement('div');
    error.className = 'compression-lab-error';
    error.setAttribute('role', 'status');
    root.appendChild(error);

    const state = {
        original: [],
        working: [],
        pruningBefore: [],
        quantization: null
    };

    const renderPruning = (before, after) => {
        const zeroBefore = before.reduce((count, value) => count + (value === 0 ? 1 : 0), 0);
        const zeroAfter = after.reduce((count, value) => count + (value === 0 ? 1 : 0), 0);
        const sparsityBefore = before.length === 0 ? 0 : zeroBefore / before.length;
        const sparsityAfter = after.length === 0 ? 0 : zeroAfter / after.length;
        pruningSummary.textContent = `Sparsity: ${(sparsityBefore * 100).toFixed(1)}% → ${(sparsityAfter * 100).toFixed(1)}%  (${zeroAfter}/${after.length} zeros)`;
        const count = Math.min(lab.previewRows, before.length, after.length);
        const rows = [];
        for (let index = 0; index < count; index++) {
            rows.push([String(index), lab.format(before[index]), lab.format(after[index])]);
        }
        lab.fillTable(pruningTable.body, rows);
    };

    const renderQuantization = () => {
        const result = quantize(state.working);
        state.quantization = result;
        quantizationSummary.textContent = `min=${lab.format(result.minimum)}, max=${lab.format(result.maximum)}, scale=${lab.format(result.scale)}, zero point=${result.zeroPoint}`;
        weightIndex.max = String(Math.max(0, state.working.length - 1));
        let selected = Number.parseInt(weightIndex.value, 10);
        if (!Number.isInteger(selected)) {
            selected = 0;
        }
        selected = Math.max(0, Math.min(state.working.length - 1, selected));
        weightIndex.value = String(selected);
        const real = state.working[selected];
        const q = result.values[selected];
        const dequantized = result.dequantized[selected];
        formula.textContent = `q = round(${lab.format(real)} / ${lab.format(result.scale)} + ${result.zeroPoint}) = ${q};  dequantized = (${q} - ${result.zeroPoint}) × ${lab.format(result.scale)} = ${lab.format(dequantized)}`;
        const rows = [];
        const count = Math.min(lab.previewRows, state.working.length);
        for (let index = 0; index < count; index++) {
            rows.push([
                String(index),
                lab.format(state.working[index]),
                String(result.values[index]),
                lab.format(result.dequantized[index]),
                lab.format(result.error[index])
            ]);
        }
        lab.fillTable(quantizationTable.body, rows);
    };

    const setEnabled = (enabled) => {
        selector.disabled = !enabled;
        threshold.disabled = !enabled;
        applyPruning.disabled = !enabled;
        resetPruning.disabled = !enabled;
        weightIndex.disabled = !enabled;
        applyQuantization.disabled = !enabled;
        resetQuantization.disabled = !enabled;
    };

    const loadSelected = async () => {
        setEnabled(false);
        error.textContent = '';
        info.textContent = 'Loading weights…';
        try {
            const entry = entries[selector.selectedIndex];
            const values = await lab.load(entry);
            state.original = values.slice();
            state.working = values.slice();
            state.pruningBefore = values.slice();
            state.quantization = null;
            info.textContent = `Layer: ${node.name || (node.type ? node.type.name : '?')} · tensor: ${entry.name} · shape: ${lab.shape(entry)} · ${values.length.toLocaleString()} values`;
            renderPruning(state.original, state.working);
            renderQuantization();
            setEnabled(true);
        } catch (err) {
            error.textContent = err.message;
            info.textContent = 'Weight preview unavailable.';
        }
    };

    pruningTab.addEventListener('click', () => {
        pruningTab.setAttribute('aria-selected', 'true');
        quantizationTab.setAttribute('aria-selected', 'false');
        pruningPanel.hidden = false;
        quantizationPanel.hidden = true;
    });
    quantizationTab.addEventListener('click', () => {
        pruningTab.setAttribute('aria-selected', 'false');
        quantizationTab.setAttribute('aria-selected', 'true');
        pruningPanel.hidden = true;
        quantizationPanel.hidden = false;
        if (state.working.length > 0) {
            renderQuantization();
        }
    });
    selector.addEventListener('change', () => {
        loadSelected();
    });
    applyPruning.addEventListener('click', () => {
        error.textContent = '';
        try {
            const value = Number(threshold.value);
            state.pruningBefore = state.working.slice();
            const result = prune(state.working, value);
            state.working = result.values.slice();
            renderPruning(state.pruningBefore, state.working);
            renderQuantization();
        } catch (err) {
            error.textContent = err.message;
        }
    });
    resetPruning.addEventListener('click', () => {
        state.working = state.original.slice();
        state.pruningBefore = state.original.slice();
        renderPruning(state.original, state.working);
        renderQuantization();
        error.textContent = '';
    });
    weightIndex.addEventListener('change', () => {
        if (state.working.length > 0) {
            renderQuantization();
        }
    });
    applyQuantization.addEventListener('click', () => {
        error.textContent = '';
        try {
            const result = quantize(state.working);
            state.quantization = result;
            state.working = result.dequantized.slice();
            renderQuantization();
            renderPruning(state.working, state.working);
        } catch (err) {
            error.textContent = err.message;
        }
    });
    resetQuantization.addEventListener('click', () => {
        state.working = state.original.slice();
        state.quantization = null;
        renderQuantization();
        renderPruning(state.original, state.working);
        error.textContent = '';
    });

    loadSelected();
};

lab.patch = () => {
    const view = window.exports && window.exports.view;
    if (!view || !view.NodeSidebar) {
        return;
    }
    const prototype = view.NodeSidebar.prototype;
    if (prototype.__compressionLabPatched) {
        return;
    }
    const render = prototype.render;
    prototype.render = function() {
        const result = render.apply(this, arguments);
        try {
            lab.attach(this);
        } catch (error) {
            this.error(error, false);
        }
        return result;
    };
    prototype.__compressionLabPatched = true;
};

lab.patch();

export const patch = lab.patch;
