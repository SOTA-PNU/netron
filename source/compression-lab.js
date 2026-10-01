import * as base from './base.js';
import { prune, quantize } from './compression.js';

const lab = {};

lab.maxValues = 100000;
lab.inferenceMaxValues = 500000;
lab.classLabels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
];

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

lab.flatten = (value, output, limit = lab.maxValues) => {
    if (output.length > limit) {
        return;
    }
    if (Array.isArray(value) || ArrayBuffer.isView(value)) {
        for (const item of value) {
            lab.flatten(item, output, limit);
            if (output.length > limit) {
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

lab.shapeArray = (entry) => {
    const type = entry.value && entry.value.type ? entry.value.type : entry.initializer.type;
    const dimensions = type && type.shape && Array.isArray(type.shape.dimensions) ? type.shape.dimensions : [];
    return dimensions.map((dimension) => Number(dimension));
};

lab.shape = (entry) => {
    const dimensions = lab.shapeArray(entry);
    return dimensions.length > 0 ? dimensions.map((dimension) => String(dimension)).join(' × ') : '?';
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

lab.load = async (entry, limit = lab.maxValues) => {
    const initializer = entry.initializer;
    if (initializer.peek && !initializer.peek()) {
        await initializer.read();
    }
    const tensor = new base.Tensor(initializer);
    if (tensor.empty) {
        throw new Error('Tensor data is empty.');
    }
    const values = [];
    lab.flatten(tensor.value, values, limit);
    if (values.length > limit) {
        throw new Error(`This tensor contains more than ${limit.toLocaleString()} values. Select a smaller model or tensor for the classroom preview.`);
    }
    if (values.length === 0) {
        throw new Error('No numeric weight values could be read from this tensor.');
    }
    return values;
};

lab.attribute = (node, name, defaultValue = null) => {
    const attributes = Array.isArray(node.attributes) ? node.attributes : [];
    const attribute = attributes.find((item) => item && item.name === name);
    return attribute ? attribute.value : defaultValue;
};

lab.activation = (node) => {
    if (Array.isArray(node.chain)) {
        for (const item of node.chain) {
            const type = item && item.type ? String(item.type.name || '').toLowerCase() : '';
            if (type === 'relu') {
                return 'relu';
            }
            if (type === 'softmax') {
                return 'softmax';
            }
            if (type === 'activation') {
                const value = lab.attribute(item, 'activation', 'linear');
                return typeof value === 'string' ? value.toLowerCase() : 'linear';
            }
        }
    }
    const value = lab.attribute(node, 'activation', 'linear');
    return typeof value === 'string' ? value.toLowerCase() : 'linear';
};

lab.applyActivation = (values, activation) => {
    switch (activation) {
        case '':
        case 'linear':
        case null:
        case undefined:
            return values;
        case 'relu':
            return values.map((value) => Math.max(0, value));
        case 'softmax': {
            const maximum = Math.max(...values);
            const exponentials = values.map((value) => Math.exp(value - maximum));
            const total = exponentials.reduce((sum, value) => sum + value, 0);
            return exponentials.map((value) => value / total);
        }
        default:
            throw new Error(`Top-3 preview does not support activation '${activation}'.`);
    }
};

lab.tensor = async (entry, cache, override) => {
    if (!entry) {
        return null;
    }
    let data = cache.get(entry.initializer);
    if (!data) {
        data = {
            values: await lab.load(entry, lab.inferenceMaxValues),
            shape: lab.shapeArray(entry)
        };
        cache.set(entry.initializer, data);
    }
    if (override && override.initializer === entry.initializer) {
        return {
            values: override.values,
            shape: data.shape
        };
    }
    return data;
};

lab.kernelAndBias = (node) => {
    const entries = lab.weights(node);
    const kernel = entries.find((entry) => /kernel|weight/i.test(entry.name)) || entries[0] || null;
    const bias = entries.find((entry) => /bias/i.test(entry.name)) || (entries.length > 1 ? entries[1] : null);
    return { kernel, bias };
};

lab.conv2d = async (data, node, cache, override) => {
    const { kernel: kernelEntry, bias: biasEntry } = lab.kernelAndBias(node);
    if (!kernelEntry) {
        throw new Error(`Top-3 preview could not find Conv2D kernel for '${node.name}'.`);
    }
    const kernel = await lab.tensor(kernelEntry, cache, override);
    const bias = await lab.tensor(biasEntry, cache, override);
    if (kernel.shape.length !== 4 || data.shape.length !== 3) {
        throw new Error('Top-3 preview expects NHWC Conv2D tensors.');
    }

    const [height, width, channels] = data.shape;
    const [kernelHeight, kernelWidth, kernelChannels, outputChannels] = kernel.shape;
    if (channels !== kernelChannels) {
        throw new Error(`Conv2D channel mismatch: input ${channels}, kernel ${kernelChannels}.`);
    }

    const strides = lab.attribute(node, 'strides', [1, 1]);
    const strideHeight = Number(strides[0] || 1);
    const strideWidth = Number(strides[1] || 1);
    const padding = String(lab.attribute(node, 'padding', 'valid')).toLowerCase();

    let outputHeight = 0;
    let outputWidth = 0;
    let padTop = 0;
    let padLeft = 0;

    if (padding === 'same') {
        outputHeight = Math.ceil(height / strideHeight);
        outputWidth = Math.ceil(width / strideWidth);
        const padHeight = Math.max((outputHeight - 1) * strideHeight + kernelHeight - height, 0);
        const padWidth = Math.max((outputWidth - 1) * strideWidth + kernelWidth - width, 0);
        padTop = Math.floor(padHeight / 2);
        padLeft = Math.floor(padWidth / 2);
    } else if (padding === 'valid') {
        outputHeight = Math.floor((height - kernelHeight) / strideHeight) + 1;
        outputWidth = Math.floor((width - kernelWidth) / strideWidth) + 1;
    } else {
        throw new Error(`Top-3 preview does not support Conv2D padding '${padding}'.`);
    }

    const output = new Array(outputHeight * outputWidth * outputChannels).fill(0);
    const biasValues = bias ? bias.values : null;

    for (let outputY = 0; outputY < outputHeight; outputY++) {
        for (let outputX = 0; outputX < outputWidth; outputX++) {
            for (let outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
                let sum = biasValues ? biasValues[outputChannel] : 0;
                for (let kernelY = 0; kernelY < kernelHeight; kernelY++) {
                    const inputY = outputY * strideHeight + kernelY - padTop;
                    if (inputY < 0 || inputY >= height) {
                        continue;
                    }
                    for (let kernelX = 0; kernelX < kernelWidth; kernelX++) {
                        const inputX = outputX * strideWidth + kernelX - padLeft;
                        if (inputX < 0 || inputX >= width) {
                            continue;
                        }
                        for (let inputChannel = 0; inputChannel < channels; inputChannel++) {
                            const inputIndex = ((inputY * width + inputX) * channels) + inputChannel;
                            const kernelIndex = (((kernelY * kernelWidth + kernelX) * channels + inputChannel) * outputChannels) + outputChannel;
                            sum += data.values[inputIndex] * kernel.values[kernelIndex];
                        }
                    }
                }
                const outputIndex = ((outputY * outputWidth + outputX) * outputChannels) + outputChannel;
                output[outputIndex] = sum;
            }
        }
    }

    return {
        values: lab.applyActivation(output, lab.activation(node)),
        shape: [outputHeight, outputWidth, outputChannels]
    };
};

lab.maxPooling2d = (data, node) => {
    if (data.shape.length !== 3) {
        throw new Error('Top-3 preview expects NHWC MaxPooling2D tensors.');
    }
    const [height, width, channels] = data.shape;
    const poolSize = lab.attribute(node, 'pool_size', [2, 2]);
    const strides = lab.attribute(node, 'strides', poolSize);
    const poolHeight = Number(poolSize[0] || 2);
    const poolWidth = Number(poolSize[1] || 2);
    const strideHeight = Number(strides[0] || poolHeight);
    const strideWidth = Number(strides[1] || poolWidth);
    const padding = String(lab.attribute(node, 'padding', 'valid')).toLowerCase();

    let outputHeight = 0;
    let outputWidth = 0;
    let padTop = 0;
    let padLeft = 0;

    if (padding === 'same') {
        outputHeight = Math.ceil(height / strideHeight);
        outputWidth = Math.ceil(width / strideWidth);
        const padHeight = Math.max((outputHeight - 1) * strideHeight + poolHeight - height, 0);
        const padWidth = Math.max((outputWidth - 1) * strideWidth + poolWidth - width, 0);
        padTop = Math.floor(padHeight / 2);
        padLeft = Math.floor(padWidth / 2);
    } else if (padding === 'valid') {
        outputHeight = Math.floor((height - poolHeight) / strideHeight) + 1;
        outputWidth = Math.floor((width - poolWidth) / strideWidth) + 1;
    } else {
        throw new Error(`Top-3 preview does not support MaxPooling2D padding '${padding}'.`);
    }

    const output = new Array(outputHeight * outputWidth * channels).fill(Number.NEGATIVE_INFINITY);

    for (let outputY = 0; outputY < outputHeight; outputY++) {
        for (let outputX = 0; outputX < outputWidth; outputX++) {
            for (let channel = 0; channel < channels; channel++) {
                let maximum = Number.NEGATIVE_INFINITY;
                for (let poolY = 0; poolY < poolHeight; poolY++) {
                    const inputY = outputY * strideHeight + poolY - padTop;
                    if (inputY < 0 || inputY >= height) {
                        continue;
                    }
                    for (let poolX = 0; poolX < poolWidth; poolX++) {
                        const inputX = outputX * strideWidth + poolX - padLeft;
                        if (inputX < 0 || inputX >= width) {
                            continue;
                        }
                        const inputIndex = ((inputY * width + inputX) * channels) + channel;
                        maximum = Math.max(maximum, data.values[inputIndex]);
                    }
                }
                const outputIndex = ((outputY * outputWidth + outputX) * channels) + channel;
                output[outputIndex] = maximum;
            }
        }
    }

    return { values: output, shape: [outputHeight, outputWidth, channels] };
};

lab.dense = async (data, node, cache, override) => {
    const { kernel: kernelEntry, bias: biasEntry } = lab.kernelAndBias(node);
    if (!kernelEntry) {
        throw new Error(`Top-3 preview could not find Dense kernel for '${node.name}'.`);
    }
    const kernel = await lab.tensor(kernelEntry, cache, override);
    const bias = await lab.tensor(biasEntry, cache, override);
    if (kernel.shape.length !== 2) {
        throw new Error('Top-3 preview expects a 2D Dense kernel.');
    }

    const [inputSize, outputSize] = kernel.shape;
    if (data.values.length !== inputSize) {
        throw new Error(`Dense input mismatch: got ${data.values.length}, expected ${inputSize}.`);
    }

    const output = new Array(outputSize).fill(0);
    const biasValues = bias ? bias.values : null;

    for (let outputIndex = 0; outputIndex < outputSize; outputIndex++) {
        let sum = biasValues ? biasValues[outputIndex] : 0;
        for (let inputIndex = 0; inputIndex < inputSize; inputIndex++) {
            sum += data.values[inputIndex] * kernel.values[inputIndex * outputSize + outputIndex];
        }
        output[outputIndex] = sum;
    }

    return {
        values: lab.applyActivation(output, lab.activation(node)),
        shape: [outputSize]
    };
};

lab.infer = async (graph, input, cache, override = null) => {
    if (!graph || !Array.isArray(graph.nodes)) {
        throw new Error('Top-3 preview could not access the current model graph.');
    }

    let data = {
        values: input.values.slice(),
        shape: input.shape.slice()
    };

    for (const node of graph.nodes) {
        const type = node && node.type ? String(node.type.name || '') : '';
        switch (type) {
            case 'InputLayer':
                break;
            case 'Conv2D':
                data = await lab.conv2d(data, node, cache, override);
                break;
            case 'MaxPooling2D':
                data = lab.maxPooling2d(data, node);
                break;
            case 'Flatten':
                data = { values: data.values.slice(), shape: [data.values.length] };
                break;
            case 'Dense':
                data = await lab.dense(data, node, cache, override);
                break;
            case 'Activation':
                data = {
                    values: lab.applyActivation(data.values, lab.activation(node)),
                    shape: data.shape.slice()
                };
                break;
            default:
                throw new Error(`Top-3 preview does not support layer '${type || node.name || '?'}'.`);
        }
    }

    return data.values;
};

lab.inputShape = (graph) => {
    if (!graph || !Array.isArray(graph.inputs) || graph.inputs.length === 0) {
        return [32, 32, 3];
    }
    const argument = graph.inputs[0];
    const value = argument && Array.isArray(argument.value) ? argument.value[0] : null;
    const dimensions = value && value.type && value.type.shape && Array.isArray(value.type.shape.dimensions) ?
        value.type.shape.dimensions : [];
    if (dimensions.length === 4) {
        return [Number(dimensions[1]), Number(dimensions[2]), Number(dimensions[3])];
    }
    if (dimensions.length === 3) {
        return dimensions.map((dimension) => Number(dimension));
    }
    return [32, 32, 3];
};

lab.loadImage = async (document, file, graph) => {
    const [height, width, channels] = lab.inputShape(graph);
    if (!Number.isInteger(height) || !Number.isInteger(width) || channels !== 3) {
        throw new Error('Top-3 preview currently supports RGB image inputs with a fixed height and width.');
    }

    const window = document.defaultView;
    const url = window.URL.createObjectURL(file);
    try {
        const image = document.createElement('img');
        await new Promise((resolve, reject) => {
            image.onload = resolve;
            image.onerror = () => reject(new Error('The selected image could not be decoded.'));
            image.src = url;
        });

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d', { willReadFrequently: true });
        context.drawImage(image, 0, 0, width, height);
        const pixels = context.getImageData(0, 0, width, height).data;
        const values = new Array(height * width * channels);

        let output = 0;
        for (let index = 0; index < pixels.length; index += 4) {
            values[output++] = pixels[index] / 255;
            values[output++] = pixels[index + 1] / 255;
            values[output++] = pixels[index + 2] / 255;
        }

        return { values, shape: [height, width, channels] };
    } finally {
        window.URL.revokeObjectURL(url);
    }
};

lab.top3 = (scores) => {
    const labels = scores.length === lab.classLabels.length ?
        lab.classLabels : scores.map((value, index) => `Class ${index}`);
    return scores
        .map((score, index) => ({ label: labels[index], score }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 3);
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
.compression-lab input[type='file'] { max-width: 100%; }
.compression-lab button { min-height: 28px; border: 1px solid #aaa; border-radius: 4px; background: #f4f4f4; color: #222; padding: 3px 9px; font: inherit; cursor: pointer; }
.compression-lab button:hover { background: #e8e8e8; }
.compression-lab button:disabled { opacity: 0.55; cursor: default; }
.compression-lab-tabs { display: flex; gap: 4px; margin: 10px 0 8px 0; }
.compression-lab-tabs button[aria-selected='true'] { font-weight: 700; border-color: #666; }
.compression-lab-summary { margin: 8px 0; padding: 7px 8px; border: 1px solid #d0d0d0; border-radius: 4px; }
.compression-lab-predictions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
.compression-lab-prediction { border: 1px solid #d0d0d0; border-radius: 4px; padding: 7px 8px; min-width: 0; }
.compression-lab-prediction-title { font-weight: 700; margin-bottom: 4px; }
.compression-lab-prediction ol { margin: 0; padding-left: 22px; }
.compression-lab-prediction li { margin: 2px 0; }
.compression-lab-table-wrap { overflow-x: auto; margin-top: 8px; }
.compression-lab table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
.compression-lab th, .compression-lab td { border-bottom: 1px solid #ddd; padding: 4px 6px; text-align: right; white-space: nowrap; }
.compression-lab th:first-child, .compression-lab td:first-child { text-align: left; }
.compression-lab-error { color: #b00020; margin-top: 6px; }
@media (max-width: 520px) {
    .compression-lab-predictions { grid-template-columns: 1fr; }
}
@media (prefers-color-scheme: dark) {
    .compression-lab-note { color: #aaa; }
    .compression-lab button { background: #555; color: #eee; border-color: #777; }
    .compression-lab button:hover { background: #666; }
    .compression-lab select, .compression-lab input { border-color: #777; }
    .compression-lab-summary, .compression-lab-prediction, .compression-lab th, .compression-lab td { border-color: #666; }
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

lab.predictionPair = (document, modifiedTitle) => {
    const root = document.createElement('div');
    root.className = 'compression-lab-predictions';

    const create = (title) => {
        const box = document.createElement('div');
        box.className = 'compression-lab-prediction';
        const heading = document.createElement('div');
        heading.className = 'compression-lab-prediction-title';
        heading.textContent = title;
        const content = document.createElement('div');
        content.textContent = 'Load a test image to calculate Top-3.';
        box.appendChild(heading);
        box.appendChild(content);
        root.appendChild(box);
        return { heading, content };
    };

    return {
        root,
        original: create('Original Top-3'),
        modified: create(modifiedTitle)
    };
};

lab.renderTop3 = (target, values, emptyText) => {
    target.content.replaceChildren();
    if (!values) {
        target.content.textContent = emptyText;
        return;
    }

    const document = target.content.ownerDocument;
    const list = document.createElement('ol');
    for (const item of lab.top3(values)) {
        const row = document.createElement('li');
        row.textContent = `${item.label}  ${(item.score * 100).toFixed(1)}%`;
        list.appendChild(row);
    }
    target.content.appendChild(list);
};

lab.attach = (sidebar) => {
    const node = sidebar._node;
    const entries = lab.weights(node);
    if (entries.length === 0 || sidebar.element.querySelector('.compression-lab')) {
        return;
    }

    const document = sidebar._host.document;
    const graph = sidebar._view ? sidebar._view.activeTarget : null;
    lab.styles(document);
    sidebar.addSection('Compression Lab');

    const root = document.createElement('div');
    root.className = 'compression-lab';
    sidebar.element.appendChild(root);

    const note = document.createElement('div');
    note.className = 'compression-lab-note';
    note.textContent = 'Educational preview only. The original model file is never overwritten. For the main class exercise, select the final Dense layer and use its kernel tensor.';
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
        const recommended = /kernel|weight/i.test(entry.name) ? ' · recommended' : '';
        option.textContent = `${entry.name} (${lab.shape(entry)})${recommended}`;
        selector.appendChild(option);
    });
    const preferred = entries.findIndex((entry) => /kernel|weight/i.test(entry.name));
    selector.selectedIndex = preferred >= 0 ? preferred : 0;
    tensorRow.appendChild(tensorLabel);
    tensorRow.appendChild(selector);
    root.appendChild(tensorRow);

    const tensorHelp = document.createElement('div');
    tensorHelp.className = 'compression-lab-note';
    root.appendChild(tensorHelp);

    const info = document.createElement('div');
    info.className = 'compression-lab-summary';
    info.textContent = 'Loading weights…';
    root.appendChild(info);

    const imageRow = document.createElement('div');
    imageRow.className = 'compression-lab-row';
    const imageLabel = document.createElement('label');
    imageLabel.textContent = 'Test image';
    const imageInput = document.createElement('input');
    imageInput.type = 'file';
    imageInput.accept = 'image/*';
    imageInput.setAttribute('aria-label', 'Test image');
    imageRow.appendChild(imageLabel);
    imageRow.appendChild(imageInput);
    root.appendChild(imageRow);

    const imageHelp = document.createElement('div');
    imageHelp.className = 'compression-lab-note';
    imageHelp.textContent = 'Load an image to compare the model Top-3 before and after compression. Classroom CIFAR-10 images are resized to the model input and normalized to 0–1.';
    root.appendChild(imageHelp);

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
    applyPruning.textContent = 'Apply Pruning';
    const resetPruning = document.createElement('button');
    resetPruning.type = 'button';
    resetPruning.textContent = 'Reset';
    thresholdRow.appendChild(thresholdLabel);
    thresholdRow.appendChild(threshold);
    thresholdRow.appendChild(applyPruning);
    thresholdRow.appendChild(resetPruning);
    pruningPanel.appendChild(thresholdRow);

    const pruningNote = document.createElement('div');
    pruningNote.className = 'compression-lab-note';
    pruningNote.textContent = 'Each click recalculates pruning from the original tensor. Pruning is not accumulated across clicks.';
    pruningPanel.appendChild(pruningNote);

    const pruningSummary = document.createElement('div');
    pruningSummary.className = 'compression-lab-summary';
    pruningPanel.appendChild(pruningSummary);

    const pruningPredictions = lab.predictionPair(document, 'Pruned Top-3');
    pruningPanel.appendChild(pruningPredictions.root);

    const quantizeRow = document.createElement('div');
    quantizeRow.className = 'compression-lab-row';
    const precisionLabel = document.createElement('label');
    precisionLabel.textContent = 'Precision';
    const precision = document.createElement('select');
    precision.setAttribute('aria-label', 'Quantization precision');
    for (const bits of [16, 8, 4, 2]) {
        const option = document.createElement('option');
        option.value = String(bits);
        option.textContent = `INT${bits}`;
        if (bits === 8) {
            option.selected = true;
        }
        precision.appendChild(option);
    }
    const applyQuantization = document.createElement('button');
    applyQuantization.type = 'button';
    applyQuantization.textContent = 'Apply Quantization';
    const resetQuantization = document.createElement('button');
    resetQuantization.type = 'button';
    resetQuantization.textContent = 'Reset';
    quantizeRow.appendChild(precisionLabel);
    quantizeRow.appendChild(precision);
    quantizeRow.appendChild(applyQuantization);
    quantizeRow.appendChild(resetQuantization);
    quantizationPanel.appendChild(quantizeRow);

    const originalPrecision = document.createElement('div');
    originalPrecision.className = 'compression-lab-note';
    originalPrecision.textContent = 'Original Precision: FP32';
    quantizationPanel.appendChild(originalPrecision);

    const quantizationSummary = document.createElement('div');
    quantizationSummary.className = 'compression-lab-summary';
    quantizationSummary.textContent = 'Choose INT16, INT8, INT4, or INT2, then press Apply Quantization.';
    quantizationPanel.appendChild(quantizationSummary);

    const quantizationNote = document.createElement('div');
    quantizationNote.className = 'compression-lab-note';
    quantizationNote.textContent = 'Weight Reduction compares bits per weight with FP32. It is not the serialized model file size.';
    quantizationPanel.appendChild(quantizationNote);

    const quantizationPredictions = lab.predictionPair(document, 'Quantized Top-3');
    quantizationPanel.appendChild(quantizationPredictions.root);

    const error = document.createElement('div');
    error.className = 'compression-lab-error';
    error.setAttribute('role', 'status');
    root.appendChild(error);

    const state = {
        original: [],
        pruned: [],
        quantization: null,
        input: null,
        originalPrediction: null,
        cache: new Map()
    };

    const updateTensorHelp = () => {
        const entry = entries[selector.selectedIndex];
        if (/bias/i.test(entry.name)) {
            tensorHelp.textContent = 'bias contains one additive offset per output. It can be previewed here, but use kernel for the main pruning/quantization exercise.';
        } else {
            tensorHelp.textContent = 'kernel contains the connection weights between inputs and outputs. This is the recommended tensor for the class exercise.';
        }
    };

    const renderPruning = (after) => {
        const before = state.original;
        const zeroBefore = before.reduce((count, value) => count + (value === 0 ? 1 : 0), 0);
        const zeroAfter = after.reduce((count, value) => count + (value === 0 ? 1 : 0), 0);
        const sparsityBefore = before.length === 0 ? 0 : zeroBefore / before.length;
        const sparsityAfter = after.length === 0 ? 0 : zeroAfter / after.length;
        pruningSummary.textContent = `Sparsity: ${(sparsityBefore * 100).toFixed(1)}% → ${(sparsityAfter * 100).toFixed(1)}%  (${zeroAfter}/${after.length} zeros)`;
    };

    const renderQuantization = () => {
        const result = state.quantization;
        if (!result) {
            quantizationSummary.textContent = 'Choose INT16, INT8, INT4, or INT2, then press Apply Quantization.';
            return;
        }
        quantizationSummary.textContent = `Precision: ${result.precision} · Weight Reduction: ${(result.weightReduction * 100).toFixed(2).replace(/\.00$/, '')}%`;
    };

    const ensureOriginalPrediction = async () => {
        if (!state.input) {
            return null;
        }
        if (!state.originalPrediction) {
            state.originalPrediction = await lab.infer(graph, state.input, state.cache);
        }
        return state.originalPrediction;
    };

    const renderPredictionError = (pair, message) => {
        lab.renderTop3(pair.original, null, message);
        lab.renderTop3(pair.modified, null, message);
    };

    const updatePruningPredictions = async () => {
        if (!state.input) {
            renderPredictionError(pruningPredictions, 'Load a test image to calculate Top-3.');
            return;
        }
        try {
            const original = await ensureOriginalPrediction();
            const entry = entries[selector.selectedIndex];
            const modified = await lab.infer(graph, state.input, state.cache, {
                initializer: entry.initializer,
                values: state.pruned
            });
            lab.renderTop3(pruningPredictions.original, original, '');
            lab.renderTop3(pruningPredictions.modified, modified, '');
        } catch (err) {
            renderPredictionError(pruningPredictions, err.message);
        }
    };

    const updateQuantizationPredictions = async () => {
        if (!state.input) {
            renderPredictionError(quantizationPredictions, 'Load a test image to calculate Top-3.');
            return;
        }
        try {
            const original = await ensureOriginalPrediction();
            lab.renderTop3(quantizationPredictions.original, original, '');
            if (!state.quantization) {
                lab.renderTop3(quantizationPredictions.modified, null, 'Apply Quantization to calculate Top-3.');
                return;
            }
            const entry = entries[selector.selectedIndex];
            const modified = await lab.infer(graph, state.input, state.cache, {
                initializer: entry.initializer,
                values: state.quantization.dequantized
            });
            lab.renderTop3(quantizationPredictions.modified, modified, '');
        } catch (err) {
            renderPredictionError(quantizationPredictions, err.message);
        }
    };

    const updatePredictions = async () => {
        await updatePruningPredictions();
        await updateQuantizationPredictions();
    };

    const setEnabled = (enabled) => {
        selector.disabled = !enabled;
        threshold.disabled = !enabled;
        applyPruning.disabled = !enabled;
        resetPruning.disabled = !enabled;
        precision.disabled = !enabled;
        applyQuantization.disabled = !enabled;
        resetQuantization.disabled = !enabled;
        imageInput.disabled = !enabled;
    };

    const loadSelected = async () => {
        setEnabled(false);
        error.textContent = '';
        info.textContent = 'Loading weights…';
        try {
            const entry = entries[selector.selectedIndex];
            const values = await lab.load(entry);
            state.original = values.slice();
            state.pruned = values.slice();
            state.quantization = null;
            updateTensorHelp();
            info.textContent = `Layer: ${node.name || (node.type ? node.type.name : '?')} · tensor: ${entry.name} · shape: ${lab.shape(entry)} · ${values.length.toLocaleString()} values`;
            renderPruning(state.pruned);
            renderQuantization();
            setEnabled(true);
            await updatePredictions();
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
        renderQuantization();
    });

    selector.addEventListener('change', () => {
        loadSelected();
    });

    imageInput.addEventListener('change', async () => {
        error.textContent = '';
        const [file] = Array.from(imageInput.files || []);
        if (!file) {
            state.input = null;
            state.originalPrediction = null;
            imageHelp.textContent = 'Load an image to compare the model Top-3 before and after compression. Classroom CIFAR-10 images are resized to the model input and normalized to 0–1.';
            await updatePredictions();
            return;
        }
        try {
            imageInput.disabled = true;
            imageHelp.textContent = `Loading ${file.name}…`;
            state.input = await lab.loadImage(document, file, graph);
            state.originalPrediction = null;
            imageHelp.textContent = `Test image: ${file.name} · resized to ${state.input.shape[1]} × ${state.input.shape[0]} · normalized to 0–1`;
            await updatePredictions();
        } catch (err) {
            state.input = null;
            state.originalPrediction = null;
            error.textContent = err.message;
            imageHelp.textContent = 'Top-3 preview unavailable for this image.';
            await updatePredictions();
        } finally {
            imageInput.disabled = false;
        }
    });

    applyPruning.addEventListener('click', async () => {
        error.textContent = '';
        try {
            const value = Number(threshold.value);
            const result = prune(state.original, value);
            state.pruned = result.values.slice();
            renderPruning(state.pruned);
            await updatePruningPredictions();
        } catch (err) {
            error.textContent = err.message;
        }
    });

    resetPruning.addEventListener('click', async () => {
        state.pruned = state.original.slice();
        renderPruning(state.pruned);
        error.textContent = '';
        await updatePruningPredictions();
    });

    applyQuantization.addEventListener('click', async () => {
        error.textContent = '';
        try {
            const bits = Number.parseInt(precision.value, 10);
            state.quantization = quantize(state.original, bits);
            renderQuantization();
            await updateQuantizationPredictions();
        } catch (err) {
            error.textContent = err.message;
        }
    });

    resetQuantization.addEventListener('click', async () => {
        state.quantization = null;
        renderQuantization();
        error.textContent = '';
        await updateQuantizationPredictions();
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
