import assert from 'node:assert/strict';
import { quantize } from '../source/compression.js';
import { setImmediate } from 'node:timers/promises';

const descendants = (element) => element.children.flatMap((child) => [child, ...descendants(child)]);

class Element {
    constructor(document, tag) {
        this.ownerDocument = document;
        this.tagName = tag.toUpperCase();
        this.children = [];
        this.attributes = new Map();
        this.listeners = new Map();
        this.selectedIndex = 0;
        this.disabled = false;
        this.hidden = false;
        this._text = '';
        this._value = '';
        this._src = '';
    }

    appendChild(element) {
        this.children.push(element);
        if (element.selected) {
            this.selectedIndex = this.children.length - 1;
        }
        return element;
    }

    replaceChildren() {
        this.children = [];
        this._text = '';
    }

    set textContent(value) {
        this.replaceChildren();
        this._text = String(value);
    }

    get textContent() {
        return this._text + this.children.map((child) => child.textContent).join('');
    }

    set value(value) {
        if (this.tagName === 'SELECT') {
            this.selectedIndex = this.children.findIndex((child) => child.value === String(value));
        } else {
            this._value = String(value);
        }
    }

    get value() {
        return this.tagName === 'SELECT' ? this.children[this.selectedIndex].value : this._value;
    }

    set src(value) {
        this._src = value;
        const file = this.ownerDocument.urls.get(value);
        if (file) {
            Promise.resolve().then(() => {
                if (file.invalid) {
                    this.onerror();
                } else {
                    this.onload();
                }
            });
        }
    }

    get src() {
        return this._src;
    }

    setAttribute(name, value) {
        this.attributes.set(name, value);
    }

    removeAttribute(name) {
        this.attributes.delete(name);
        if (name === 'src') {
            this._src = '';
        }
    }

    addEventListener(name, listener) {
        this.listeners.set(name, listener);
    }

    async dispatch(name) {
        if (this.disabled) {
            return;
        }
        const listener = this.listeners.get(name);
        if (listener) {
            await listener();
            await setImmediate();
        }
    }

    querySelector(selector) {
        return descendants(this).find((element) => selector === `.${element.className}`) || null;
    }

    getContext() {
        return {
            drawImage: (image) => {
                this.image = this.ownerDocument.urls.get(image.src);
            },
            getImageData: () => ({ data: this.image.pixels })
        };
    }

    toDataURL() {
        return `data:image/png;base64,${this.image.preview}`;
    }
}

class Document {
    constructor() {
        this.urls = new Map();
        this.revoked = [];
        this.counter = 0;
        this.head = this.createElement('head');
        this.defaultView = {
            URL: {
                createObjectURL: (file) => {
                    const url = `blob:test-${++this.counter}`;
                    this.urls.set(url, file);
                    return url;
                },
                revokeObjectURL: (url) => {
                    this.revoked.push(url);
                    this.urls.delete(url);
                }
            }
        };
    }

    createElement(tag) {
        return new Element(this, tag);
    }

    getElementById(id) {
        return descendants(this.head).find((element) => element.id === id) || null;
    }
}

class NodeSidebar {
    constructor(node, graph) {
        const document = new Document();
        this._node = node;
        this._host = { document };
        this._view = { activeTarget: graph };
        this.element = document.createElement('div');
    }

    addSection() {}

    render() {}

    error(error) {
        throw error;
    }
}

globalThis.window = { exports: { view: { NodeSidebar } } };
await import('../source/compression-lab.js');

const tensor = (name, shape, values) => {
    const type = { dataType: 'float32', shape: { dimensions: shape } };
    const initializer = { name, type, encoding: '|', values };
    return { name, value: [{ name, type, initializer }] };
};

const original = [2, ...new Array(9).fill(0), ...new Array(11).fill(0.003)];
const bias = [0, 0, 0, 0, 0, 0, 1];
const node = {
    name: 'dense',
    type: { name: 'Dense' },
    inputs: [
        tensor('kernel', [3, 7], original),
        tensor('bias_weight', [7], bias),
        tensor('recurrent_kernel', [3, 1], [0.1, 0.2, 0.3])
    ],
    attributes: [{ name: 'activation', value: 'softmax' }]
};
const graph = {
    nodes: [node],
    inputs: [{ value: [{ type: { shape: { dimensions: [1, 1, 1, 3] } } }] }]
};
const sidebar = new NodeSidebar(node, graph);
sidebar.render();
await setImmediate();
const document = sidebar._host.document;
const elements = descendants(sidebar.element);
const byLabel = (label) => {
    const element = elements.find((item) => item.attributes.get('aria-label') === label);
    assert.ok(element, label);
    return element;
};
const buttons = (text) => elements.filter((element) => element.tagName === 'BUTTON' && element.textContent === text);
const summaries = elements.filter((element) => element.className === 'compression-lab-summary');
const bodies = elements.filter((element) => element.tagName === 'TBODY');
const ranges = elements.filter((element) => element.tagName === 'SPAN' && element.attributes.get('role') === 'status');
const applied = elements.find((element) => element.textContent.startsWith('Applied Threshold:'));
const image = elements.find((element) => element.className === 'compression-lab-image');
const predictions = elements.filter((element) => element.className === 'compression-lab-prediction').map((element) => element.children[1]);
const error = elements.find((element) => element.className === 'compression-lab-error');
const selector = byLabel('Weight tensor');
const threshold = byLabel('Pruning threshold');
const precision = byLabel('Quantization precision');
const imageInput = byLabel('Test image');

// Selection excludes bias, while the shared inference path still reads it.
assert.deepEqual(selector.children.map((option) => option.textContent.split(' ')[0]), ['kernel', 'recurrent_kernel']);
assert.equal(image.hidden, true);
assert.match(ranges[0].textContent, /Showing 1–10 of 21 weights/);
assert.equal(bodies[0].children.length, 10);
assert.equal(byLabel('Pruning previous weights').disabled, true);
await byLabel('Pruning next weights').dispatch('click');
assert.match(ranges[0].textContent, /Showing 11–20 of 21 weights/);
assert.equal(bodies[0].children[0].children[0].textContent, '10');
await byLabel('Pruning next weights').dispatch('click');
assert.match(ranges[0].textContent, /Showing 21–21 of 21 weights/);
assert.equal(bodies[0].children.length, 1);
assert.equal(byLabel('Pruning next weights').disabled, true);
await byLabel('Pruning previous weights').dispatch('click');
assert.equal(bodies[0].children.length, 10);

// Editing or rejecting a threshold must not relabel an existing result.
threshold.value = '0.02';
await buttons('Apply Pruning')[0].dispatch('click');
assert.equal(applied.textContent, 'Applied Threshold: 0.02');
assert.match(summaries[1].textContent, /20\/21 zeros/);
threshold.value = '0.5';
assert.equal(applied.textContent, 'Applied Threshold: 0.02');
threshold.value = '-1';
await buttons('Apply Pruning')[0].dispatch('click');
assert.equal(applied.textContent, 'Applied Threshold: 0.02');
assert.match(error.textContent, /non-negative/);
await buttons('Reset')[0].dispatch('click');
assert.match(applied.textContent, /original weights/);
assert.match(summaries[1].textContent, /9\/21 zeros/);

// MAE includes errors beyond the visible first ten weights.
precision.value = '2';
await precision.dispatch('change');
await buttons('Apply Quantization')[0].dispatch('click');
const quantized = quantize(original, 2);
const mean = quantized.error.reduce((sum, value) => sum + value, 0) / original.length;
assert.equal(quantized.error.slice(0, 10).reduce((sum, value) => sum + value, 0), 0);
assert.ok(mean > 0);
assert.equal(quantized.meanAbsoluteError, mean);
assert.match(summaries[2].textContent, /32bit → 2bit/);
assert.match(summaries[2].textContent, /93.75%/);
assert.ok(summaries[2].textContent.includes(mean.toFixed(6).replace(/0+$/, '')));
assert.match(summaries[2].textContent, /all 21 weights/);
const summary = summaries[2].textContent;
await byLabel('Quantization next weights').dispatch('click');
await byLabel('Quantization next weights').dispatch('click');
assert.match(ranges[1].textContent, /Showing 21–21 of 21 weights/);
assert.equal(bodies[1].children.length, 1);
assert.equal(bodies[1].children[0].children[0].textContent, '20');
assert.equal(summaries[2].textContent, summary);
precision.value = '8';
await precision.dispatch('change');
assert.equal(bodies[1].children.length, 0);
assert.equal(byLabel('Quantization next weights').disabled, true);
await buttons('Apply Quantization')[0].dispatch('click');
assert.match(summaries[2].textContent, /32bit → 8bit/);
assert.match(summaries[2].textContent, /75%/);
assert.match(ranges[1].textContent, /Showing 1–10 of 21 weights/);
await buttons('Reset')[1].dispatch('click');
assert.equal(bodies[1].children.length, 0);
assert.match(summaries[3].textContent, /Scale: —/);

// Preview, filename, inference input, and URL cleanup follow each selected file.
imageInput.files = [{ name: 'red.png', pixels: new Uint8Array([255, 0, 0, 255]), preview: 'RED' }];
await imageInput.dispatch('change');
assert.equal(image.hidden, false);
assert.equal(image.src, 'data:image/png;base64,RED');
assert.equal(image.alt, 'Test image: red.png');
assert.ok(elements.some((element) => element.textContent.startsWith('Test image: red.png, resized')));
assert.equal(document.urls.size, 0);
assert.equal(document.revoked.length, 1);
assert.match(predictions[0].textContent, /^Class 0/);
threshold.value = '3';
await buttons('Apply Pruning')[0].dispatch('click');
assert.match(predictions[1].textContent, /^Class 6/);
assert.deepEqual(bias, [0, 0, 0, 0, 0, 0, 1]);
assert.deepEqual(original, [2, ...new Array(9).fill(0), ...new Array(11).fill(0.003)]);
imageInput.files = [{ name: 'green.png', pixels: new Uint8Array([0, 255, 0, 255]), preview: 'GREEN' }];
await imageInput.dispatch('change');
assert.equal(image.src, 'data:image/png;base64,GREEN');
assert.equal(image.alt, 'Test image: green.png');
assert.equal(document.revoked.length, 2);
assert.match(predictions[0].textContent, /^Class 6/);
imageInput.files = [{ name: 'invalid.png', invalid: true }];
await imageInput.dispatch('change');
assert.equal(image.hidden, true);
assert.equal(image.src, '');
assert.equal(document.urls.size, 0);
assert.equal(document.revoked.length, 3);
assert.match(error.textContent, /could not be decoded/);
assert.match(predictions[0].textContent, /Load a test image/);
assert.equal(imageInput.disabled, false);
imageInput.files = [];
await imageInput.dispatch('change');
assert.equal(image.hidden, true);
assert.equal(image.alt, '');

// Switching tensors resets result state and clamps pages to the new count.
selector.value = '1';
await selector.dispatch('change');
assert.match(ranges[0].textContent, /Showing 1–3 of 3 weights/);
assert.equal(bodies[0].children.length, 3);
assert.equal(byLabel('Pruning next weights').disabled, true);
assert.match(applied.textContent, /original weights/);
assert.equal(bodies[1].children.length, 0);

const biasOnly = new NodeSidebar({ inputs: [tensor('bias', [1], [1])] }, graph);
biasOnly.render();
assert.equal(biasOnly.element.querySelector('.compression-lab'), null);
assert.equal(quantize([], 8).meanAbsoluteError, 0);
assert.equal(quantize([0, 0], 8).meanAbsoluteError, 0);
process.stdout.write('Compression Lab UI regression checks passed.\n');
