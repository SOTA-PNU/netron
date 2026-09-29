# Compression Lab

This fork adds a small classroom-oriented Pruning and Quantization panel to the Netron browser UI.

## Purpose

The panel is intended for a short hands-on exercise on the **last layer of a model** (for example `Dense(10)`). Students can inspect real weight values and apply Pruning or an educational INT8 Quantization preview without overwriting the original model file.

## Run

Use the `feature/compression-lab` branch.

```bash
git clone -b feature/compression-lab https://github.com/SOTA-PNU/netron.git
cd netron
python package.py build start
```

Open a model in the browser and click a layer that has weight tensors.

## Classroom workflow

1. Open the model used in class.
2. Click the final layer, such as `Dense(10)`.
3. In **Compression Lab**, choose the kernel/weight tensor.
4. Try **Pruning**:
   - change the threshold,
   - apply it to the working copy,
   - compare the Before/After values,
   - observe the change in sparsity.
5. Try **Quantization**:
   - choose a weight index,
   - inspect `min`, `max`, `scale`, and `zero point`,
   - compare FP32, INT8, dequantized value, and absolute error.
6. Use **Reset** to return to the original in-memory working copy.

## Important behavior

- The panel uses a **non-destructive working copy** of the selected weight tensor.
- It does **not** overwrite or serialize the original `.keras` or `.tflite` file.
- The Quantization tab is an educational per-tensor affine INT8 preview.
- A real TensorFlow/LiteRT converter can use a different scheme, such as symmetric or per-channel quantization, so internal integer weights may differ from the preview.

## Files added by this fork

- `source/compression.js`: Pruning and affine INT8 preview calculations.
- `source/compression-lab.js`: Netron sidebar UI integration.
- `source/index.js`: loads the Compression Lab module in the browser build.

## Recommended lesson connection

Use this UI for the manual last-layer exercise, then return to the Jupyter notebook for whole-model conversion and the final FP32 vs INT8 comparison of model size, accuracy, and inference time.
