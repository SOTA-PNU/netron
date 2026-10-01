# Compression Lab

This fork adds a small classroom-oriented Pruning and Quantization panel to the Netron browser UI.

## Purpose

The panel is intended for a short hands-on exercise on the **last layer of a model** (for example `Dense(10)`). Students can inspect real weight values, apply Pruning, compare multiple integer precisions, and see how compression changes the model's Top-3 prediction without overwriting the original model file.

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
4. Load a test image to enable the Top-3 comparison.
5. Try **Pruning**:
   - change the threshold,
   - apply it to the working copy,
   - observe the change in sparsity,
   - compare Original Top-3 and Pruned Top-3.
6. Try **Quantization**:
   - the original precision is FP32,
   - choose `INT16`, `INT8`, `INT4`, or `INT2`,
   - apply quantization to the selected tensor,
   - observe Weight Reduction,
   - compare Original Top-3 and Quantized Top-3.
7. Use **Reset** to return to the original in-memory working copy.

## Quantization view

The Quantization tab intentionally keeps the classroom UI small:

- **Original Precision:** FP32
- **Quantization Precision:** INT16 / INT8 / INT4 / INT2
- **Weight Reduction:** reduction in bits per weight relative to FP32
  - INT16: 50%
  - INT8: 75%
  - INT4: 87.5%
  - INT2: 93.75%
- **Top-3 Prediction:** original model vs quantized working copy

Weight Reduction is not the serialized `.keras` or `.tflite` file-size reduction.

## Top-3 preview

For the CIFAR-10 classroom model, the browser-side preview supports the Sequential path used in class:

- Conv2D
- MaxPooling2D
- Flatten
- Dense
- ReLU / Softmax

The selected image is resized to the model input size and normalized to `0–1`, matching the classroom CIFAR-10 preprocessing. Quantized inference uses the dequantized low-precision weights so students can observe the numerical effect of quantization on predictions.

If an unsupported layer or input format is encountered, the Compression Lab reports that Top-3 preview is unavailable instead of inventing a prediction.

## Important behavior

- The panel uses a **non-destructive working copy** of the selected weight tensor.
- It does **not** overwrite or serialize the original `.keras` or `.tflite` file.
- Quantization is an educational per-tensor affine preview at the selected integer precision.
- A real TensorFlow/LiteRT converter can use a different scheme, such as symmetric or per-channel quantization, so internal integer weights can differ from the preview.
- No deployment model is generated.

## Files added by this fork

- `source/compression.js`: Pruning and selectable INT16 / INT8 / INT4 / INT2 quantization calculations.
- `source/compression-lab.js`: Netron sidebar UI integration and classroom Top-3 preview.
- `source/index.js`: loads the Compression Lab module in the browser build.
