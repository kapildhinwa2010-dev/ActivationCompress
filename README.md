# ActivationCompress

ActivationCompress is a small Python/PyTorch implementation of activation projection and reconstruction utilities. It can derive projection bases from model outputs using SVD, centered PCA, or a random orthogonal projection, and includes a separate trainable compressor/decompressor module.

**Status:** Experimental utility code, not a validated memory- or compute-optimization system. The repository contains implementation code and tests, but no benchmark evidence or end-to-end model-quality evaluation.

## Implemented functionality

- `SVDCompressor`, `PCACompressor`, and `RandomProjectionCompressor` project activation features onto a lower-dimensional basis and reconstruct an output.
- `ActivationCompressor.calibrate()` collects outputs from `nn.Linear` and `nn.Conv2d` modules during a model forward pass, stores bases, and reports per-layer shapes, an estimated compression ratio, retained variance, and reconstruction error.
- `ActivationCompressorWrapper` attaches hooks to matching `nn.Linear` modules and removes them on exit from its context manager.
- `DynamicActivationCompressor` is a trainable PyTorch MLP-style encoder/decoder. Its parameters can be optimized by the caller; it does not train itself.
- `CompressionStats` and `create_activation_compressor()` provide statistics and a factory for the SVD, PCA, and random-projection implementations.

## Installation and use

There is no command-line interface. The implementation is a single file, `activation_compress.py`. Run commands from the repository root so Python can import that file directly. Install the declared dependencies with:

```bash
python -m pip install -r requirements.txt
```

Example using a sample tensor to calibrate a model and then attach the calibrated projections:

```python
import torch
from torch import nn
from activation_compress import ActivationCompressorWrapper, SVDCompressor

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
inputs = torch.randn(16, 128)
compressor = SVDCompressor(target_dim=16)

stats = compressor.calibrate(model, inputs)
with ActivationCompressorWrapper(model, compressor):
    outputs = model(inputs)

print(outputs.shape)
print(stats)
```

`compress()` is also available for an individual tensor and returns `(reconstructed_tensor, stats)`. The returned tensor has the original feature dimension: this implementation projects to codes and reconstructs them rather than exposing a compact tensor for downstream storage or computation.

## Packaging note

Although `pyproject.toml` declares project metadata and dependencies, its setuptools configuration discovers packages while the implementation is a top-level module. A wheel built from this revision contains metadata and the license but omits `activation_compress.py`. Consequently, this README does not recommend `pip install .` as a usable installation of the module; use it from a source checkout instead.

## Tests

Run the repository tests from its root with:

```bash
python -m pytest -q
```

On revision `c3635ed191187d0d7313b52226b7e7eb80bec08d`, this command was run locally with Python 3.12.3 and PyTorch 2.15.0+cpu: **37 tests passed**. This is a local run, not a claim about a GitHub Actions result. The tests exercise tensor shapes, basic statistics, calibration/hooks, the dynamic module, edge cases, and small model integrations; they do not establish end-to-end accuracy, speed, or memory savings.

## Limitations

- `compress()` and the forward hooks return reconstructed activations with the original shape. The reported ratio estimates the sizes of codes plus a basis for an individual projection; it is not a measurement of peak memory, throughput, or end-to-end compression. No timing, memory, or task-quality benchmarks are included.
- SVD/PCA bases are derived from the calibration inputs, and reconstruction is lossy. The repository does not evaluate behavior under distribution shift or quantify task-level degradation.
- The random projection is generated for each compression call; it is not a calibrated, persisted basis in the same way as the SVD/PCA methods.
- `calibrate()` collects both `Linear` and `Conv2d` outputs, but `ActivationCompressorWrapper` installs hooks only on `Linear` modules. The wrapper's matching behavior is based on a name pattern or the module type name.
- The base method validator accepts the string `autoencoder`, but no corresponding basis implementation exists; the factory supports only `svd`, `pca`, and `random`. `DynamicActivationCompressor` is a separate learnable module, not that factory method.
- `target_dim` is not comprehensively validated for every input shape or device combination. Check shapes, device placement, and numerical behavior for the model and hardware you intend to use.

## License

The repository includes an MIT License, copyright © 2024 Kapil Dhinwa. See [LICENSE](LICENSE) for its terms.
