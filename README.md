# ActivationCompress

[![CI](https://github.com/kapildhinwa2010-dev/ActivationCompress/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ActivationCompress/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/ActivationCompress/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ActivationCompress/actions/workflows/codeql.yml)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Compress neural-network activations during forward passes via SVD, PCA, random projections, or a learnable basis.

## What it does

ActivationCompressor subclasses calibrate a basis on sample activations
and install forward hooks that project activations to a lower dimension
and back. Four methods are included: SVD, PCA, random orthogonal
projections, and a learnable (Dynamic) compressor trained with ordinary
optimizers.

## Installation

    pip install torch numpy pytest

## Usage

    from activation_compress import create_activation_compressor

    compressor = create_activation_compressor(method="svd", target_dim=64)
    stats = compressor.calibrate(model, inputs)
    print(stats["layer_0"].compression_ratio, stats["layer_0"].variance_retained)

    # Hook-based use
    from activation_compress import SVDCompressor, ActivationCompressorWrapper
    wrapper = ActivationCompressorWrapper(model, SVDCompressor(target_dim=64))
    with wrapper:
        output = model(x)

## API

- ActivationCompressor(target_dim=64, method="svd", layer_idx=None):
  calibrate, compress, create_hook, get_stats, clear_hooks
- SVDCompressor, PCACompressor, RandomProjectionCompressor
- DynamicActivationCompressor(input_dim, target_dim=64, num_layers=2)
- ActivationCompressorWrapper(model, compressor, layer_pattern):
  add_compression_hooks / remove_compression_hooks, context manager
- create_activation_compressor(method, target_dim=64, layer_idx=None)

## Honest results and limitations

- The dimension ratio (e.g. 2048 -> 64 = 32x) is structural, not a
  measured end-to-end result. Actual memory savings depend on where
  activations live and whether decompression is needed immediately.
- No timing or accuracy numbers in this README are measured: earlier
  revisions quoted per-layer milliseconds on an A100 and "minimal
  accuracy loss" without experiments behind them. Treat speed and
  accuracy as unmeasured.
- SVD/PCA bases are calibrated per layer on a few sample batches; they
  degrade when the data distribution shifts away from calibration.
- The random-projection method loses information by design; its
  reconstruction error depends on the activation spectrum.
- Hooks add Python overhead per forward call; this is a research tool,
  not a serving-grade activation checkpointing system.

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that activation/behavior-based
  compression did not preserve task behaviour without retraining, and
  noise-robustness of the related carrier-mode scheme failed about half
  the trials. The robust design kept in the toolkit is the Reed-Solomon
  slot lattice (ECC-Memory).

## License

MIT - see [LICENSE](LICENSE).
