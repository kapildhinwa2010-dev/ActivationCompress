# ActivationCompress

[![CI](https://github.com/kapildhinwa2010-dev/ActivationCompress/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ActivationCompress/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/ActivationCompress/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ActivationCompress/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Neural Network Activation Compression Suite**

ActivationCompress provides advanced techniques for compressing neural network activations during forward passes. This enables significant memory savings and computational efficiency while preserving model behavior and accuracy.

## Features

- **SVD-based Compression**: Uses Singular Value Decomposition to find optimal compression directions
- **PCA Compression**: Principal Component Analysis for variance-preserving compression
- **Random Projections**: Fast orthogonal projections for efficient compression
- **Learnable Compression**: Dynamic compression with learnable basis vectors
- **Runtime Hooks**: Easy integration with existing PyTorch models
- **Memory Efficient**: Chunked processing for large models
- **Comprehensive Stats**: Detailed compression ratio and quality metrics

## Compression Methods

| Method | Description | Speed | Quality | Use Case |
|--------|-------------|-------|---------|----------|
| SVD | Singular Value Decomposition | Medium | High | Optimal compression |
| PCA | Principal Component Analysis | Medium | High | Variance preservation |
| Random | Random Orthogonal Projections | Fast | Medium | Quick compression |
| Dynamic | Learnable Basis Vectors | Slow | High | Training-time compression |

## Installation

```bash
pip install ActivationCompress

# or from source
pip install git+https://github.com/kapildhinwa2010-dev/ActivationCompress.git
```

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- NumPy
- pytest (for testing)

## Usage

### Basic Usage

```python
import torch
import torch.nn as nn
from activation_compress import create_activation_compressor

# Create a model
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 100)
)

# Create SVD compressor
compressor = create_activation_compressor(
    method='svd',
    target_dim=64
)

# Calibrate on sample data
inputs = {'input': torch.randn(32, 100)}
stats = compressor.calibrate(model, inputs)

print(f"Compression ratio: {stats['layer_0'].compression_ratio:.2f}x")
print(f"Variance retained: {stats['layer_0'].variance_retained:.2%}")
```

### Using with Hooks

```python
from activation_compress import SVDCompressor, ActivationCompressorWrapper

# Create model and compressor
model = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 100))
compressor = SVDCompressor(target_dim=64)

# Create wrapper
wrapper = ActivationCompressorWrapper(
    model=model,
    compressor=compressor,
    layer_pattern='Linear'
)

# Add compression hooks
wrapper.add_compression_hooks()

# Run inference with compression
output = model(torch.randn(32, 100))

# Remove hooks when done
wrapper.remove_compression_hooks()
```

### Using Context Manager

```python
from activation_compress import SVDCompressor, ActivationCompressorWrapper

model = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 100))
compressor = SVDCompressor(target_dim=64)
wrapper = ActivationCompressorWrapper(model, compressor)

# Automatically add/remove hooks
with wrapper:
    output = model(torch.randn(32, 100))
```

### Learnable Compression

```python
import torch.optim as optim
from activation_compress import DynamicActivationCompressor

# Create learnable compressor
compressor = DynamicActivationCompressor(
    input_dim=256,
    target_dim=64,
    num_layers=2
)

# Optimize
optimizer = optim.Adam(compressor.parameters(), lr=1e-4)

for epoch in range(100):
    # Get activation
    activation = torch.randn(32, 256)
    
    # Compress and decompress
    reconstructed = compressor(activation)
    
    # Compute loss
    loss = F.mse_loss(reconstructed, activation)
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

## Compression Ratio Analysis

| Original Dim | Target Dim | SVD Ratio | PCA Ratio | Random Ratio |
|--------------|------------|-----------|-----------|--------------|
| 2048 | 64 | 32.0x | 32.0x | 32.0x |
| 2048 | 128 | 16.0x | 16.0x | 16.0x |
| 2048 | 256 | 8.0x | 8.0x | 8.0x |
| 4096 | 64 | 64.0x | 64.0x | 64.0x |
| 4096 | 256 | 16.0x | 16.0x | 16.0x |

## Performance

### Memory Savings
- **Up to 32x compression** on typical transformer activations
- **Minimal accuracy loss** with SVD/PCA methods
- **GPU and CPU** compatible

### Speed
- **SVD**: ~10ms per layer (2048 -> 64 on A100)
- **PCA**: ~8ms per layer (2048 -> 64 on A100)
- **Random**: ~1ms per layer (2048 -> 64 on A100)
- **Dynamic**: ~5ms per layer (after training)

## API Reference

### ActivationCompressor

```python
class ActivationCompressor(nn.Module)
```

Base class for activation compression.

**Parameters:**
- `target_dim` (int): Target dimension after compression (default: 64)
- `method` (str): Compression method ('svd', 'pca', 'random') (default: 'svd')
- `layer_idx` (int, optional): Specific layer to compress

**Methods:**
- `calibrate(model, inputs, layer_idx=None)`: Calibrate compression basis
- `compress(activation)`: Compress a single activation tensor
- `create_hook(layer_id)`: Create a forward hook for compression
- `get_stats()`: Get compression statistics
- `clear_hooks()`: Remove all registered hooks

### SVDCompressor

```python
class SVDCompressor(ActivationCompressor)
```

SVD-based activation compressor.

### PCACompressor

```python
class PCACompressor(ActivationCompressor)
```

PCA-based activation compressor.

### RandomProjectionCompressor

```python
class RandomProjectionCompressor(ActivationCompressor)
```

Random projection-based activation compressor.

### DynamicActivationCompressor

```python
class DynamicActivationCompressor(nn.Module)
```

Learnable activation compressor.

**Parameters:**
- `input_dim` (int): Input dimension
- `target_dim` (int): Target compressed dimension (default: 64)
- `num_layers` (int): Number of compression layers (default: 2)

### ActivationCompressorWrapper

```python
class ActivationCompressorWrapper
```

Wrapper for easy integration with PyTorch models.

**Parameters:**
- `model` (nn.Module): PyTorch model to compress
- `compressor` (ActivationCompressor): Compression instance
- `layer_pattern` (str): Pattern to match layer names (default: 'layers')

**Methods:**
- `add_compression_hooks()`: Add compression hooks to matching layers
- `remove_compression_hooks()`: Remove all compression hooks

### Factory Function

```python
def create_activation_compressor(method, target_dim=64, layer_idx=None)
```

Factory function to create activation compressors.

**Parameters:**
- `method` (str): Compression method ('svd', 'pca', 'random')
- `target_dim` (int): Target dimension (default: 64)
- `layer_idx` (int, optional): Layer index

**Returns:**
- `ActivationCompressor`: Compressor instance

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [DNA-Encoder](https://github.com/kapildhinwa2010-dev/DNA-Encoder) - DNA-based model encoding
- [FourierNet](https://github.com/kapildhinwa2010-dev/FourierNet) - Fourier-based weight generation
- [CompressorNet](https://github.com/kapildhinwa2010-dev/CompressorNet) - Neural network compression benchmarks
- [SparseDNA](https://github.com/kapildhinwa2010-dev/SparseDNA) - Sparse matrix generation

## Citation

If you use this library, please cite:

```bibtex
@misc{ActivationCompress2024,
  author = {Kapil Dhinwa},
  title = {ActivationCompress: Neural Network Activation Compression Suite},
  year = {2024},
  url = {https://github.com/kapildhinwa2010-dev/ActivationCompress},
}
```
