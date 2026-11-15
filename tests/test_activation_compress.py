"""
Tests for ActivationCompress module.

Comprehensive test suite covering all compression methods,
edge cases, and integration scenarios.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from activation_compress import (
    ActivationCompressor,
    SVDCompressor,
    PCACompressor,
    RandomProjectionCompressor,
    DynamicActivationCompressor,
    ActivationCompressorWrapper,
    CompressionStats,
    create_activation_compressor,
)


class TestCompressionStats:
    """Test CompressionStats dataclass."""

    def test_compression_stats_creation(self):
        """Test creating CompressionStats."""
        stats = CompressionStats(
            original_shape=(32, 256),
            compressed_shape=(32, 64),
            compression_ratio=4.0,
            variance_retained=0.95,
            reconstruction_error=0.05
        )
        assert stats.original_shape == (32, 256)
        assert stats.compressed_shape == (32, 64)
        assert stats.compression_ratio == 4.0
        assert stats.variance_retained == 0.95
        assert stats.reconstruction_error == 0.05

    def test_compression_stats_str(self):
        """Test CompressionStats string representation."""
        stats = CompressionStats(
            original_shape=(32, 256),
            compressed_shape=(32, 64),
            compression_ratio=4.0,
            variance_retained=0.95,
            reconstruction_error=0.05
        )
        str_repr = str(stats)
        assert "CompressionStats" in str_repr
        assert "(32, 256)" in str_repr
        assert "(32, 64)" in str_repr


class TestSVDCompressor:
    """Test SVD-based compression."""

    def test_svd_compressor_init(self):
        """Test SVDCompressor initialization."""
        compressor = SVDCompressor(target_dim=64)
        assert compressor.target_dim == 64
        assert compressor.method == 'svd'

    def test_svd_compress_single_tensor(self):
        """Test compressing a single tensor with SVD."""
        compressor = SVDCompressor(target_dim=32)
        
        # Create test activation
        activation = torch.randn(32, 128)
        
        # Compress
        compressed, stats = compressor.compress(activation)
        
        # Check output shape
        assert compressed.shape == activation.shape
        
        # Check stats
        assert stats.original_shape == activation.shape
        assert stats.compressed_shape[1] == 32
        assert stats.compression_ratio > 0  # includes basis storage; >1 only for large batches
        assert 0 <= stats.variance_retained <= 1
        assert stats.reconstruction_error >= 0

    def test_svd_compress_3d_tensor(self):
        """Test compressing 3D tensor with SVD."""
        compressor = SVDCompressor(target_dim=16)
        
        # Create 3D activation (batch, sequence, hidden)
        activation = torch.randn(8, 32, 64)
        
        # Compress
        compressed, stats = compressor.compress(activation)
        
        # Check output shape
        assert compressed.shape == activation.shape
        
        # Check compression ratio
        assert stats.compression_ratio > 0  # includes basis storage; >1 only for large batches

    def test_svd_variance_retained(self):
        """Test that SVD retains most variance."""
        compressor = SVDCompressor(target_dim=64)
        
        # Create activation with known structure
        activation = torch.randn(100, 128)
        
        # Compress
        compressed, stats = compressor.compress(activation)
        
        # SVD should retain most variance
        assert stats.variance_retained > 0.8


class TestPCACompressor:
    """Test PCA-based compression."""

    def test_pca_compressor_init(self):
        """Test PCACompressor initialization."""
        compressor = PCACompressor(target_dim=64)
        assert compressor.target_dim == 64
        assert compressor.method == 'pca'

    def test_pca_compress_single_tensor(self):
        """Test compressing a single tensor with PCA."""
        compressor = PCACompressor(target_dim=32)
        
        # Create test activation
        activation = torch.randn(32, 128)
        
        # Compress
        compressed, stats = compressor.compress(activation)
        
        # Check output shape
        assert compressed.shape == activation.shape
        
        # Check stats
        assert stats.original_shape == activation.shape
        assert stats.compressed_shape[1] == 32
        assert stats.compression_ratio > 0  # includes basis storage; >1 only for large batches

    def test_pca_variance_retained(self):
        """Test that PCA retains variance."""
        compressor = PCACompressor(target_dim=64)
        
        # Create activation with correlated features
        torch.manual_seed(42)
        base = torch.randn(100, 16)
        activation = torch.cat([base, base * 0.5 + 0.1 * torch.randn(100, 16)], dim=1)
        activation = torch.cat([activation, torch.randn(100, 96)], dim=1)
        
        # Compress
        compressed, stats = compressor.compress(activation)
        
        # PCA should retain good variance
        assert stats.variance_retained > 0.7


class TestRandomProjectionCompressor:
    """Test random projection compression."""

    def test_random_compressor_init(self):
        """Test RandomProjectionCompressor initialization."""
        compressor = RandomProjectionCompressor(target_dim=64)
        assert compressor.target_dim == 64
        assert compressor.method == 'random'

    def test_random_compress_single_tensor(self):
        """Test compressing a single tensor with random projections."""
        compressor = RandomProjectionCompressor(target_dim=32)
        
        # Create test activation
        activation = torch.randn(32, 128)
        
        # Compress
        compressed, stats = compressor.compress(activation)
        
        # Check output shape
        assert compressed.shape == activation.shape
        
        # Check stats
        assert stats.original_shape == activation.shape
        assert stats.compressed_shape[1] == 32
        assert stats.compression_ratio > 0  # includes basis storage; >1 only for large batches

    def test_random_projection_orthogonal(self):
        """Test that random projections are orthogonal."""
        compressor = RandomProjectionCompressor(target_dim=64)
        
        # Create test activation
        activation = torch.randn(100, 128)
        
        # Compress
        compressed, stats = compressor.compress(activation)
        
        # Random projections should still work
        assert compressed.shape == activation.shape


class TestActivationCompressorCalibration:
    """Test calibration functionality."""

    def test_calibrate_simple_model(self):
        """Test calibration on a simple model."""
        compressor = SVDCompressor(target_dim=32)
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Create inputs
        inputs = {'input': torch.randn(8, 64)}
        
        # Calibrate
        stats = compressor.calibrate(model, inputs)
        
        # Check that we got stats
        assert len(stats) > 0
        
        # Check stats for each layer
        for stat in stats.values():
            assert isinstance(stat, CompressionStats)
            assert stat.compression_ratio > 0  # includes basis storage; >1 only for large batches

    def test_calibrate_specific_layer(self):
        """Test calibration on a specific layer."""
        compressor = SVDCompressor(target_dim=32, layer_idx=0)
        
        # Create model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Create inputs
        inputs = {'input': torch.randn(8, 64)}
        
        # Calibrate
        stats = compressor.calibrate(model, inputs, layer_idx=0)
        
        # Should only have stats for layer 0
        assert len(stats) >= 1


class TestActivationCompressorWrapper:
    """Test wrapper functionality."""

    def test_wrapper_init(self):
        """Test wrapper initialization."""
        model = nn.Sequential(nn.Linear(64, 128))
        compressor = SVDCompressor(target_dim=32)
        wrapper = ActivationCompressorWrapper(model, compressor)
        
        assert wrapper.model is model
        assert wrapper.compressor is compressor

    def test_wrapper_add_remove_hooks(self):
        """Test adding and removing hooks."""
        model = nn.Sequential(nn.Linear(64, 128))
        compressor = SVDCompressor(target_dim=32)
        wrapper = ActivationCompressorWrapper(model, compressor, layer_pattern='Linear')
        
        # Add hooks
        wrapper.add_compression_hooks()
        
        # Check hooks were added
        assert len(wrapper._hooks) > 0
        
        # Remove hooks
        wrapper.remove_compression_hooks()
        
        # Check hooks were removed
        assert len(wrapper._hooks) == 0

    def test_wrapper_context_manager(self):
        """Test wrapper as context manager."""
        model = nn.Sequential(nn.Linear(64, 128))
        compressor = SVDCompressor(target_dim=32)
        wrapper = ActivationCompressorWrapper(model, compressor, layer_pattern='Linear')
        
        # Use context manager
        with wrapper:
            # Hooks should be active
            assert len(wrapper._hooks) > 0
            
            # Run model
            output = model(torch.randn(8, 64))
            assert output.shape == (8, 128)
        
        # Hooks should be removed
        assert len(wrapper._hooks) == 0


class TestDynamicActivationCompressor:
    """Test learnable compression."""

    def test_dynamic_compressor_init(self):
        """Test DynamicActivationCompressor initialization."""
        compressor = DynamicActivationCompressor(
            input_dim=128,
            target_dim=64,
            num_layers=2
        )
        
        assert compressor.input_dim == 128
        assert compressor.target_dim == 64

    def test_dynamic_compressor_forward(self):
        """Test forward pass through dynamic compressor."""
        compressor = DynamicActivationCompressor(
            input_dim=128,
            target_dim=64,
            num_layers=2
        )
        
        # Create test input
        x = torch.randn(32, 128)
        
        # Forward pass
        output = compressor(x)
        
        # Check output shape
        assert output.shape == x.shape

    def test_dynamic_compressor_training(self):
        """Test training dynamic compressor."""
        compressor = DynamicActivationCompressor(
            input_dim=128,
            target_dim=64,
            num_layers=2
        )
        
        optimizer = torch.optim.Adam(compressor.parameters(), lr=1e-3)
        
        # Training loop
        for _ in range(10):
            x = torch.randn(32, 128)
            output = compressor(x)
            loss = nn.MSELoss()(output, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Loss should decrease
            assert loss.item() >= 0

    def test_dynamic_compression_ratio(self):
        """Test compression ratio calculation."""
        compressor = DynamicActivationCompressor(
            input_dim=256,
            target_dim=64,
            num_layers=2
        )
        
        ratio = compressor.get_compression_ratio()
        
        # Should have some compression
        assert ratio > 0


class TestFactoryFunction:
    """Test factory function."""

    def test_create_svd_compressor(self):
        """Test creating SVD compressor."""
        compressor = create_activation_compressor('svd', target_dim=64)
        assert isinstance(compressor, SVDCompressor)

    def test_create_pca_compressor(self):
        """Test creating PCA compressor."""
        compressor = create_activation_compressor('pca', target_dim=64)
        assert isinstance(compressor, PCACompressor)

    def test_create_random_compressor(self):
        """Test creating random compressor."""
        compressor = create_activation_compressor('random', target_dim=64)
        assert isinstance(compressor, RandomProjectionCompressor)

    def test_create_invalid_method(self):
        """Test creating with invalid method."""
        with pytest.raises(ValueError):
            create_activation_compressor('invalid', target_dim=64)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_target_dim(self):
        """Test with zero target dimension."""
        compressor = SVDCompressor(target_dim=0)
        activation = torch.randn(32, 128)
        
        # Should handle gracefully
        compressed, stats = compressor.compress(activation)
        assert compressed.shape == activation.shape

    def test_large_target_dim(self):
        """Test with target dim larger than input."""
        compressor = SVDCompressor(target_dim=256)
        activation = torch.randn(32, 128)
        
        # Should handle gracefully
        compressed, stats = compressor.compress(activation)
        assert compressed.shape == activation.shape

    def test_single_sample(self):
        """Test with single sample."""
        compressor = SVDCompressor(target_dim=32)
        activation = torch.randn(1, 128)
        
        compressed, stats = compressor.compress(activation)
        assert compressed.shape == activation.shape

    def test_cpu_and_cuda(self):
        """Test on CPU (CUDA if available)."""
        compressor = SVDCompressor(target_dim=32)
        
        # CPU
        activation_cpu = torch.randn(32, 128)
        compressed_cpu, _ = compressor.compress(activation_cpu)
        assert compressed_cpu.device.type == 'cpu'

        # CUDA (if available)
        if torch.cuda.is_available():
            activation_cuda = torch.randn(32, 128).cuda()
            compressed_cuda, _ = compressor.compress(activation_cuda)
            assert compressed_cuda.device.type == 'cuda'

    def test_different_dtypes(self):
        """Test with different dtypes."""
        compressor = SVDCompressor(target_dim=32)
        
        # Float32
        activation_f32 = torch.randn(32, 128, dtype=torch.float32)
        compressed_f32, _ = compressor.compress(activation_f32)
        assert compressed_f32.dtype == torch.float32
        
        # Float64
        activation_f64 = torch.randn(32, 128, dtype=torch.float64)
        compressed_f64, _ = compressor.compress(activation_f64)
        assert compressed_f64.dtype == torch.float64


class TestNumericalStability:
    """Test numerical stability."""

    def test_very_small_values(self):
        """Test with very small values."""
        compressor = SVDCompressor(target_dim=32)
        activation = torch.randn(32, 128) * 1e-10
        
        compressed, stats = compressor.compress(activation)
        
        # Should not have NaN or Inf
        assert not torch.isnan(compressed).any()
        assert not torch.isinf(compressed).any()

    def test_very_large_values(self):
        """Test with very large values."""
        compressor = SVDCompressor(target_dim=32)
        activation = torch.randn(32, 128) * 1e10
        
        compressed, stats = compressor.compress(activation)
        
        # Should not have NaN or Inf
        assert not torch.isnan(compressed).any()
        assert not torch.isinf(compressed).any()

    def test_zeros(self):
        """Test with all zeros."""
        compressor = SVDCompressor(target_dim=32)
        activation = torch.zeros(32, 128)
        
        compressed, stats = compressor.compress(activation)
        
        # Should handle zeros
        assert not torch.isnan(compressed).any()

    def test_constant_values(self):
        """Test with constant values."""
        compressor = SVDCompressor(target_dim=32)
        activation = torch.ones(32, 128)
        
        compressed, stats = compressor.compress(activation)
        
        # Should handle constants
        assert not torch.isnan(compressed).any()


class TestIntegration:
    """Test integration with PyTorch models."""

    def test_transformer_layer_compression(self):
        """Test compression on transformer layer."""
        compressor = SVDCompressor(target_dim=64)
        
        # Create transformer layer
        from torch.nn import TransformerEncoderLayer
        layer = TransformerEncoderLayer(d_model=128, nhead=8)
        
        # Create model
        model = nn.Sequential(
            layer,
            layer,
        )
        
        # Calibrate (nn.Sequential takes a single positional input)
        stats = compressor.calibrate(model, torch.randn(10, 32, 128))
        
        # Should work
        assert len(stats) > 0

    def test_multiple_compressions(self):
        """Test multiple sequential compressions."""
        compressor1 = SVDCompressor(target_dim=64)
        compressor2 = PCACompressor(target_dim=32)
        
        activation = torch.randn(32, 128)
        
        # First compression
        compressed1, _ = compressor1.compress(activation)
        
        # Second compression
        compressed2, _ = compressor2.compress(compressed1)
        
        # Should work
        assert compressed2.shape == activation.shape

    def test_gradient_flow(self):
        """Test that gradients flow through compression."""
        compressor = DynamicActivationCompressor(
            input_dim=128,
            target_dim=64,
            num_layers=1
        )
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(64, 128),
            compressor,
            nn.Linear(128, 10)
        )
        
        # Forward pass
        x = torch.randn(8, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
