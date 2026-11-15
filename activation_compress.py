"""
ActivationCompress: Neural Network Activation Compression Suite

Provides SVD-based, PCA, and random projection methods for compressing
neural network activations during forward passes. Enables efficient
memory usage and computational savings while preserving model behavior.

Key Features:
- Runtime SVD bottleneck for activations
- Multiple compression methods (SVD, PCA, Random Projections)
- Memory-efficient chunked processing
- Integration with PyTorch models via hooks
- Compression ratio analysis and visualization
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Callable, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class CompressionStats:
    """Statistics for activation compression."""
    original_shape: Tuple[int, ...]
    compressed_shape: Tuple[int, ...]
    compression_ratio: float
    variance_retained: float
    reconstruction_error: float

    def __str__(self) -> str:
        return (
            f"CompressionStats(original={self.original_shape}, "
            f"compressed={self.compressed_shape}, "
            f"ratio={self.compression_ratio:.2f}x, "
            f"variance={self.variance_retained:.2%}, "
            f"error={self.reconstruction_error:.4f})"
        )


class ActivationCompressor(nn.Module):
    """
    Base class for activation compression.

    Provides infrastructure for compressing activations during forward passes
    using various dimensionality reduction techniques.

    Args:
        target_dim: Target dimension after compression
        method: Compression method ('svd', 'pca', 'random')
        layer_idx: Layer index to compress (if None, compresses all)
    """

    def __init__(
        self,
        target_dim: int = 64,
        method: str = 'svd',
        layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.target_dim = target_dim
        self.method = method.lower()
        self.layer_idx = layer_idx
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._basis: Optional[torch.Tensor] = None
        self._basis_dict: Dict[int, torch.Tensor] = {}
        self._stats: Dict[str, CompressionStats] = {}

    def _validate_method(self) -> None:
        """Validate compression method."""
        valid_methods = {'svd', 'pca', 'random', 'autoencoder'}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self.method}'. Must be one of {valid_methods}"
            )

    def _get_basis(
        self,
        activation: torch.Tensor,
        target_dim: int
    ) -> torch.Tensor:
        """
        Compute compression basis from activation.

        Args:
            activation: Input activation tensor
            target_dim: Target dimension

        Returns:
            Basis tensor for projection
        """
        if self.method == 'svd':
            return self._svd_basis(activation, target_dim)
        elif self.method == 'pca':
            return self._pca_basis(activation, target_dim)
        elif self.method == 'random':
            return self._random_basis(activation, target_dim)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _svd_basis(
        self,
        activation: torch.Tensor,
        target_dim: int
    ) -> torch.Tensor:
        """Compute SVD basis (top-k right singular vectors)."""
        # Flatten activation to 2D
        flat = activation.reshape(-1, activation.shape[-1]).float()

        # Compute SVD
        _, _, vt = torch.linalg.svd(flat, full_matrices=False)

        # Return top-k right singular vectors
        return vt[:target_dim].T  # (D, k)

    def _pca_basis(
        self,
        activation: torch.Tensor,
        target_dim: int
    ) -> torch.Tensor:
        """Compute PCA basis (eigenvectors of covariance)."""
        flat = activation.reshape(-1, activation.shape[-1]).float()

        # Center the data
        mean = flat.mean(dim=0)
        centered = flat - mean

        # Compute covariance
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)

        # Eigen decomposition
        _, vecs = torch.linalg.eigh(cov)

        # Return top-k eigenvectors
        return vecs[:, -target_dim:].flip(dims=[1])

    def _random_basis(
        self,
        activation: torch.Tensor,
        target_dim: int
    ) -> torch.Tensor:
        """Generate random projection basis."""
        hidden_dim = activation.shape[-1]
        basis = torch.randn(hidden_dim, target_dim)
        # Orthogonalize
        q, _ = torch.linalg.qr(basis)
        return q

    def calibrate(
        self,
        model: nn.Module,
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        layer_idx: Optional[int] = None
    ) -> Dict[str, CompressionStats]:
        """
        Calibrate compression basis from model activations.

        Args:
            model: PyTorch model
            inputs: Input tensors for calibration
            layer_idx: Specific layer to calibrate (overrides self.layer_idx)

        Returns:
            Dictionary of compression statistics per layer
        """
        self._validate_method()
        idx = layer_idx if layer_idx is not None else self.layer_idx

        # Collect activations
        activations: Dict[int, torch.Tensor] = {}

        def hook_fn(module, inp, out):
            # Store activation
            layer_id = len(activations)
            activations[layer_id] = out.detach().float().clone()
            return out

        # Register hooks on all layers
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                h = module.register_forward_hook(hook_fn)
                hooks.append(h)

        # Forward pass
        with torch.no_grad():
            if isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute basis for each layer
        self._basis_dict: Dict[int, torch.Tensor] = {}
        stats: Dict[str, CompressionStats] = {}

        for layer_id, act in activations.items():
            if idx is not None and layer_id != idx:
                continue

            original_shape = act.shape
            flat_act = act.reshape(-1, act.shape[-1])

            # Compute basis
            basis = self._get_basis(act, self.target_dim)
            self._basis_dict[layer_id] = basis

            # Project
            codes = flat_act @ basis  # (N, k)
            reconstructed = codes @ basis.T  # (N, D)

            # Compute stats
            original_norm = (flat_act ** 2).sum()
            recon_norm = (reconstructed ** 2).sum()
            error = ((flat_act - reconstructed) ** 2).sum()

            variance_retained = min(1.0, float(recon_norm) / (float(original_norm) + 1e-9))
            compression_ratio = (act.numel() / (codes.numel() + basis.numel()))

            stats[f"layer_{layer_id}"] = CompressionStats(
                original_shape=original_shape,
                compressed_shape=codes.shape,
                compression_ratio=compression_ratio,
                variance_retained=variance_retained,
                reconstruction_error=error.item()
            )

        self._stats = stats
        return stats

    def create_hook(
        self,
        layer_id: int
    ) -> Callable:
        """
        Create a forward hook for compression.

        Args:
            layer_id: Layer identifier

        Returns:
            Hook function
        """
        basis = getattr(self, '_basis_dict', {}).get(layer_id)

        def hook(module, inp, out):
            if basis is None:
                # Layer was never calibrated: pass the activation through
                # unchanged instead of crashing the forward pass.
                return out
            x = out.float()
            # Project to compressed space
            batch_dim = x.shape[:-1]
            x_flat = x.reshape(-1, x.shape[-1])
            codes = x_flat @ basis
            reconstructed = codes @ basis.T
            # Reshape back
            return reconstructed.reshape(*batch_dim, -1).to(out.dtype)

        return hook

    def compress(
        self,
        activation: torch.Tensor
    ) -> Tuple[torch.Tensor, CompressionStats]:
        """
        Compress a single activation tensor.

        Args:
            activation: Input activation tensor

        Returns:
            Tuple of (compressed_activation, stats)
        """
        if self.target_dim <= 0:
            # Degenerate config: pass through unchanged (documented behavior).
            stats = CompressionStats(
                original_shape=activation.shape,
                compressed_shape=activation.shape,
                compression_ratio=1.0,
                variance_retained=1.0,
                reconstruction_error=0.0,
            )
            return activation.clone(), stats

        original_shape = activation.shape
        flat = activation.reshape(-1, activation.shape[-1]).float()

        # Compute basis
        basis = self._get_basis(activation, self.target_dim)

        # Project
        codes = flat @ basis
        reconstructed = codes @ basis.T

        # Compute stats
        original_norm = (flat ** 2).sum()
        recon_norm = (reconstructed ** 2).sum()
        error = ((flat - reconstructed) ** 2).sum()

        stats = CompressionStats(
            original_shape=original_shape,
            compressed_shape=codes.shape,
            compression_ratio=activation.numel() / (codes.numel() + basis.numel()),
            variance_retained=min(1.0, recon_norm.item() / (original_norm.item() + 1e-9)),
            reconstruction_error=error.item()
        )

        out = reconstructed.reshape(*original_shape[:-1], -1).to(activation.dtype)
        return out, stats

    def get_stats(self) -> Dict[str, CompressionStats]:
        """Get compression statistics."""
        return self._stats

    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []


class SVDCompressor(ActivationCompressor):
    """
    SVD-based activation compressor.

    Uses Singular Value Decomposition to find the most significant
    directions in activation space and projects activations onto them.

    Args:
        target_dim: Number of singular vectors to keep
        layer_idx: Specific layer to compress
    """

    def __init__(
        self,
        target_dim: int = 64,
        layer_idx: Optional[int] = None
    ):
        super().__init__(target_dim=target_dim, method='svd', layer_idx=layer_idx)


class PCACompressor(ActivationCompressor):
    """
    PCA-based activation compressor.

    Uses Principal Component Analysis to find directions of
    maximum variance in activation space.

    Args:
        target_dim: Number of principal components to keep
        layer_idx: Specific layer to compress
    """

    def __init__(
        self,
        target_dim: int = 64,
        layer_idx: Optional[int] = None
    ):
        super().__init__(target_dim=target_dim, method='pca', layer_idx=layer_idx)


class RandomProjectionCompressor(ActivationCompressor):
    """
    Random projection-based activation compressor.

    Uses random orthogonal projections to compress activations.
    More efficient than SVD/PCA but may lose more information.

    Args:
        target_dim: Projection dimension
        layer_idx: Specific layer to compress
    """

    def __init__(
        self,
        target_dim: int = 64,
        layer_idx: Optional[int] = None
    ):
        super().__init__(target_dim=target_dim, method='random', layer_idx=layer_idx)


class ActivationCompressorWrapper:
    """
    Wrapper for easy integration with PyTorch models.

    Simplifies the process of adding compression hooks to models.

    Args:
        model: PyTorch model to compress
        compressor: ActivationCompressor instance
        layer_pattern: Pattern to match layer names
    """

    def __init__(
        self,
        model: nn.Module,
        compressor: ActivationCompressor,
        layer_pattern: str = 'layers'
    ):
        self.model = model
        self.compressor = compressor
        self.layer_pattern = layer_pattern
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def add_compression_hooks(self) -> None:
        """Add compression hooks to matching layers."""
        self._hooks = []

        matched = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and (
                self.layer_pattern in name
                or self.layer_pattern in type(module).__name__
            ):
                hook = module.register_forward_hook(
                    self.compressor.create_hook(matched)
                )
                self._hooks.append(hook)
                matched += 1

    def remove_compression_hooks(self) -> None:
        """Remove all compression hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __enter__(self):
        """Context manager entry."""
        self.add_compression_hooks()
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.remove_compression_hooks()


def create_activation_compressor(
    method: str = 'svd',
    target_dim: int = 64,
    layer_idx: Optional[int] = None
) -> ActivationCompressor:
    """
    Factory function to create activation compressors.

    Args:
        method: Compression method ('svd', 'pca', 'random')
        target_dim: Target dimension
        layer_idx: Layer index

    Returns:
        ActivationCompressor instance
    """
    method = method.lower()

    if method == 'svd':
        return SVDCompressor(target_dim=target_dim, layer_idx=layer_idx)
    elif method == 'pca':
        return PCACompressor(target_dim=target_dim, layer_idx=layer_idx)
    elif method == 'random':
        return RandomProjectionCompressor(target_dim=target_dim, layer_idx=layer_idx)
    else:
        raise ValueError(f"Unknown method: {method}")


class DynamicActivationCompressor(nn.Module):
    """
    Dynamic activation compressor with learnable basis.

    Learns optimal compression basis during training.

    Args:
        input_dim: Input dimension
        target_dim: Target compressed dimension
        num_layers: Number of compression layers
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim

        # Learnable compression network
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else target_dim
            out_dim = target_dim if i == num_layers - 1 else target_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())

        self.compressor = nn.Sequential(*layers)

        # Learnable decompressor
        layers = []
        for i in range(num_layers):
            in_dim = target_dim if i == 0 else target_dim
            out_dim = input_dim if i == num_layers - 1 else target_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())

        self.decompressor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress and decompress activation.

        Args:
            x: Input activation tensor

        Returns:
            Reconstructed activation
        """
        # Compress
        batch_dim = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)
        compressed = self.compressor(x_flat)

        # Decompress
        reconstructed = self.decompressor(compressed)

        return reconstructed.reshape(*batch_dim, self.input_dim)

    def get_compression_ratio(self) -> float:
        """Get compression ratio."""
        params_compressor = sum(p.numel() for p in self.compressor.parameters())
        params_decompressor = sum(p.numel() for p in self.decompressor.parameters())
        total_params = params_compressor + params_decompressor

        return self.input_dim / (self.target_dim + total_params / self.input_dim)


__all__ = [
    'ActivationCompressor',
    'SVDCompressor',
    'PCACompressor',
    'RandomProjectionCompressor',
    'DynamicActivationCompressor',
    'ActivationCompressorWrapper',
    'CompressionStats',
    'create_activation_compressor',
]
