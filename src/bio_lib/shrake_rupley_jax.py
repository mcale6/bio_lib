import jax.numpy as jnp
from jax import jit
import numpy as np
import pkg_resources
from jax import lax

# Create constant
SPHERE_POINTS = jnp.array(np.loadtxt(pkg_resources.resource_filename('bio_lib', 'data/thomson1000.xyz') , skiprows=1))

# Each point comparison involved N=16,428 × 16,428 ≈ 270 million pairs
# If we assume ~100 neighbors per atom, it's only 16,428 × 100 ≈ 1.6 million pairs
@jit
def calculate_sasa(
    coords: jnp.ndarray, 
    vdw_radii: jnp.ndarray, 
    mask: jnp.ndarray, 
    sphere_points: jnp.ndarray = SPHERE_POINTS,
    probe_radius: float = 1.4
) -> jnp.ndarray:
    """
    Calculate the solvent-accessible surface area (SASA).
    
    Args:
        coords: [N, 3] Array of atom coordinates.
        vdw_radii: [N] Array of van der Waals radii.
        mask: [N] Binary mask (1 for valid atoms, 0 for masked).
        sphere_points: [M, 3] Predefined sphere points.
        probe_radius: Probe radius for SASA calculation.
        
    Returns:
        sasa: [N] Solvent-accessible surface area for each atom.
    """
    # Apply mask to coordinates and radii
    masked_coords = coords * mask[:, None]  # [N, 3]
    masked_radii = vdw_radii * mask         # [N]
    radii_with_probe = (masked_radii + probe_radius) * mask

    # Interaction matrix: check for overlapping atoms
    diff = masked_coords[:, None, :] - masked_coords[None, :, :]  # Pairwise differences
    dist2 = jnp.sum(diff ** 2, axis=-1)  # Squared distances
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    interaction_matrix = (dist2 <= radsum2) & ~jnp.eye(coords.shape[0], dtype=bool)

    # SASA calculation
    scaled_points = sphere_points[None, :, :] * radii_with_probe[:, None, None] + masked_coords[:, None, :]
    diff = scaled_points[:, :, None, :] - masked_coords[None, None, :, :]  # [N, M, N, 3] # TO DO too much memory 
    dist2 = jnp.sum(diff ** 2, axis=-1)  # [N, M, N]
    is_buried = (dist2 <= radii_with_probe[None, None, :] ** 2) & interaction_matrix[:, None, :]
    buried_points = jnp.any(is_buried, axis=-1)  # [N, M]
    n_accessible = sphere_points.shape[0] - jnp.sum(buried_points, axis=-1)  # [N]

    # Surface area per atom
    areas = 4.0 * jnp.pi * (radii_with_probe ** 2)
    sasa = areas * (n_accessible / sphere_points.shape[0])

    return sasa

def estimate_optimal_block_size(n_atoms: int) -> int:
    a_block = 6.8879e+02  # Amplitude
    b_block = -2.6156e-04  # Decay rate
    c_block = 17.4525  # Offset

    # Estimate block size using the exponential decay equation
    block_size = int(round(a_block * np.exp(b_block * n_atoms) + c_block))

    # Add tighter bounds to prevent memory issues
    max_block = min(
        250,  # Absolute maximum
        int(5000 / np.sqrt(n_atoms / 1000))  # Dynamic limit based on atom count
    )

    return max(5, min(block_size, max_block))

def estimate_time(n_atoms: int) -> float:
    a_time = 5.7156e-01  # Amplitude
    b_time = 1.7124e-04  # Growth rate
    c_time = -0.3772  # Offset
    return a_time * np.exp(b_time * n_atoms) + c_time

def calculate_sasa2(
    coords: jnp.ndarray, 
    vdw_radii: jnp.ndarray, 
    mask: jnp.ndarray, 
    block_size: jnp.ndarray,
    sphere_points: jnp.ndarray = SPHERE_POINTS,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """
    Calculate the solvent-accessible surface area (SASA).
    """
    # Apply mask to coordinates and radii
    masked_coords = coords * mask[:, None]  # [N, 3]
    masked_radii = vdw_radii * mask        # [N]
    radii_with_probe = (masked_radii + probe_radius) * mask  # [N]

    # Interaction matrix: check for overlapping atoms
    diff = masked_coords[:, None, :] - masked_coords[None, :, :]  # [N, N, 3]
    dist2 = jnp.sum(diff ** 2, axis=-1)  # [N, N]
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    interaction_matrix = (dist2 <= radsum2) & ~jnp.eye(coords.shape[0], dtype=bool)

    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]
    buried_points = jnp.zeros((n_atoms, n_points), dtype=bool)

    # Process atoms in blocks to reduce peak memory usage
    for start_idx in range(0, n_atoms, block_size):
        end_idx = min(start_idx + block_size, n_atoms)
        
        # Calculate scaled points for this block
        block_scaled_points = (sphere_points[None, :, :] * 
                             radii_with_probe[start_idx:end_idx, None, None] + 
                             masked_coords[start_idx:end_idx, None, :])  # [block, M, 3]
        
        # Calculate distances to all atoms using a more memory-efficient formulation
        # |a-b|² = |a|² + |b|² - 2⟨a,b⟩
        scaled_points_norm2 = jnp.sum(block_scaled_points**2, axis=-1)  # [block, M]
        coords_norm2 = jnp.sum(masked_coords**2, axis=-1)  # [N]
        
        # Compute dot product term efficiently
        dot_prod = jnp.einsum('bms,ns->bmn', 
                            block_scaled_points,  # [block, M, 3]
                            masked_coords)        # [N, 3]
        
        # Calculate distances
        dist2 = (scaled_points_norm2[:, :, None] + 
                coords_norm2[None, None, :] - 
                2 * dot_prod)  # [block, M, N]
        
        # Check which points are buried
        is_buried = (dist2 <= radii_with_probe[None, None, :]**2) & \
                   interaction_matrix[start_idx:end_idx, None, :]
        block_buried = jnp.any(is_buried, axis=-1)  # [block, M]
        
        # Update buried points for this block
        buried_points = buried_points.at[start_idx:end_idx].set(block_buried)

    # Calculate final SASA
    n_accessible = n_points - jnp.sum(buried_points, axis=-1)
    areas = 4.0 * jnp.pi * (radii_with_probe ** 2)
    sasa = areas * (n_accessible / n_points)

    return sasa