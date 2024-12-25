import jax.numpy as jnp
from jax import jit
import numpy as np
import pkg_resources

def _load_sphere_points() -> np.ndarray:
    """Load sphere points from package data."""
    sphere_path = pkg_resources.resource_filename('bio_lib', 'data/thomson15092.xyz')
    return np.loadtxt(sphere_path, skiprows=1)

# Create constant
SPHERE_POINTS = jnp.array(_load_sphere_points())

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

    # Interaction matrix: check for overlapping atoms
    radii_with_probe = masked_radii + probe_radius
    diff = masked_coords[:, None, :] - masked_coords[None, :, :]  # Pairwise differences
    dist2 = jnp.sum(diff ** 2, axis=-1)  # Squared distances
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    interaction_matrix = (dist2 <= radsum2) & ~jnp.eye(coords.shape[0], dtype=bool)

    # SASA calculation
    scaled_points = sphere_points[None, :, :] * radii_with_probe[:, None, None] + masked_coords[:, None, :]
    diff = scaled_points[:, :, None, :] - masked_coords[None, None, :, :]  # [N, M, N, 3]
    dist2 = jnp.sum(diff ** 2, axis=-1)  # [N, M, N]
    is_buried = (dist2 <= radii_with_probe[None, None, :] ** 2) & interaction_matrix[:, None, :]
    buried_points = jnp.any(is_buried, axis=-1)  # [N, M]
    n_accessible = sphere_points.shape[0] - jnp.sum(buried_points, axis=-1)  # [N]

    # Surface area per atom
    areas = 4.0 * jnp.pi * (radii_with_probe ** 2)
    sasa = areas * (n_accessible / sphere_points.shape[0])

    return sasa