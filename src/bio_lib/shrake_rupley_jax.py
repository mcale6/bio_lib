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

def calculate_sasa_optimized(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    sphere_points: jnp.ndarray = SPHERE_POINTS,
    probe_radius: float = 1.4,
    max_neighbors: int = 200
) -> jnp.ndarray:
    """Optimized SASA calculation using neighbor lists and JAX loops."""
    # ===== 1. Preprocessing =====
    N = coords.shape[0]
    M = sphere_points.shape[0]
    
    # Mask invalid atoms
    masked_coords = coords * mask[:, None]
    masked_radii = vdw_radii * mask
    radii_with_probe = (masked_radii + probe_radius) * mask
    
    # Precompute squared radii with probe
    radii_with_probe_sq = (radii_with_probe ** 2).astype(coords.dtype)
    
    # ===== 2. Compute Interaction Matrix (Neighbor List) =====
    # Pairwise distances between all atoms
    diff = masked_coords[:, None, :] - masked_coords[None, :, :]
    dist2 = jnp.sum(diff ** 2, axis=-1)
    
    # Compute interaction cutoff matrix
    radsum = radii_with_probe[:, None] + radii_with_probe[None, :]
    interaction_mask = (dist2 <= (radsum ** 2)) & ~jnp.eye(N, dtype=bool)
    
    # ===== 3. Process Each Atom's Neighbors =====
    def process_atom(i):
        # Generate sphere points for atom i
        atom_radius = radii_with_probe[i]
        points = masked_coords[i] + sphere_points * atom_radius  # [M, 3]
        
        # Find neighbors for atom i
        neighbors = jnp.where(interaction_mask[i], size=max_neighbors, fill_value=-1)[0]
        
        # Check burial for each point
        def check_point(point):
            # Compute distances to neighbors
            neighbor_coords = masked_coords[neighbors]
            neighbor_radii_sq = radii_with_probe_sq[neighbors]
            
            # Calculate distances and check burial
            dist2_neighbors = jnp.sum((point - neighbor_coords) ** 2, axis=-1)
            is_buried = jnp.any((dist2_neighbors <= neighbor_radii_sq) & (neighbors != -1))
            return ~is_buried
        
        # Vectorize over sphere points
        accessible = jnp.sum(lax.map(check_point, points))
        return 4.0 * jnp.pi * (atom_radius ** 2) * (accessible / M)
    
    # Compute SASA for all atoms
    sasa_values = lax.map(process_atom, jnp.arange(N))
    return sasa_values * mask