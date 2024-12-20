import jax.numpy as jnp
from jax import jit
from functools import partial

@jit
def calculate_sasa(coords: jnp.ndarray, vdw_radii: jnp.ndarray, 
        mask: jnp.ndarray, sphere_points: jnp.ndarray) -> jnp.ndarray:
    n_points = len(sphere_points)
    probe_radius = jnp.array(1.4)
    #mask = jnp.ones_like(vdw_radii)  # All atoms valid

    # Apply mask directly (1=valid, 0=masked)
    masked_coords = coords * mask[:, None]  # [N, 3]
    masked_radii = vdw_radii * mask        # [N]

    # Compute interaction matrix
    radii = masked_radii + probe_radius
    diff = masked_coords[:, None, :] - masked_coords[None, :, :]
    dist2 = jnp.sum(diff * diff, axis=-1)
    radsum = radii[:, None] + radii[None, :]
    radsum2 = radsum * radsum
    interaction_matrix = (dist2 <= radsum2) & ~jnp.eye(coords.shape[0], dtype=bool)

    # Compute SASA
    radii = masked_radii + probe_radius
    scaled_points = sphere_points[None, :, :] * radii[:, None, None] + masked_coords[:, None, :]
    diff = scaled_points[:, :, None, :] - masked_coords[None, None, :, :]
    dist2 = jnp.sum(diff * diff, axis=-1)
    radii2 = jnp.square(masked_radii + probe_radius)
    is_buried = (dist2 <= radii2[None, None, :]) & interaction_matrix[:, None, :]
    buried_points = jnp.any(is_buried, axis=-1)
    n_accessible = n_points - jnp.sum(buried_points, axis=-1)
    areas = 4.0 * jnp.pi * jnp.square(radii)
    sasa = areas * (n_accessible / n_points)

    return sasa