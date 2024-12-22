import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np

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

class ShrakeRupleyCalculator:
    def __init__(self, probe_radius: float = 1.4, points_file: str = "thomson1000.xyz"):
        self._sphere_points = jnp.array(np.loadtxt(points_file, skiprows=1))
        self.n_points = len(self._sphere_points)
        self.probe_radius = jnp.array(probe_radius)

    @partial(jit, static_argnums=(0,))
    def _compute_interaction_matrix(self, coords: jnp.ndarray, vdw_radii: jnp.ndarray) -> jnp.ndarray:
        radii = vdw_radii + self.probe_radius

        diff = coords[:, None, :] - coords[None, :, :]
        dist2 = jnp.sum(diff * diff, axis=-1)

        radsum = radii[:, None] + radii[None, :]
        radsum2 = radsum * radsum

        result = (dist2 <= radsum2) & ~jnp.eye(coords.shape[0], dtype=bool)
        return result

    @partial(jit, static_argnums=(0,))
    def _compute_all_atom_sasa(self, coords: jnp.ndarray, vdw_radii: jnp.ndarray,
                               interaction_matrix: jnp.ndarray) -> jnp.ndarray:
        radii = vdw_radii + self.probe_radius

        # Expand sphere points for all atoms
        scaled_points = self._sphere_points[None, :, :] * radii[:, None, None] + coords[:, None, :]

        # Compute distances from all scaled points to all atoms
        diff = scaled_points[:, :, None, :] - coords[None, None, :, :]
        dist2 = jnp.sum(diff * diff, axis=-1)

        # Compare against squared radii
        radii2 = jnp.square(vdw_radii + self.probe_radius)
        is_buried = (dist2 <= radii2[None, None, :]) & interaction_matrix[:, None, :]

        # Determine buried points for each atom
        buried_points = jnp.any(is_buried, axis=-1)
        n_accessible = self.n_points - jnp.sum(buried_points, axis=-1)

        # Calculate SASA
        areas = 4.0 * jnp.pi * jnp.square(radii)
        sasa = areas * (n_accessible / self.n_points)

        return sasa

    @partial(jit, static_argnums=(0,))
    def calculate_all(self, coords: jnp.ndarray, vdw_radii: jnp.ndarray,
                      mask: jnp.ndarray = None) -> jnp.ndarray:
        """
        mask: [N] mask (1 = valid atom, 0 = masked/padding in AF2)
        """
        if mask is None:
            mask = jnp.ones_like(vdw_radii)  # All atoms valid

        # Apply mask directly (1=valid, 0=masked)
        masked_coords = coords * mask[:, None]  # [N, 3]
        masked_radii = vdw_radii * mask        # [N]

        # Calculate with masked values
        interaction_matrix = self._compute_interaction_matrix(masked_coords, masked_radii)
        sasa = self._compute_all_atom_sasa(masked_coords, masked_radii, interaction_matrix)

        return sasa
    

class ShrakeRupleyCalculator:
    def __init__(self, probe_radius: float = 1.4, points_file: str = "thomson1000.xyz"):
        self._sphere_points = jnp.array(np.loadtxt(points_file, skiprows=1))
        self.n_points = len(self._sphere_points)
        self.probe_radius = jnp.array(probe_radius)

    @partial(jit, static_argnums=(0,))
    def _compute_interaction_matrix(self, coords: jnp.ndarray, vdw_radii: jnp.ndarray) -> jnp.ndarray:
        radii = vdw_radii + self.probe_radius

        diff = coords[:, None, :] - coords[None, :, :]
        dist2 = jnp.sum(diff * diff, axis=-1)

        radsum = radii[:, None] + radii[None, :]
        radsum2 = radsum * radsum

        result = (dist2 <= radsum2) & ~jnp.eye(coords.shape[0], dtype=bool)
        return result

    @partial(jit, static_argnums=(0,))
    def _compute_all_atom_sasa(self, coords: jnp.ndarray, vdw_radii: jnp.ndarray,
                               interaction_matrix: jnp.ndarray) -> jnp.ndarray:
        radii = vdw_radii + self.probe_radius

        # Expand sphere points for all atoms
        scaled_points = self._sphere_points[None, :, :] * radii[:, None, None] + coords[:, None, :]

        # Compute distances from all scaled points to all atoms
        diff = scaled_points[:, :, None, :] - coords[None, None, :, :]
        dist2 = jnp.sum(diff * diff, axis=-1)

        # Compare against squared radii
        radii2 = jnp.square(vdw_radii + self.probe_radius)
        is_buried = (dist2 <= radii2[None, None, :]) & interaction_matrix[:, None, :]

        # Determine buried points for each atom
        buried_points = jnp.any(is_buried, axis=-1)
        n_accessible = self.n_points - jnp.sum(buried_points, axis=-1)

        # Calculate SASA
        areas = 4.0 * jnp.pi * jnp.square(radii)
        sasa = areas * (n_accessible / self.n_points)

        return sasa

    @partial(jit, static_argnums=(0,))
    def calculate_all(self, coords: jnp.ndarray, vdw_radii: jnp.ndarray,
                      mask: jnp.ndarray = None) -> jnp.ndarray:
        """
        mask: [N] mask (1 = valid atom, 0 = masked/padding in AF2)
        """
        if mask is None:
            mask = jnp.ones_like(vdw_radii)  # All atoms valid

        # Apply mask directly (1=valid, 0=masked)
        masked_coords = coords * mask[:, None]  # [N, 3]
        masked_radii = vdw_radii * mask        # [N]

        # Calculate with masked values
        interaction_matrix = self._compute_interaction_matrix(masked_coords, masked_radii)
        sasa = self._compute_all_atom_sasa(masked_coords, masked_radii, interaction_matrix)

        return sasa