from pathlib import Path
from typing import List
from datetime import datetime
from typing import Any, Dict, Union
import jax.numpy as jnp
import numpy as np
import json

def convert_jax_arrays(obj: Any) -> Any:
    """Convert JAX arrays to native Python types recursively."""
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_jax_arrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_jax_arrays(x) for x in obj]
    return obj

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
def collect_pdb_files(input_path: Path) -> List[Path]:
    """Collect all PDB files from input path."""
    if input_path.suffix.lower() in ['.pdb', '.ent']:
        return [input_path]
    elif input_path.is_dir():
        # If directory, collect all PDB files
        pdb_files = list(input_path.glob('*.pdb')) + list(input_path.glob('*.ent'))
        if not pdb_files:
            raise ValueError(f"No PDB files found in directory: {input_path}")
        return sorted(pdb_files)
    else:
        raise ValueError(f"Input path must be a PDB file or directory, got: {input_path}")

def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.2f}s"

def setup_output_path(pdb_path: Path, output_dir: Path) -> Path:
    """Setup output directory and generate unique output filename."""
    # Create a subdirectory with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = output_dir / timestamp
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    base_name = pdb_path.stem
    return output_subdir / f"{base_name}_results.json"

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


def generate_sphere_points(n: int) -> jnp.ndarray:
    """ Generate approximately evenly distributed points on a unit sphere using golden spiral. """
    if n <= 0: return jnp.zeros((0, 3))
    if n == 1: return jnp.array([[0., 1., 0.]])
    
    i = jnp.arange(n, dtype=jnp.float32)
    y = 1. - (2. * i) / (n - 1)
    r = jnp.sqrt(1 - y**2)
    theta = jnp.pi * (3. - jnp.sqrt(5.)) * i
    
    return jnp.stack([r * jnp.cos(theta), y, r * jnp.sin(theta)], axis=1)