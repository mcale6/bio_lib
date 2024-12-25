"""Bio-lib: A library for biological computations with JAX support."""
from src.common.protein import Protein
from src.common.residue_library import default_library as residue_library
from src.common.residue_classification import ResidueClassification, ResidueCharacter
from src.custom_prodigy_jax import run as run_prodigy
from src.shrake_rupley_jax import calculate_sasa
import pkg_resources

# Version
__version__ = "0.1.0"

# Data path helper
def get_data_path(filename: str) -> str:
    return pkg_resources.resource_filename('bio_lib', f'data/{filename}')

# Public API
__all__ = [
    'Protein',
    'residue_library',
    'run_prodigy',
    'calculate_sasa',
    'get_data_path',
    'ResidueClassification',
    'ResidueCharacter',
]