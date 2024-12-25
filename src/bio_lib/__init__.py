"""Bio-lib: A library for biological computations with JAX support."""
from bio_lib.common.protein import Protein
from bio_lib.common.residue_library import default_library as residue_library
from bio_lib.common.residue_classification import ResidueClassification, ResidueCharacter
from bio_lib.custom_prodigy_jax import run as run_prodigy
from bio_lib.shrake_rupley_jax import calculate_sasa
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