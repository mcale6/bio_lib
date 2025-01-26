"""Bio-lib: A library for biological computations with JAX support."""
from bio_lib.common.protein import Protein
from bio_lib.common.protein_jax import JaxProtein
from bio_lib.common.residue_library import default_library as residue_library
from bio_lib.common.residue_classification import ResidueClassification, ResidueCharacter
from bio_lib.shrake_rupley_jax import calculate_sasa
from bio_lib.helpers import tqdm
from bio_lib.helpers import utils
from bio_lib.custom_prodigy_jax import predict_binding_affinity_jax
from bio_lib.custom_prodigy import predict_binding_affinity
import pkg_resources
from .version import __version__

# Data path helper
def get_data_path(filename: str) -> str:
    return pkg_resources.resource_filename('bio_lib', f'data/{filename}')

# Public API
__all__ = [
    'Protein',
    'JaxProtein',
    "tqdm",
    "utils",
    'residue_library',
    'predict_binding_affinity_jax',
    'predict_binding_affinity,'
    'calculate_sasa',
    'get_data_path',
    'ResidueClassification',
    'ResidueCharacter',
]