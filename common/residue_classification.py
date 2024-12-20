
import numpy as np
from typing import *
from enum import Enum
import residue_constants

class RelativeASAReference(NamedTuple):
    """Reference values for relative ASA calculations."""
    total: float
    backbone: float
    sidechain: float

class ResidueCharacter(str, Enum):
    """Enum defining possible residue characters."""
    ALIPHATIC = "A"
    POLAR = "P"
    CHARGED = "C"

class ResidueClassification:
    """Handles residue classifications, properties, and SASA calculations."""
    def __init__(self, classification_type: Literal["ic", "protorp"] = "protorp"):
        # Character classifications
        self.aa_character = {
            "ic": {
                "ALA": ResidueCharacter.ALIPHATIC, "CYS": ResidueCharacter.ALIPHATIC,
                "GLU": ResidueCharacter.CHARGED, "ASP": ResidueCharacter.CHARGED,
                "GLY": ResidueCharacter.ALIPHATIC, "PHE": ResidueCharacter.ALIPHATIC,
                "ILE": ResidueCharacter.ALIPHATIC, "HIS": ResidueCharacter.CHARGED,
                "LYS": ResidueCharacter.CHARGED, "MET": ResidueCharacter.ALIPHATIC,
                "LEU": ResidueCharacter.ALIPHATIC, "ASN": ResidueCharacter.POLAR,
                "GLN": ResidueCharacter.POLAR, "PRO": ResidueCharacter.ALIPHATIC,
                "SER": ResidueCharacter.POLAR, "ARG": ResidueCharacter.CHARGED,
                "THR": ResidueCharacter.POLAR, "TRP": ResidueCharacter.ALIPHATIC,
                "VAL": ResidueCharacter.ALIPHATIC, "TYR": ResidueCharacter.ALIPHATIC,
            },
            "protorp": {
                "ALA": ResidueCharacter.ALIPHATIC, "CYS": ResidueCharacter.POLAR,
                "GLU": ResidueCharacter.CHARGED, "ASP": ResidueCharacter.CHARGED,
                "GLY": ResidueCharacter.ALIPHATIC, "PHE": ResidueCharacter.ALIPHATIC,
                "ILE": ResidueCharacter.ALIPHATIC, "HIS": ResidueCharacter.POLAR,
                "LYS": ResidueCharacter.CHARGED, "MET": ResidueCharacter.ALIPHATIC,
                "LEU": ResidueCharacter.ALIPHATIC, "ASN": ResidueCharacter.POLAR,
                "GLN": ResidueCharacter.POLAR, "PRO": ResidueCharacter.ALIPHATIC,
                "SER": ResidueCharacter.POLAR, "ARG": ResidueCharacter.CHARGED,
                "THR": ResidueCharacter.POLAR, "TRP": ResidueCharacter.POLAR,
                "VAL": ResidueCharacter.ALIPHATIC, "TYR": ResidueCharacter.POLAR,
            }
        }

        # Residue properties
        self.residue_props = {
            # Kyte-Doolittle hydrophobicity scale
            'hydrophobicity': {
                'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8,  'K': -3.9, 'M': 1.9,  'F': 2.8,  'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            },
            # Chou-Fasman helix propensity
            'helix_propensity': {
                'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
                'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
                'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
                'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
            },
            # Chou-Fasman sheet propensity
            'sheet_propensity': {
                'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
                'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
                'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
                'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
            }
        }

        # Reference ASA values
        self.rel_asa = {
            "ALA": RelativeASAReference(total=107.95, backbone=38.54, sidechain=69.41),
            "CYS": RelativeASAReference(total=134.28, backbone=37.53, sidechain=96.75),
            "ASP": RelativeASAReference(total=140.39, backbone=37.70, sidechain=102.69),
            "GLU": RelativeASAReference(total=172.25, backbone=37.51, sidechain=134.74),
            "PHE": RelativeASAReference(total=199.48, backbone=35.37, sidechain=164.11),
            "GLY": RelativeASAReference(total=80.10, backbone=47.77, sidechain=32.33),
            "HIS": RelativeASAReference(total=182.88, backbone=35.80, sidechain=147.08),
            "ILE": RelativeASAReference(total=175.12, backbone=37.16, sidechain=137.96),
            "LYS": RelativeASAReference(total=200.81, backbone=37.51, sidechain=163.30),
            "LEU": RelativeASAReference(total=178.63, backbone=37.51, sidechain=141.12),
            "MET": RelativeASAReference(total=194.15, backbone=37.51, sidechain=156.64),
            "ASN": RelativeASAReference(total=143.94, backbone=37.70, sidechain=106.24),
            "PRO": RelativeASAReference(total=136.13, backbone=16.23, sidechain=119.90),
            "GLN": RelativeASAReference(total=178.50, backbone=37.51, sidechain=140.99),
            "ARG": RelativeASAReference(total=238.76, backbone=37.51, sidechain=201.25),
            "SER": RelativeASAReference(total=116.50, backbone=38.40, sidechain=78.11),
            "THR": RelativeASAReference(total=139.27, backbone=37.57, sidechain=101.70),
            "VAL": RelativeASAReference(total=151.44, backbone=37.16, sidechain=114.28),
            "TRP": RelativeASAReference(total=249.36, backbone=38.10, sidechain=211.26),
            "TYR": RelativeASAReference(total=212.76, backbone=35.38, sidechain=177.38),
        }

        self.classification_type = classification_type
        # Convert property dictionaries to ordered arrays
        self._init_ordered_props()

    def _init_ordered_props(self):
        """Initialize ordered property arrays matching residue_constants order."""
        self.ordered_props = {}
        for prop_name, prop_dict in self.residue_props.items():
            self.ordered_props[prop_name] = np.array([
                prop_dict[aa] for aa in residue_constants.restypes
            ])

    def get_character(self, residue: str) -> ResidueCharacter:
        """Get the character classification of a residue."""
        return self.aa_character[self.classification_type].get(residue, ResidueCharacter.ALIPHATIC)

    def get_reference_asa(self, residue: str) -> RelativeASAReference:
        """Get the reference ASA values for a residue."""
        return self.rel_asa.get(residue)

    def get_properties(self, residue: str) -> Dict[str, float]:
        """Get all properties for a residue."""
        aa_idx = residue_constants.restype_order[residue_constants.restype_3to1[residue]]
        return {
            prop_name: float(prop_array[aa_idx])
            for prop_name, prop_array in self.ordered_props.items()
        }

    def calculate_relative_asa(self, residue: str, total_asa: float,
                           backbone_asa: float, sidechain_asa: float) -> Dict[str, float]:
        """Calculate relative ASA values for a residue."""
        ref = self.get_reference_asa(residue)
        if ref is None:
            return {
                "total_relative": float('nan'),
                "backbone_relative": float('nan'),
                "sidechain_relative": float('nan')
            }

        return {
            "total_relative": (total_asa / ref.total) * 100,
            "backbone_relative": (backbone_asa / ref.backbone) * 100,
            "sidechain_relative": (sidechain_asa / ref.sidechain) * 100
        }
