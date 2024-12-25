# Example of actual atoms vs padded representation
from bio_lib.common import residue_constants
from bio_lib.common.residue_library import ResidueLibrary
import numpy as np

residue_library = ResidueLibrary()
radii_by_aa = {}
for aa in residue_constants.restypes:
    res_name = residue_constants.restype_1to3[aa]
    radii_by_aa[aa] = np.array([
        residue_library.get_radius(res_name, atom_name, atom_name[0])
        for atom_name in residue_constants.atom_types
        ])

residue_examples = {
    'GLY': {
        'actual_atoms': residue_constants.residue_atoms['GLY'],  # ['C', 'CA', 'N', 'O']
        'num_actual': len(residue_constants.residue_atoms['GLY']),  # 4 atoms
        'padded_length': 37,  # Padded to 37 in AF2
        'example_radii': radii_by_aa['G'],  # Shows which values are "real" vs padding
    },

    'ALA': {
        'actual_atoms': residue_constants.residue_atoms['ALA'],  # ['C', 'CA', 'CB', 'N', 'O']
        'num_actual': len(residue_constants.residue_atoms['ALA']),  # 5 atoms
        'padded_length': 37,
        'example_radii': radii_by_aa['A'],
    },

    'TRP': {  # Most complex amino acid
        'actual_atoms': residue_constants.residue_atoms['TRP'],
        # ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'N', 'NE1', 'O']
        'num_actual': len(residue_constants.residue_atoms['TRP']),  # 14 atoms
        'padded_length': 37,
        'example_radii': radii_by_aa['W'],
    }
}
# Show which values correspond to real atoms vs padding
def analyze_residue_radii(aa_code):
    residue = residue_examples[aa_code]
    actual_atoms = residue['actual_atoms']
    radii = residue['example_radii']

    print(f"\n{aa_code} Analysis:")
    print(f"Actual atoms ({len(actual_atoms)}): {actual_atoms}")
    print(f"Total padded length: {residue['padded_length']}")
    print("\nFirst few radii values:")
    for i, (atom, radius) in enumerate(zip(residue_constants.atom_types[:10], radii[:10])):
        status = "REAL" if atom in actual_atoms else "PADDING"
        print(f"{atom}: {radius:.2f} - {status}")

    return analyze_residue_radii

def demo_radius_calculation():
    # 1. Setup simplified radii dictionary for just 3 amino acids
    simple_radii = {
        'A': [1.7] * 37,  # Alanine - all atoms carbon radius
        'G': [1.5] * 37,  # Glycine - different radius to see effect
        'W': [2.0] * 37   # Tryptophan - different radius to see effect
    }

    # 2. Create array with all radii [3, 37]
    all_radii = np.array([simple_radii[aa] for aa in ['A', 'G', 'W']])
    print("All radii shape:", all_radii.shape)

    # 3. Make a sequence example for 2 positions [2, 3]
    # Position 1: 60% Ala, 30% Gly, 10% Trp
    # Position 2: 20% Ala, 50% Gly, 30% Trp
    seq = np.array([
        [0.6, 0.3, 0.1],  # First position
        [0.2, 0.5, 0.3]   # Second position
    ])
    print("\nSequence probabilities shape:", seq.shape)

    # 4. Calculate weighted radii
    radii = np.matmul(seq, all_radii)  # [2, 37]
    print("\nWeighted radii shape:", radii.shape)

    # 5. Show example calculation for first atom of first position
    pos1_radius = (0.6 * 1.7 + 0.3 * 1.5 + 0.1 * 2.0)
    print("\nManual calculation for position 1, atom 1:")
    print(f"0.6 * 1.7 + 0.3 * 1.5 + 0.1 * 2.0 = {pos1_radius}")
    print("Matrix multiply result:", radii[0,0])

    # 6. Reshape to final form
    final_radii = radii.reshape(-1)  # [2 * 37]
    print("\nFinal radii shape:", final_radii.shape)

    return radii, final_radii

analyze_residue_radii("ALA")
weighted_radii, flattened_radii = demo_radius_calculation()
