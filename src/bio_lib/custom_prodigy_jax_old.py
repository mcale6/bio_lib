from typing import Dict, Literal, Optional, Any
from pathlib import Path
import os, json
from dataclasses import dataclass
import jax.numpy as jnp
import jax
import numpy as np
import bio_lib.common.residue_constants as residue_constants
import bio_lib.common.protein as Protein
from bio_lib.common.residue_classification import ResidueClassification
from bio_lib.common.residue_library import default_library as residue_library
from bio_lib.common.protein_jax import JAXStructureData, JaxProtein
from bio_lib.shrake_rupley_jax import calculate_sasa, calculate_sasa2

RESIDUE_RADII_MATRIX = jnp.array(residue_library.radii_matrix)
REFERENCE_RELATIVE_SASA_ARRAY = jnp.array(ResidueClassification().ref_rel_sasa_array)
rc_ic = ResidueClassification("ic")
rc_pro = ResidueClassification("protorp")
_RESCLASS_MATRICES = rc_ic.classification_matrix  # Access the precomputed matrix, Access the cached indices

charged_idx_protorp, polar_idx_protorp, aliphatic_idx_protorp = ResidueClassification().get_residue_character_indices("protorp")
CHARACTER_MATRIX_PROTORP = jnp.zeros((len(residue_constants.restypes), 3))
CHARACTER_MATRIX_PROTORP = CHARACTER_MATRIX_PROTORP.at[charged_idx_protorp, 0].set(1.0)
CHARACTER_MATRIX_PROTORP = CHARACTER_MATRIX_PROTORP.at[polar_idx_protorp, 1].set(1.0)
CHARACTER_MATRIX_PROTORP = CHARACTER_MATRIX_PROTORP.at[aliphatic_idx_protorp, 2].set(1.0)
charged_idx_ic, polar_idx_ic, aliphatic_idx_ic = ResidueClassification().get_residue_character_indices("ic")
CHARACTER_MATRIX_IC = jnp.zeros((len(residue_constants.restypes), 3))
CHARACTER_MATRIX_IC = CHARACTER_MATRIX_IC.at[charged_idx_ic, 0].set(1.0)
CHARACTER_MATRIX_IC = CHARACTER_MATRIX_IC.at[polar_idx_ic, 1].set(1.0)
CHARACTER_MATRIX_IC = CHARACTER_MATRIX_IC.at[aliphatic_idx_ic, 2].set(1.0)

@dataclass
class ContactAnalysis:
    """Results from analyzing interface contacts.
    CC: charged-charged contacts
    PP: polar-polar contacts
    AA: aliphatic-aliphatic contacts
    AC: aliphatic-charged contacts
    AP: aliphatic-polar contacts
    CP: charged-polar contacts
    """
    values: jnp.ndarray  # Array containing [CC, PP, AA, AC, AP, CP] values in this order!

    def __post_init__(self):
        """Convert input to float array if needed."""
        self.values = jnp.asarray(self.values, dtype=float)
        if self.values.shape != (6,):
            raise ValueError("Contact values must be array of shape (6,)")

    @property
    def total_contacts(self) -> float:
        """Total number of interface contacts."""
        return float(jnp.sum(self.values))

    @property 
    def charged_contacts(self) -> float:
        """Total contacts involving charged residues."""
        return float(self.values[0] + self.values[3] + self.values[5])

    @property
    def polar_contacts(self) -> float:
        """Total contacts involving polar residues."""
        return float(self.values[1] + self.values[4] + self.values[5])

    @property
    def aliphatic_contacts(self) -> float:
        """Total contacts involving aliphatic residues."""
        return float(self.values[2] + self.values[3] + self.values[4])

    def get_percentages(self) -> Dict[str, float]:
        """Calculate percentage for each contact type."""
        total = self.total_contacts
        if total == 0:
            return {name: 0.0 for name in ['CC', 'PP', 'AA', 'AC', 'AP', 'CP']}
        return {
            'CC': 100 * self.values[0] / total,
            'PP': 100 * self.values[1] / total,
            'AA': 100 * self.values[2] / total,
            'AC': 100 * self.values[3] / total,
            'AP': 100 * self.values[4] / total,
            'CP': 100 * self.values[5] / total
        }

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with basic contact counts."""
        return {
            'CC': float(jnp.asarray(self.values[0])),
            'PP': float(jnp.asarray(self.values[1])), 
            'AA': float(jnp.asarray(self.values[2])),
            'AC': float(jnp.asarray(self.values[3])),
            'AP': float(jnp.asarray(self.values[4])),
            'CP': float(jnp.asarray(self.values[5]))
        }

@dataclass
class ProdigyResults:
    contact_types: ContactAnalysis
    binding_affinity: jnp.ndarray
    dissociation_constant: jnp.ndarray
    nis_aliphatic: jnp.ndarray
    nis_charged: jnp.ndarray
    nis_polar: jnp.ndarray
    structure_id: str = "_"
    sasa_data: np.ndarray = None

    def save_results(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save binding results
        binding_path = os.path.join(output_dir, f"{self.structure_id}_binding_results.json") 
        with open(binding_path, 'w') as f:
            json.dump(self.contact_types.to_dict(), f, indent=2)

        # Save SASA data
        if self.sasa_data is not None:
            sasa_path = os.path.join(output_dir, f"{self.structure_id}_sasa_data.csv")
            with open(sasa_path, 'w') as f:
                f.write("chain,resname,resid,atom,sasa,relative_sasa\n")
                for row in self.sasa_data:
                    f.write(f"{row['chain']},{row['resname']},{row['resindex']},"
                           f"{row['atomname']},{row['atom_sasa']:.3f},{row['relative_sasa']:.3f}\n")

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary including all data."""
        return {
            'ba_val': self.binding_affinity,
            'kd': self.dissociation_constant,
            'contacts': self.contact_types.to_dict(),
            'contact_percentages': self.contact_types.get_percentages(),
            'nis': {
                'aliphatic': self.nis_aliphatic,
                'charged': self.nis_charged, 
                'polar': self.nis_polar
            }
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        contact_types_dict = self.contact_types.to_dict()
        return (
            f"------------------------\n"
            f"PRODIGY Analysis Results\n"
            f"------------------------\n"
            f"Binding Energy (ΔG): {self.binding_affinity:.2f} kcal/mol\n"
            f"Dissociation Constant (Kd): {self.dissociation_constant:.2e} M\n"
            f"------------------------\n"
            f"\nContact Analysis:\n"
            f"  Charged-Charged: {contact_types_dict['CC']:.1f}\n"
            f"  Polar-Polar: {contact_types_dict['PP']:.1f}\n"
            f"  Aliphatic-Aliphatic: {contact_types_dict['AA']:.1f}\n"
            f"  Aliphatic-Charged: {contact_types_dict['AC']:.1f}\n"
            f"  Aliphatic-Polar: {contact_types_dict['AP']:.1f}\n"
            f"  Charged-Polar: {contact_types_dict['CP']:.1f}\n"
            f"------------------------\n"
            f"\nNon-Interacting Surface:\n"
            f"  Aliphatic: {self.nis_aliphatic:.1f}%\n"
            f"  Charged: {self.nis_charged:.1f}%\n"
            f"  Polar: {self.nis_polar:.1f}%\n"
            f"------------------------\n"
        )

def load_pdb_to_af(pdb_path: str, target_chain: str, binder_chain: str):
    with open(pdb_path, 'r') as f:
        pdb_str = f.read()
    
    target = Protein.from_pdb_string(pdb_str, chain_id=target_chain)
    binder = Protein.from_pdb_string(pdb_str, chain_id=binder_chain)
    
    return target, binder

def get_atom_radii(aatype: jnp.ndarray) -> jnp.ndarray: ### to docheck this
    seq_one_hot = jax.nn.one_hot(aatype, len(residue_constants.restypes))
    return jnp.matmul(seq_one_hot, RESIDUE_RADII_MATRIX).reshape(-1)

def calculate_contacts_af(
    target_pos: jnp.ndarray,
    binder_pos: jnp.ndarray,
    target_mask: jnp.ndarray,
    binder_mask: jnp.ndarray,
    cutoff: float = 5.5
) -> jnp.ndarray:
    """Calculate contacts using AlphaFold protein format with 37 atoms per residue."""
    #print(f"Input shapes:")
    #print(f"target_pos: {target_pos.shape}")
    #print(f"binder_pos: {binder_pos.shape}")
    #print(f"target_mask: {target_mask.shape}")
    #print(f"binder_mask: {binder_mask.shape}")

    # Reshape to atoms
    target_atoms = target_pos.reshape(-1, 3)
    binder_atoms = binder_pos.reshape(-1, 3)
    target_atom_mask = target_mask.reshape(-1)
    binder_atom_mask = binder_mask.reshape(-1)

    #print(f"\nAfter initial reshape:")
    #print(f"target_atoms: {target_atoms.shape}")
    #print(f"binder_atoms: {binder_atoms.shape}")
    #print(f"target_atom_mask: {target_atom_mask.shape}")
    #print(f"binder_atom_mask: {binder_atom_mask.shape}")

    # Calculate pairwise distances
    diff = target_atoms[:, None, :] - binder_atoms[None, :, :]
    #print(f"\nDiff shape: {diff.shape}")  # [target_len*37, binder_len*37, 3]
    
    dist2 = jnp.sum(diff * diff, axis=-1)
    #print(f"Distance matrix shape: {dist2.shape}")  # [target_len*37, binder_len*37]

    # Convert masks to boolean and combine with distance check
    atom_contacts = (dist2 <= (cutoff * cutoff)) & \
                   (target_atom_mask[:, None] > 0) & \
                   (binder_atom_mask[None, :] > 0)
    #print(f"Atom contacts shape: {atom_contacts.shape}")

    # Reshape to residue pairs
    target_len = target_pos.shape[0]
    binder_len = binder_pos.shape[0]
    atoms_per_res = target_pos.shape[1]  # 37

    contacts_reshaped = atom_contacts.reshape(target_len, atoms_per_res,
                                            binder_len, atoms_per_res)
    #print(f"\nContacts reshaped: {contacts_reshaped.shape}")  # [target_len, 37, binder_len, 37]

    # Any atom-atom contact makes a residue-residue contact
    residue_contacts = jnp.any(jnp.any(contacts_reshaped, axis=-1), axis=1)
    #print(f"Final residue contacts shape: {residue_contacts.shape}")  # [target_len, binder_len]

    return residue_contacts

def analyse_contacts_af(contacts: jnp.ndarray, target_seq: jnp.ndarray, binder_seq: jnp.ndarray) -> jnp.ndarray:
    # Get indices for charged and polar residues
    charged_idx, polar_idx, aliphatic_idx = ResidueClassification().get_residue_character_indices("ic")

    # For target (one-hot), direct classification
    target_charged = target_seq[:, charged_idx].sum(axis=-1)  # [target_len]
    target_polar = target_seq[:, polar_idx].sum(axis=-1)      # [target_len]

    # For binder (probabilities), weighted classification
    # Sum probabilities for each class
    binder_charged = binder_seq[:, charged_idx].sum(axis=-1)  # [binder_len]
    binder_polar = binder_seq[:, polar_idx].sum(axis=-1)      # [binder_len]

    # Make 2D for broadcasting
    target_class_prob = jnp.stack([
        1 - (target_charged + target_polar),  # aliphatic prob
        target_polar,                         # polar prob
        target_charged                        # charged prob
    ], axis=-1)[:, None, :]  # [target_len, 1, 3]

    binder_class_prob = jnp.stack([
        1 - (binder_charged + binder_polar),  # aliphatic prob
        binder_polar,                         # polar prob
        binder_charged                        # charged prob
    ], axis=-1)[None, :, :]  # [1, binder_len, 3]

    # Calculate contact type probabilities
    # Each contact is weighted by probability of residue types
    cc = jnp.sum(contacts * target_class_prob[:, :, 2] * binder_class_prob[:, :, 2])  # charged-charged
    pp = jnp.sum(contacts * target_class_prob[:, :, 1] * binder_class_prob[:, :, 1])  # polar-polar
    aa = jnp.sum(contacts * target_class_prob[:, :, 0] * binder_class_prob[:, :, 0])  # aliph-aliph

    ac = jnp.sum(contacts * (
        (target_class_prob[:, :, 0] * binder_class_prob[:, :, 2]) +  # target-aliph & binder-charged
        (target_class_prob[:, :, 2] * binder_class_prob[:, :, 0])    # target-charged & binder-aliph
    ))

    ap = jnp.sum(contacts * (
        (target_class_prob[:, :, 0] * binder_class_prob[:, :, 1]) +  # target-aliph & binder-polar
        (target_class_prob[:, :, 1] * binder_class_prob[:, :, 0])    # target-polar & binder-aliph
    ))

    cp = jnp.sum(contacts * (
        (target_class_prob[:, :, 2] * binder_class_prob[:, :, 1]) +  # target-charged & binder-polar
        (target_class_prob[:, :, 1] * binder_class_prob[:, :, 2])    # target-polar & binder-charged
    ))

    return jnp.array([cc, pp, aa, ac, ap, cp])

def calculate_contacts(
    target_pos: jnp.ndarray,
    binder_pos: jnp.ndarray,
    target_mask: jnp.ndarray,
    binder_mask: jnp.ndarray,
    cutoff: float = 5.5
) -> jnp.ndarray:
    # Reshape masks to residue-atom format
    target_mask = target_mask.reshape(target_pos.shape[0], -1)
    binder_mask = binder_mask.reshape(binder_pos.shape[0], -1)

    # Calculate pairwise distances using broadcasting
    diff = target_pos[:, None, :, None, :] - binder_pos[None, :, None, :, :]
    dist2 = jnp.sum(diff ** 2, axis=-1)  # Shape: [T, B, 37, 37]

    # Combine masks and distance criteria
    cutoff_sq = cutoff ** 2
    contact_mask = (
        (dist2 <= cutoff_sq) &
        (target_mask[:, None, :, None] > 0) & 
        (binder_mask[None, :, None, :] > 0)
    )

    # Aggregate to residue-level contacts
    return jnp.any(contact_mask, axis=(2, 3))

def analyse_contacts_af(
    contacts: jnp.ndarray, 
    target_seq: jnp.ndarray, 
    binder_seq: jnp.ndarray,
    classification_type: Literal["ic", "protorp"] = "ic"
) -> jnp.ndarray:
    """Analyze contacts and calculate interaction probabilities."""
    # Get precomputed classification matrix
    class_matrix = _RESCLASS_MATRICES[classification_type]

    # Convert sequences to class probabilities (matches original logic)
    target_classes = target_seq @ class_matrix  # [T, 3]
    binder_classes = binder_seq @ class_matrix  # [B, 3]

    # Calculate all pairwise interaction probabilities
    interaction_probs = jnp.einsum('ti,bj->tbij', target_classes, binder_classes)
    
    # Apply contact mask and sum interactions
    masked_interactions = interaction_probs * contacts[:, :, None, None]
    total = masked_interactions.sum(axis=(0, 1))
    
    # Return in original order: [cc, pp, aa, ac, ap, cp]
    return jnp.array([
        total[2, 2],                   # Charged-Charged
        total[1, 1],                   # Polar-Polar 
        total[0, 0],                   # Aliphatic-Aliphatic
        total[0, 2] + total[2, 0],     # Aliphatic-Charged
        total[0, 1] + total[1, 0],     # Aliphatic-Polar
        total[1, 2] + total[2, 1]      # Charged-Polar
    ])


def analyse_nis(sasa_values: jnp.ndarray, aa_probs: jnp.ndarray, threshold: float = 0.05) -> jnp.ndarray:
    """Calculate NIS percentages for n_aliph, n_charged, and n_polar residues."""
    # Combined character probabilities via matrix multiplication
    p_chars = jnp.matmul(aa_probs, CHARACTER_MATRIX_PROTORP)  # [n_res, 3]
    p_charged, p_polar, p_aliph = p_chars[:, 0], p_chars[:, 1], p_chars[:, 2]
    
    # Mask for residues meeting SASA threshold
    nis_mask = (sasa_values >= threshold)
    n_total = jnp.sum(nis_mask)
    
    # Calculate weighted counts using mask
    n_charged = jnp.sum(nis_mask * p_charged)
    n_polar = jnp.sum(nis_mask * p_polar)
    n_aliph = jnp.sum(nis_mask * p_aliph)
    
    # Avoid division by zero
    total = n_total + 1e-8
    
    return jnp.array([
        100.0 * n_aliph / total,
        100.0 * n_charged / total, 
        100.0 * n_polar / total
    ])

def IC_NIS(ic_cc: jnp.ndarray, ic_ca: jnp.ndarray, ic_pp: jnp.ndarray, ic_pa: jnp.ndarray, p_nis_a: jnp.ndarray, p_nis_c: jnp.ndarray) -> jnp.ndarray:
    """Calculate binding affinity (ΔG) using interface composition and NIS.
    Args:
        ic_cc: number of charged-charged contacts
        ic_ca: number of charged-aliphatic contacts
        ic_pp: number of polar-polar contacts
        ic_pa: number of polar-aliphatic contacts
        p_nis_a: percentage of non-interacting aliphatic surface
        p_nis_c: percentage of non-interacting charged surface
    """
    # Bound NIS values to reasonable ranges
    if p_nis_a < 0 or p_nis_a > 100:
        print(f"p_nis_a out of range: {p_nis_a}, clipping to [0, 100]")
    if p_nis_c < 0 or p_nis_c > 100:
        print(f"p_nis_c out of range: {p_nis_c}, clipping to [0, 100]")
    
    p_nis_a = jnp.clip(p_nis_a, 0, 100)
    p_nis_c = jnp.clip(p_nis_c, 0, 100)
    
    return (-0.09459 * ic_cc +
          -0.10007 * ic_ca +
           0.19577 * ic_pp +
          -0.22671 * ic_pa +
           0.18681 * (p_nis_a) +
           0.13810 * (p_nis_c) +
          -15.9433)

def calculate_relative_sasa(
    complex_sasa: jnp.ndarray,  # [n_atoms] or [n_res, 37] depending on format
    total_seq: jnp.ndarray,     # [n_res, n_restypes] one-hot or probability vectors
) -> jnp.ndarray:
    """Calculate relative SASA using ResidueClassification."""
    # AlphaFold format with 37 atoms per residue
    atoms_per_res = 37
    residue_sasa = complex_sasa.reshape(-1, atoms_per_res).sum(axis=1)

    # Calculate expected reference SASA based on amino acid probabilities
    complex_ref = jnp.matmul(total_seq, REFERENCE_RELATIVE_SASA_ARRAY)  # [n_res]
    
    # Calculate relative SASA
    return residue_sasa / (complex_ref + 1e-8)

def dg_to_kd(dg: jnp.ndarray, temperature: float = 25.0) -> jnp.ndarray:
    """Convert binding free energy to dissociation constant.
        - Uses the relationship: ΔG = RT ln(Kd)
        - R = 0.0019858775 kcal/(mol·K)
    """
    # Ensure reasonable DG range to avoid overflow
    if dg < -100 or dg > 100:
        print(f"dg out of range: {dg}, clipping to [-100, 100]")
    dg = jnp.clip(dg, -100, 100)
    
    # Convert temperature and calculate RT
    rt = 0.0019858775 * (temperature + 273.15)  # R in kcal/(mol·K)
    
    # Calculate Kd with numerical stability
    return jnp.exp(dg / rt)

def convert_sasa_to_array(
    complex_sasa: jnp.ndarray,
    relative_sasa: jnp.ndarray,
    target: Protein,
    binder: Protein,
) -> np.ndarray:
    atoms_per_res = 37
    atom_types = np.array(residue_constants.atom_types)
    restypes = residue_constants.restypes #+ ['X']
    restype_1to3 = residue_constants.restype_1to3

    # Combine target and binder data
    target_res = len(target.aatype)
    binder_res = len(binder.aatype)
    total_res = target_res + binder_res

    # Combined atom mask (flattened)
    combined_mask = jnp.concatenate([target.atom_mask, binder.atom_mask]).ravel()
    combined_mask_np = np.asarray(combined_mask, dtype=bool)

    # Chain IDs for each residue
    chain_ids = np.concatenate([np.full(target_res, 'A'), np.full(binder_res, 'B')])

    # Residue indices (1-based)
    res_indices = np.concatenate([target.residue_index, binder.residue_index]).astype(int)

    # Residue names using vectorized lookup
    restype_1to3_arr = np.array([restype_1to3[r] for r in restypes], dtype='U3')
    target_resnames = restype_1to3_arr[np.array(target.aatype)]
    binder_resnames = restype_1to3_arr[np.array(binder.aatype)]
    resnames = np.concatenate([target_resnames, binder_resnames])

    # Atom names for all atoms
    atom_names = np.tile(atom_types, total_res)

    # Expand residue-level data to atom-level
    chain_ids_atom = np.repeat(chain_ids, atoms_per_res)
    resnames_atom = np.repeat(resnames, atoms_per_res)
    resindices_atom = np.repeat(res_indices, atoms_per_res)
    relative_sasa_atom = np.repeat(relative_sasa, atoms_per_res)

    # Apply mask to filter valid atoms
    filtered_data = (
        chain_ids_atom[combined_mask_np],
        resnames_atom[combined_mask_np],
        resindices_atom[combined_mask_np],
        atom_names[combined_mask_np],
        np.asarray(complex_sasa)[combined_mask_np],
        relative_sasa_atom[combined_mask_np]
    )

    # Create structured array
    dtype = [
        ('chain', 'U1'), ('resname', 'U3'), ('resindex', 'i4'),
        ('atomname', 'U4'), ('atom_sasa', 'f4'), ('relative_sasa', 'f4')
    ]
    return np.array(list(zip(*filtered_data)), dtype=dtype)

def predict_binding_affinity_jax(
    pdb_path: str | Path,
    selection: str = "A,B",
    cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    temperature: float = 25.0,
    output_dir: Optional[str] = ".",
    quiet: bool = True,
) -> ProdigyResults:
    """Run the full PRODIGY analysis pipeline."""
    target_chain, binder_chain = selection.split(",")
    target, binder = load_pdb_to_af(pdb_path, target_chain, binder_chain)
    complex_positions = jnp.concatenate([target.atom_positions, binder.atom_positions], axis=0).reshape(-1, 3)
    complex_radii = jnp.concatenate([get_atom_radii(target.aatype), get_atom_radii(binder.aatype)])
    complex_mask = jnp.concatenate([target.atom_mask, binder.atom_mask], axis=0).reshape(-1)

    print("Convert sequences to one-hot")
    num_classes = len(residue_constants.restypes)
    target_seq = jax.nn.one_hot(target.aatype, num_classes=num_classes) #sequence_to_onehot
    binder_seq = jax.nn.one_hot(binder.aatype, num_classes=num_classes)
    total_seq = jnp.concatenate([target_seq, binder_seq])

    print("Calculate and analyze contacts")
    contacts = calculate_contacts(target.atom_positions, binder.atom_positions, 
                                     target.atom_mask,  binder.atom_mask, cutoff=cutoff)
    contact_types = analyse_contacts_af(contacts, target_seq, binder_seq)
    
    print("Calculate SASA and relative SASA")
    complex_sasa = calculate_sasa(coords=complex_positions, vdw_radii=complex_radii, mask=complex_mask)
    relative_sasa = calculate_relative_sasa(complex_sasa, total_seq)
  
    print("Calculate NIS")
    nis_acp = analyse_nis(relative_sasa, total_seq, acc_threshold)

    print("Calculate binding affinity and convert to kd")
    #ic_cc: float, ic_ca: float, ic_pp: float, ic_pa:
    dg = IC_NIS(contact_types[0], contact_types[3], contact_types[1], contact_types[4], nis_acp[0], nis_acp[1]) #jnp.array([cc, pp, aa, ac, ap, cp])
    kd = dg_to_kd(dg, temperature=temperature)

    print("Convert SASA data to array")
    sasa_dict = convert_sasa_to_array(complex_sasa, relative_sasa, target, binder)

    results = ProdigyResults(
        contact_types=ContactAnalysis(contact_types),
        binding_affinity=dg,
        dissociation_constant=kd,
        nis_aliphatic=nis_acp[0],
        nis_charged=nis_acp[1],
        nis_polar=nis_acp[2],
        structure_id=Path(pdb_path).stem,
        sasa_data=sasa_dict
    )
    print("Save Results")
    if output_dir:
        results.save_results(output_dir)
    
    if quiet == False:
        print(results)
    
    return results