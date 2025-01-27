from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import os, json
from dataclasses import dataclass
import jax.numpy as jnp
import jax
import numpy as np
import bio_lib.common.residue_constants as residue_constants
import bio_lib.common.protein as Protein
from bio_lib.common.residue_classification import ResidueClassification, ResidueCharacter
from bio_lib.common.residue_library import default_library as residue_library
from bio_lib.common.protein_jax import JAXStructureData, JaxProtein
from bio_lib.shrake_rupley_jax import calculate_sasa

RESIDUE_RADII_MATRIX = jnp.array(residue_library.radii_matrix)
REFERENCE_RELATIVE_SASA_ARRAY = jnp.array(ResidueClassification().ref_rel_sasa_array)


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
    binding_affinity: float  
    dissociation_constant: float
    nis_aliphatic: float
    nis_charged: float  
    nis_polar: float
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

def get_residue_character_indices(classification_type: str = "ic") -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    residue_classifier = ResidueClassification(classification_type)
    
    character_indices: Dict[ResidueCharacter, list] = {
        ResidueCharacter.CHARGED: [],
        ResidueCharacter.POLAR: [],
        ResidueCharacter.ALIPHATIC: []
    }
    
    for res in residue_constants.restypes:
        res3 = residue_constants.restype_1to3[res]
        char = residue_classifier.aa_character[classification_type][res3]
        idx = residue_constants.restype_order[res]
        character_indices[char].append(idx)
            
    return (
        jnp.array(character_indices[ResidueCharacter.CHARGED]),
        jnp.array(character_indices[ResidueCharacter.POLAR]),
        jnp.array(character_indices[ResidueCharacter.ALIPHATIC])
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

def analyse_contacts(contacts: jnp.ndarray, target_seq: jnp.ndarray, binder_seq: jnp.ndarray) -> jnp.ndarray:
    # Get indices for charged and polar residues
    charged_idx, polar_idx, aliphatic_idx = get_residue_character_indices("ic")

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

def analyse_nis(sasa_values: jnp.ndarray, aa_probs: jnp.ndarray, threshold: float = 0.05) -> jnp.ndarray:
    """Calculate NIS percentages for n_aliph, n_charged, and n_polar residues."""
    charged_idx, polar_idx, aliphatic_idx = get_residue_character_indices("protorp")
    
    p_charged = jnp.sum(aa_probs[..., charged_idx], axis=-1)
    p_polar = jnp.sum(aa_probs[..., polar_idx], axis=-1) 
    p_aliph = jnp.sum(aa_probs[..., aliphatic_idx], axis=-1)
        
    nis_mask = (sasa_values >= threshold)
    n_total = jnp.sum(nis_mask)
    
    n_charged = jnp.sum(nis_mask * p_charged)
    n_polar = jnp.sum(nis_mask * p_polar)
    n_aliph = jnp.sum(nis_mask * p_aliph)
    
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
    """Vectorized version that's ~100x faster for large structures."""
    atoms_per_res = 37
    atom_types = np.array(residue_constants.atom_types)
    restypes = residue_constants.restypes + ['X']
    restype_1to3 = residue_constants.restype_1to3

    # Combine target and binder data
    target_res = len(target.aatype)
    binder_res = len(binder.aatype)
    total_res = target_res + binder_res

    # Reshape SASA data to [total_res, atoms_per_res]
    sasa_matrix = complex_sasa.reshape(total_res, atoms_per_res)

    # Create chain identifiers
    chain_ids = np.concatenate([
        np.full(target_res, 'A'),
        np.full(binder_res, 'B')
    ])

    # Create residue indices (1-based)
    res_indices = np.concatenate([target.residue_index, binder.residue_index]).astype(int)

    # Create residue names
    target_resnames = np.array([restype_1to3[restypes[aa]] for aa in target.aatype])
    binder_resnames = np.array([restype_1to3[restypes[aa]] for aa in binder.aatype])
    resnames = np.concatenate([target_resnames, binder_resnames])

    # Create atom name grid
    atom_names = np.tile(atom_types, total_res)
    
    # Create full index grids
    res_idx_grid = np.repeat(np.arange(total_res), atoms_per_res)
    chain_ids_grid = np.repeat(chain_ids, atoms_per_res)
    resnames_grid = np.repeat(resnames, atoms_per_res)
    resindices_grid = np.repeat(res_indices, atoms_per_res)
    relative_sasa_grid = np.repeat(relative_sasa, atoms_per_res)

    # Filter valid atoms (SASA > 0)
    mask = sasa_matrix.ravel() > 0
    filtered = (
        chain_ids_grid[mask],
        resnames_grid[mask],
        resindices_grid[mask],
        atom_names[mask],
        sasa_matrix.ravel()[mask],
        relative_sasa_grid[mask]
    )

    # Create structured array
    dtype = [
        ('chain', 'U1'), ('resname', 'U3'), ('resindex', 'i4'),
        ('atomname', 'U4'), ('atom_sasa', 'f4'), ('relative_sasa', 'f4')
    ]
    
    return np.array(list(zip(*filtered)), dtype=dtype)

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
    contacts = calculate_contacts_af(target.atom_positions, binder.atom_positions, 
                                     target.atom_mask,  binder.atom_mask, cutoff=cutoff)
    contact_types = analyse_contacts(contacts, target_seq, binder_seq)
    
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