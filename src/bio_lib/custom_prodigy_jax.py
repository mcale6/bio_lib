from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import os, json
from dataclasses import dataclass
import jax.numpy as jnp
import jax
import bio_lib.common.residue_constants as residue_constants
import bio_lib.common.protein as Protein
from bio_lib.common.residue_classification import ResidueClassification, ResidueCharacter
from bio_lib.common.residue_library import default_library as residue_library
from bio_lib.common.protein_jax import JAXStructureData, JaxProtein
from .shrake_rupley_jax import calculate_sasa

RESIDUE_RADII_MATRIX = jnp.array(residue_library.radii_matrix)
REFERENCE_RELATIVE_SASA_ARRAY = jnp.array(ResidueClassification().ref_rel_sasa_array)

@dataclass
class ContactAnalysis:
    """Results from analyzing interface contacts."""
    CC: float  # charged-charged contacts
    PP: float  # polar-polar contacts
    AA: float  # aliphatic-aliphatic contacts
    AC: float  # aliphatic-charged contacts
    AP: float  # aliphatic-polar contacts
    CP: float  # charged-polar contacts

    def __post_init__(self):
        """Validate contact values."""
        for field_name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"Contact count {field_name} cannot be negative: {value}")
            # Convert to float in case we get jax arrays
            setattr(self, field_name, float(value))

    @property
    def total_contacts(self) -> float:
        """Total number of interface contacts."""
        return self.CC + self.PP + self.AA + self.AC + self.AP + self.CP

    @property
    def charged_contacts(self) -> float:
        """Total contacts involving charged residues."""
        return self.CC + self.AC + self.CP

    @property
    def polar_contacts(self) -> float:
        """Total contacts involving polar residues."""
        return self.PP + self.AP + self.CP

    @property
    def aliphatic_contacts(self) -> float:
        """Total contacts involving aliphatic residues."""
        return self.AA + self.AC + self.AP
    
    def get_percentages(self) -> Dict[str, float]:
        """Calculate percentage for each contact type."""
        total = self.total_contacts
        if total == 0:
            return {name: 0.0 for name in ['CC', 'PP', 'AA', 'AC', 'AP', 'CP']}
        return {
            'CC': 100 * self.CC / total,
            'PP': 100 * self.PP / total,
            'AA': 100 * self.AA / total,
            'AC': 100 * self.AC / total,
            'AP': 100 * self.AP / total,
            'CP': 100 * self.CP / total
        }

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with basic contact counts."""
        return {
            'CC': self.CC,
            'PP': self.PP,
            'AA': self.AA,
            'AC': self.AC,
            'AP': self.AP,
            'CP': self.CP
        }
    
@dataclass
class ProdigyResults:
    contact_types: ContactAnalysis
    binding_affinity: float  
    dissociation_constant: float
    nis_aliphatic: float
    nis_charged: float  
    nis_polar: float
    sasa_data: Optional[Dict[Tuple[str,str,int,str], float]] = None
    relative_sasa: Optional[Dict[Tuple[str,str,int], float]] = None

    def save_results(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save binding results
        binding_path = os.path.join(output_dir, "binding_results.json") 
        with open(binding_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save SASA data
        if self.sasa_data:
            sasa_path = os.path.join(output_dir, "sasa_data.csv")
            with open(sasa_path, 'w') as f:
                f.write("chain,resname,resid,atom,sasa,relative_sasa\n")
                for (chain,resname,resid,atom), sasa in self.sasa_data.items():
                    rel_sasa = self.relative_sasa.get((chain,resname,resid), 0.0) if self.relative_sasa else 0.0
                    f.write(f"{chain},{resname},{resid},{atom},{sasa:.3f},{rel_sasa:.3f}\n")

    def print_prediction(self) -> None:
        print("y")

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
        return (
            f"------------------------\n"
            f"PRODIGY Analysis Results\n"
            f"------------------------\n"
            f"Binding Energy (ΔG): {self.binding_affinity:.2f} kcal/mol\n"
            f"Dissociation Constant (Kd): {self.dissociation_constant:.2e} M\n"
            f"------------------------\n"
            f"\nContact Analysis:\n"
            f"  Charged-Charged: {self.contact_types.CC:.1f}\n"
            f"  Polar-Polar: {self.contact_types.PP:.1f}\n"
            f"  Aliphatic-Aliphatic: {self.contact_types.AA:.1f}\n"
            f"  Aliphatic-Charged: {self.contact_types.AC:.1f}\n"
            f"  Aliphatic-Polar: {self.contact_types.AP:.1f}\n"
            f"  Charged-Polar: {self.contact_types.CP:.1f}\n"
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


def analyse_contacts(contacts: jnp.ndarray, target_seq: jnp.ndarray, binder_seq: jnp.ndarray) -> Dict[str, float]:
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
    cc = contacts * target_class_prob[:, :, 2] * binder_class_prob[:, :, 2]  # charged-charged
    pp = contacts * target_class_prob[:, :, 1] * binder_class_prob[:, :, 1]  # polar-polar
    aa = contacts * target_class_prob[:, :, 0] * binder_class_prob[:, :, 0]  # aliph-aliph

    ac = contacts * (
        (target_class_prob[:, :, 0] * binder_class_prob[:, :, 2]) +  # target-aliph & binder-charged
        (target_class_prob[:, :, 2] * binder_class_prob[:, :, 0])    # target-charged & binder-aliph
    )

    ap = contacts * (
        (target_class_prob[:, :, 0] * binder_class_prob[:, :, 1]) +  # target-aliph & binder-polar
        (target_class_prob[:, :, 1] * binder_class_prob[:, :, 0])    # target-polar & binder-aliph
    )

    cp = contacts * (
        (target_class_prob[:, :, 2] * binder_class_prob[:, :, 1]) +  # target-charged & binder-polar
        (target_class_prob[:, :, 1] * binder_class_prob[:, :, 2])    # target-polar & binder-charged
    )

    return {
        "CC": float(jnp.sum(cc)),
        "PP": float(jnp.sum(pp)),
        "AA": float(jnp.sum(aa)),
        "AC": float(jnp.sum(ac)),
        "AP": float(jnp.sum(ap)),
        "CP": float(jnp.sum(cp))
    }

def analyse_nis_soft(sasa_values: jnp.ndarray, aa_probs: jnp.ndarray, threshold: float = 0.05) -> Tuple[float, float, float]:
    charged_idx, polar_idx, aliphatic_idx = get_residue_character_indices("protorp")
    
    p_charged = jnp.sum(aa_probs[..., charged_idx], axis=-1)
    p_polar = jnp.sum(aa_probs[..., polar_idx], axis=-1)
    p_aliph = jnp.sum(aa_probs[..., aliphatic_idx], axis=-1)
        
    nis_mask = (sasa_values >= threshold)
    n_total = jnp.sum(nis_mask)
    
    n_charged = jnp.sum(nis_mask * p_charged)
    n_polar = jnp.sum(nis_mask * p_polar)
    n_aliph = jnp.sum(nis_mask * p_aliph)
    
    #assert (n_aliph + n_charged + n_polar) == n_total
    
    total = n_total + 1e-8
    return (float(100.0 * n_aliph / total), float(100.0 * n_charged / total), float(100.0 * n_polar / total))

def IC_NIS(ic_cc: float, ic_ca: float, ic_pp: float, ic_pa: float, p_nis_a: float, p_nis_c: float) -> float:
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
    
    dg = (-0.09459 * ic_cc +
          -0.10007 * ic_ca +
           0.19577 * ic_pp +
          -0.22671 * ic_pa +
           0.18681 * (p_nis_a) +
           0.13810 * (p_nis_c) +
          -15.9433)
    
    return float(dg)

def calculate_relative_sasa(
    complex_sasa: jnp.ndarray,  # [n_atoms] or [n_res, 37] depending on format
    total_seq: jnp.ndarray,     # [n_res, n_restypes] one-hot or probability vectors
) -> jnp.ndarray:
    """Calculate relative SASA using ResidueClassification.
    
    Args:
        complex_sasa: SASA values for each atom
        total_seq: [n_res, n_restypes] one-hot or probability vectors
        use_jax_class: Whether using JAX protein format
        residue_numbers: PDB residue numbers for each atom
        residue_index: Sequential residue indices matching total_seq order
    """
    # AlphaFold format with 37 atoms per residue
    atoms_per_res = 37
    residue_sasa = complex_sasa.reshape(-1, atoms_per_res).sum(axis=1)

    # Calculate expected reference SASA based on amino acid probabilities
    complex_ref = jnp.matmul(total_seq, REFERENCE_RELATIVE_SASA_ARRAY)  # [n_res]
    
    # Calculate relative SASA
    return residue_sasa / (complex_ref + 1e-8)

def dg_to_kd(dg: float, temperature: float = 25.0) -> float:
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
    kd = jnp.exp(dg / rt)
    
    return float(kd)


def convert_sasa_to_dict(
    complex_sasa: jnp.ndarray,
    target: Any,
    binder: Any,
) -> Dict[Tuple[str, str, int, str], float]:
    """Convert SASA array to dictionary mapping (chain, resname, resid, atom) to SASA value."""
    sasa_dict = {}
    # For AlphaFold format (37 atoms per residue)
    atoms_per_res = 37
    target_len = target.atom_positions.shape[0]
    binder_len = binder.atom_positions.shape[0]
    
    # Process target chain
    target_sasa = complex_sasa[:target_len * atoms_per_res].reshape(target_len, atoms_per_res)
    for i in range(target_len):
        resname = residue_constants.restype_1to3[residue_constants.restypes[target.aatype[i]]]
        for j, (sasa, mask) in enumerate(zip(target_sasa[i], target.atom_mask[i])):
            if mask > 0:
                atom_name = residue_constants.atom_types[j]
                sasa_dict[("A", resname, i + 1, atom_name)] = float(sasa)
    
    # Process binder chain
    binder_sasa = complex_sasa[target_len * atoms_per_res:].reshape(binder_len, atoms_per_res)
    for i in range(binder_len):
        resname = residue_constants.restype_1to3[residue_constants.restypes[binder.aatype[i]]]
        for j, (sasa, mask) in enumerate(zip(binder_sasa[i], binder.atom_mask[i])):
            if mask > 0:
                atom_name = residue_constants.atom_types[j]
                sasa_dict[("B", resname, i + 1, atom_name)] = float(sasa)
    
    return sasa_dict

def convert_relative_sasa_to_dict(
    relative_sasa: jnp.ndarray,
    target: Any,
    binder: Any,
) -> Dict[Tuple[str, str, int], float]:
    """Convert relative SASA array to dictionary mapping (chain, resname, resid) to relative SASA value."""
    rel_sasa_dict = {}
    target_len = len(target.aatype)
    
    # Process target chain
    target_rel_sasa = relative_sasa[:target_len]
    for i, sasa in enumerate(target_rel_sasa):
        resname = residue_constants.restype_1to3[residue_constants.restypes[target.aatype[i]]]
        resid = i + 1
        rel_sasa_dict[("A", resname, resid)] = float(sasa)
    
    # Process binder chain
    binder_rel_sasa = relative_sasa[target_len:]
    for i, sasa in enumerate(binder_rel_sasa):
        resname = residue_constants.restype_1to3[residue_constants.restypes[binder.aatype[i]]]
        resid = i + 1
        rel_sasa_dict[("B", resname, resid)] = float(sasa)
    
    return rel_sasa_dict

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
    target_seq = jax.nn.one_hot(target.aatype, num_classes=num_classes)
    binder_seq = jax.nn.one_hot(binder.aatype, num_classes=num_classes)
    total_seq = jnp.concatenate([target_seq, binder_seq])

    print("Calculate and analyze contacts")
    contacts = calculate_contacts_af(target.atom_positions, binder.atom_positions, 
                                     target.atom_mask,  binder.atom_mask, cutoff=cutoff)
    contact_types = analyse_contacts(contacts, target_seq, binder_seq)
    
    print("Calculate SASA and relative SASA")
    complex_sasa = calculate_sasa(coords=complex_positions, vdw_radii=complex_radii, mask=complex_mask)
    sasa_dict = convert_sasa_to_dict(complex_sasa, target, binder)
    relative_sasa = calculate_relative_sasa(complex_sasa, total_seq)
    rsa_dict = convert_relative_sasa_to_dict(relative_sasa, target, binder)

    print("Calculate NIS")
    nis_a, nis_c, nis_p = analyse_nis_soft(relative_sasa, total_seq, acc_threshold)

    print("Calculate binding affinity and convert to kd")
    dg = IC_NIS(contact_types["CC"], contact_types["AC"], 
                contact_types["PP"], contact_types["AP"], nis_a, nis_c)
    kd = dg_to_kd(dg, temperature=temperature)

    results = ProdigyResults(
        contact_types=ContactAnalysis(**contact_types),
        binding_affinity=dg,
        dissociation_constant=kd,
        nis_aliphatic=nis_a,
        nis_charged=nis_c,
        nis_polar=nis_p,
        sasa_data=sasa_dict,
        relative_sasa=rsa_dict
    )

    if output_dir:
        results.save_results(output_dir)
    
    if quiet == False:
        print(results)
    
    return results
