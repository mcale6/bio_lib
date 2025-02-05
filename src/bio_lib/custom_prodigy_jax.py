from typing import Optional
from pathlib import Path
import pkg_resources
import jax.numpy as jnp
import jax
import numpy as np
import bio_lib.common.residue_constants as residue_constants 
import bio_lib.common.protein as Protein
from bio_lib.common.residue_classification import ResidueClassification
from bio_lib.common.residue_library import default_library as residue_library
from bio_lib.shrake_rupley_jax import calculate_sasa, calculate_sasa_batch
from bio_lib.helpers.utils import estimate_optimal_block_size
from bio_lib.helpers.types import ContactAnalysis, ProdigyResults

_DEFAULT_BACKEND = jax.default_backend()
_SPHERE_POINTS_100 = jnp.array(np.loadtxt(pkg_resources.resource_filename('bio_lib', 'data/thomson100.xyz') , skiprows=1))
_SPHERE_POINTS_1000 = jnp.array(np.loadtxt(pkg_resources.resource_filename('bio_lib', 'data/thomson1000.xyz') , skiprows=1))
_RESIDUE_RADII_MATRIX = jnp.array(residue_library.radii_matrix)
_RELATIVE_SASA_ARRAY = jnp.array(ResidueClassification().relative_sasa_array)
_RESCLASS_MATRICES_IC = jnp.array(ResidueClassification("ic").classification_matrix)
_RESCLASS_MATRICES_PROTORP  = jnp.array(ResidueClassification("protorp").classification_matrix) 
_ATOMS_PER_RES = residue_constants.atom_type_num # 37

# NIS Constants from the PRODIGY model
nis_constants = {
    'ic_cc': -0.09459,
    'ic_ca': -0.10007,
    'ic_pp': 0.19577,
    'ic_pa': -0.22671,
    'p_nis_a': 0.18681,
    'p_nis_c': 0.13810,
    'intercept': -15.9433
}
_COEEFS= jnp.array([
    nis_constants['ic_cc'],
    nis_constants['ic_ca'],
    nis_constants['ic_pp'], 
    nis_constants['ic_pa'],
    nis_constants['p_nis_a'],
    nis_constants['p_nis_c']
])
_INTERCEPT = jnp.array([nis_constants['intercept']])

def load_pdb_to_af(struct_path: str, target_chain: str, binder_chain: str):
    with open(struct_path, 'r') as f:
        pdb_str = f.read()
    
    target = Protein.from_pdb_string(pdb_str, chain_id=target_chain)
    binder = Protein.from_pdb_string(pdb_str, chain_id=binder_chain)
    
    return target, binder

def get_atom_radii(seq_one_hot: jnp.ndarray, residue_raddi_matrix: jnp.ndarray = _RESIDUE_RADII_MATRIX ) -> jnp.ndarray:
    return jnp.matmul(seq_one_hot, residue_raddi_matrix).reshape(-1)

def convert_sasa_to_array(
    complex_sasa: jnp.ndarray,
    relative_sasa: jnp.ndarray,
    target: Protein,
    binder: Protein,
) -> np.ndarray:
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
    chain_ids_atom = np.repeat(chain_ids, _ATOMS_PER_RES)
    resnames_atom = np.repeat(resnames, _ATOMS_PER_RES)
    resindices_atom = np.repeat(res_indices, _ATOMS_PER_RES)
    relative_sasa_atom = np.repeat(relative_sasa, _ATOMS_PER_RES)

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

def calculate_contacts(
    target_pos: jnp.ndarray,
    binder_pos: jnp.ndarray,
    target_mask: jnp.ndarray,
    binder_mask: jnp.ndarray,
    distance_cutoff: float = 5.5
) -> jnp.ndarray:
    # Reshape masks to residue-atom format
    target_mask = target_mask.reshape(target_pos.shape[0], -1)
    binder_mask = binder_mask.reshape(binder_pos.shape[0], -1)

    # Calculate pairwise distances using broadcasting
    diff = target_pos[:, None, :, None, :] - binder_pos[None, :, None, :, :]
    dist2 = jnp.sum(diff ** 2, axis=-1)  # Shape: [T, B, 37, 37]

    # Combine masks and distance criteria
    cutoff_sq = distance_cutoff ** 2
    contact_mask = (
        (dist2 <= cutoff_sq) &
        (target_mask[:, None, :, None] > 0) & 
        (binder_mask[None, :, None, :] > 0)
    )
    # Aggregate to residue-level contacts
    return jnp.any(contact_mask, axis=(2, 3))

def analyse_contacts(
    contacts: jnp.ndarray, 
    oh_target_seq: jnp.ndarray, 
    oh_binder_seq: jnp.ndarray,
    class_matrix: jnp.ndarray = _RESCLASS_MATRICES_IC
) -> jnp.ndarray:
    """Analyze contacts and calculate interaction probabilities."""
    # Convert sequences to class probabilities (matches original logic)
    target_classes = oh_target_seq @ class_matrix  # [T, 3]
    binder_classes = oh_binder_seq @ class_matrix  # [B, 3]

    # Calculate all pairwise interaction probabilities
    interaction_probs = jnp.einsum('ti,bj->tbij', target_classes, binder_classes)
    
    # Apply contact mask and sum interactions
    masked_interactions = interaction_probs * contacts[:, :, None, None]
    total = masked_interactions.sum(axis=(0, 1)) # matrix with a, c, p are the columns x rows
    # Returns order: [aa, cc, pp, ac, ap, cp]
    return jnp.array([
        total[0, 0],                   # Aliphatic-Aliphatic
        total[1, 1],                   # Charged-Charged
        total[2, 2],                   # Polar-Polar 
        total[0, 1] + total[1, 0],     # Aliphatic-Charged
        total[0, 2] + total[2, 0],     # Aliphatic-Polar
        total[1, 2] + total[2, 1]      # Charged-Polar
    ])

def analyse_nis(
    sasa_values: jnp.ndarray, 
    aa_probs: jnp.ndarray, 
    threshold: float = 0.05,
    character_matrix: jnp.ndarray = _RESCLASS_MATRICES_PROTORP
) -> jnp.ndarray:
    """Calculate NIS percentages using precomputed character matrix."""
    p_chars = aa_probs @ character_matrix  # [n_res, 3]
    mask = (sasa_values >= threshold)  # [n_res]
    
    counts = jnp.sum(p_chars[:, ] * mask[:, None], axis=0)
    total = jnp.sum(mask) + 1e-8
    
    return 100.0 * counts / total

def IC_NIS(
    ic_cc: jnp.ndarray,
    ic_ca: jnp.ndarray, 
    ic_pp: jnp.ndarray,
    ic_pa: jnp.ndarray,
    p_nis_a: jnp.ndarray,
    p_nis_c: jnp.ndarray
) -> jnp.ndarray:
    """Calculate protein-protein binding affinity (ΔG) using interface contacts and non-interacting surface areas.
        ic_cc: Number of charged-charged contacts at interface
        ic_ca: Number of charged-aliphatic contacts at interface 
        ic_pp: Number of polar-polar contacts at interface
        ic_pa: Number of polar-aliphatic contacts at interface
        p_nis_a: Percentage of non-interacting aliphatic surface area (0-100)
    """
    # Clip NIS values to valid range
    p_nis_a = jnp.clip(p_nis_a, 0, 100)
    p_nis_c = jnp.clip(p_nis_c, 0, 100)

    # Stack inputs into single array
    inputs = jnp.array([ic_cc, ic_ca, ic_pp, ic_pa, p_nis_a, p_nis_c])
    return jnp.dot(_COEEFS, inputs) +_INTERCEPT

def calculate_relative_sasa(
    complex_sasa: jnp.ndarray,  # [n_atoms] or [n_res, 37] depending on format
    oh_total_seq: jnp.ndarray,     # [n_res, n_restypes] one-hot or probability vectors
) -> jnp.ndarray:
    """Calculate relative SASA using ResidueClassification."""
    # AlphaFold format with 37 atoms per residue
    residue_sasa = complex_sasa.reshape(-1, _ATOMS_PER_RES).sum(axis=1)

    # Calculate expected reference SASA based on amino acid probabilities
    complex_ref = jnp.matmul(oh_total_seq, _RELATIVE_SASA_ARRAY)  # [n_res]
    
    # Calculate relative SASA
    return residue_sasa / (complex_ref + 1e-8)

def dg_to_kd(dg: jnp.ndarray, temperature: float = 25.0) -> jnp.ndarray:
    """Convert binding free energy (ΔG) to dissociation constant (Kd)."""
    # Physical constants
    R = 0.0019858775
    temp_k = temperature + 273.15
    
    # Clip dG to prevent overflow
    dg_clipped = jnp.clip(dg, -100.0, 100.0)
    if jnp.any(dg != dg_clipped):
        print(f"Warning: ΔG values outside [-100, 100] kcal/mol were clipped")

    return jnp.exp(dg_clipped / (R * temp_k))

def predict_binding_affinity_jax(
    struct_path: str | Path,
    selection: str = "A,B",
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    temperature: float = 25.0,
    sphere_points: int = 100,
    save_results=False,
    output_dir: Optional[str] = ".",
    quiet: bool = True,
) -> ProdigyResults:
    """Run the full PRODIGY analysis pipeline."""
    print("Load pdb")
    target_chain, binder_chain = selection.split(",")
    target, binder = load_pdb_to_af(struct_path, target_chain, binder_chain)
    complex_positions = jnp.concatenate([target.atom_positions, binder.atom_positions], axis=0).reshape(-1, 3)
    complex_mask = jnp.concatenate([target.atom_mask, binder.atom_mask], axis=0).reshape(-1)
    print("Convert sequences to one-hot")
    num_classes = len(residue_constants.restypes)
    oh_target_seq = jax.nn.one_hot(target.aatype, num_classes=num_classes) #sequence_to_onehot
    oh_binder_seq = jax.nn.one_hot(binder.aatype, num_classes=num_classes)
    oh_total_seq = jnp.concatenate([oh_target_seq, oh_binder_seq])
    complex_radii = jnp.concatenate([get_atom_radii(oh_target_seq), get_atom_radii(oh_binder_seq)])

    if "METAL" in _DEFAULT_BACKEND and complex_positions.shape[0] > 15000:
        raise ValueError("Too many atoms for this method")

    print("Calculate contacts")
    contacts = calculate_contacts(target.atom_positions, binder.atom_positions, target.atom_mask,  binder.atom_mask, distance_cutoff=distance_cutoff)
    print("Analyse_ contacts")
    contact_types = analyse_contacts(contacts, oh_target_seq, oh_binder_seq)
    print("Calculate SASA")
    sp = _SPHERE_POINTS_100 if sphere_points == 100 else _SPHERE_POINTS_1000
    if "METAL" in _DEFAULT_BACKEND:
        bs = estimate_optimal_block_size(complex_positions.shape[0])
        complex_sasa = calculate_sasa_batch(
            coords=complex_positions, 
            vdw_radii=complex_radii, 
            mask=complex_mask, 
            sphere_points=sp, 
            block_size=bs
        )
    else:
        complex_sasa = calculate_sasa(
            coords=complex_positions, 
            vdw_radii=complex_radii, 
            mask=complex_mask, 
            sphere_points=sp
        )
    
    print("Calculate relative SASA")
    relative_sasa = calculate_relative_sasa(complex_sasa, oh_total_seq)
    print("Calculate NIS")
    nis_acp = analyse_nis(relative_sasa, oh_total_seq, acc_threshold) 
    print("Calculate binding affinity")
    # contact_types Returns order: [aa, cc, pp, ac, ap, cp]: we need ic_cc, ic_ca, ic_pp,, ic_pa, 
    dg = IC_NIS(contact_types[1], contact_types[3], contact_types[2], contact_types[4], nis_acp[0], nis_acp[1])
    print("Convert to KD")
    kd = dg_to_kd(dg, temperature=temperature)
    print("Convert SASA data to array")
    sasa_dict = convert_sasa_to_array(complex_sasa, relative_sasa, target, binder)
    print("Save Results")
    results = ProdigyResults(
        contact_types=ContactAnalysis(contact_types),
        binding_affinity=np.float32(dg[0]),
        dissociation_constant=np.float32(kd[0]),
        nis_aliphatic=np.float32(nis_acp[0]),
        nis_charged=np.float32(nis_acp[1]),
        nis_polar=np.float32(nis_acp[2]),
        structure_id=Path(struct_path).stem,
        sasa_data=sasa_dict
    )

    if save_results:
        results.save_results(output_dir)
    
    if quiet == False:
        print(results)
    else:
        print(f' Predicted binding affinity (kcal.mol-1): {np.float32(dg[0])}')
    
    return results