import bio_lib.common.residue_constants as residue_constants
from bio_lib.common.residue_library import default_library as residue_library
import numpy as np
from pathlib import Path
import dataclasses
from typing import Optional
import jax.numpy as jnp
from Bio import PDB
from Bio.PDB import is_aa
from Bio.PDB.Polypeptide import PPBuilder

@dataclasses.dataclass
class JAXStructureData:
    """JAX-compatible structure representation of protein structure.
    Arrays that require computation are stored as JAX arrays (jnp.ndarray).
    Arrays that are purely for reference/lookup remain as numpy arrays (np.ndarray).
    """
    # Required fields - computational arrays as JAX
    atom_positions: jnp.ndarray  # [num_atoms, 3] - Cartesian coordinates
    atom_radii: jnp.ndarray     # [num_atoms] - Atomic radii
    atom_mask: jnp.ndarray      # [num_atoms] - Mask for each atom

    # Required fields - reference arrays as numpy
    atom_names: np.ndarray      # [num_atoms] - PDB atom names
    residue_names: np.ndarray   # [num_atoms] - Three letter residue codes
    chain_ids: np.ndarray       # [num_atoms] - Chain identifiers
    residue_numbers: np.ndarray # [num_atoms] - Residue sequence numbers
    elements: np.ndarray        # [num_atoms] - Element symbols
    aatype: np.ndarray         # [num_res] - Residue type indices
    atom_types: np.ndarray     # List of atom names
    residue_index: np.ndarray  # [num_res] - Residue sequence numbers

    # Optional fields with defaults
    b_factors: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    charges: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    structure_id: str = "structure"

class JaxProtein:
    """Class to process PDB files into JAXStructureData format with support for splitting chains."""
    
    def __init__(self):
        """Initialize with required constants and libraries."""
        self.atom_types = residue_constants.atom_types
        self.atom_order = residue_constants.atom_order
        self.restype_3to1_dict = residue_constants.restype_3to1
        self.restype_order_dict = residue_constants.restype_order
        self.residue_library = residue_library
    
    def _validate_structure(self, structure: PDB.Structure.Structure, selected_chains: Optional[list[str]] = None) -> PDB.Structure.Structure:
        """Validate and clean PDB structure."""
        # Keep first model only
        if len(structure) > 1:
            print("[!] Structure contains more than one model. Only the first one will be kept")
            model_one = structure[0].id
            for m in structure.child_list[:]:
                if m.id != model_one:
                    structure.detach_child(m.id)
                    print(f"Removed model: {m.id}")

        # Process selected chains
        chains = list(structure.get_chains())
        chain_ids = set(c.id for c in chains)

        if selected_chains:
            # Validate selected chains exist
            for chain in selected_chains:
                if chain not in chain_ids:
                    raise ValueError(f"Selected chain not present in structure: {chain}")
            
            # Remove unselected chains
            for chain in chains:
                if chain.id not in selected_chains:
                    chain.parent.detach_child(chain.id)
                    print(f"Removed unselected chain: {chain.id}")

        # Handle double occupancy
        for atom in list(structure.get_atoms()):
            if atom.is_disordered():
                residue = atom.parent
                sel_at = atom.selected_child
                sel_at.altloc = " "
                sel_at.disordered_flag = 0
                residue.detach_child(atom.id)
                residue.add(sel_at)
                print(f"Resolved double occupancy for atom: {atom.id} in residue: {residue.id}")

        # Remove residues with insertion codes
        for chain in structure.get_chains():
            for residue in chain.get_residues():
                if residue.get_id()[2] != " ":
                    chain.detach_child(residue.id)
                    print(f"Removed residue with insertion code: {residue.id}")

        # Remove HETATM, water, and hydrogens
        res_list = list(structure.get_residues())
        for res in res_list:
            if res.id[0][0] in ["W", "H"]:  # Water or HETATM
                chain = res.parent
                chain.detach_child(res.id)
                print(f"Removed HETATM or water residue: {res.id}")
            elif not is_aa(res, standard=True):
                raise ValueError(f"Unsupported non-standard amino acid found: {res.resname}")

        # Remove hydrogen atoms
        for atom in list(structure.get_atoms()):
            if atom.element == "H":
                residue = atom.parent
                residue.detach_child(atom.name)
                print(f"Removed hydrogen atom: {atom.name} in residue: {residue.id}")

        # Check for gaps in structure
        peptides = PPBuilder().build_peptides(structure)
        if len(peptides) != len(chain_ids):
            print("[!] Structure contains gaps:")
            for i, peptide in enumerate(peptides):
                print(f"\t{peptide[0].parent.id} {peptide[0].resname}{peptide[0].id[1]} < Fragment {i} > "
                      f"{peptide[-1].parent.id} {peptide[-1].resname}{peptide[-1].id[1]}")

        return structure

    def _count_valid_atoms(self, structure, print_summary=True) -> int:
        """Pre-calculate number of valid atoms in structure."""
        valid_atom_count = 0
        skipped_residues = []
        skipped_atoms = []
        chains_stats = {}

        for model in structure:
            for chain in model:
                chain_id = chain.id
                chains_stats[chain_id] = {'residues': 0, 'atoms': 0, 'skipped_atoms': 0}
                
                for residue in chain:
                    resname = residue.get_resname()
                    if resname not in self.restype_3to1_dict:
                        skipped_residues.append(f"{resname} {chain_id}:{residue.id[1]}")
                        continue
                    
                    chains_stats[chain_id]['residues'] += 1
                    for atom in residue:
                        if atom.get_name() in self.atom_order:
                            valid_atom_count += 1
                            chains_stats[chain_id]['atoms'] += 1
                        else:
                            skipped_atoms.append(f"{atom.get_name()} in {resname} {chain_id}:{residue.id[1]}")
                            chains_stats[chain_id]['skipped_atoms'] += 1

        # Print summary
        if print_summary:
            print("\n=== Structure Summary ===")
            print(f"Total valid atoms: {valid_atom_count}")
            
            if skipped_residues:
                print("\nSkipped residues:")
                for res in skipped_residues:
                    print(f"- {res}")
            
            if skipped_atoms:
                print("\nSkipped atoms:")
                for atom in skipped_atoms:
                    print(f"- {atom}")
            
            print("\nChain statistics:")
            for chain_id, stats in chains_stats.items():
                print(f"\nChain {chain_id}:")
                print(f"- Residues: {stats['residues']}")
                print(f"- Valid atoms: {stats['atoms']}")
                print(f"- Skipped atoms: {stats['skipped_atoms']}")
            print("=====================\n")
        
        return valid_atom_count

    def _initialize_arrays(self, atom_count: int):
        """Initialize numpy arrays for atom data."""
        return {
            'positions': np.zeros((atom_count, 3)),
            'atom_mask': np.zeros(atom_count),
            'b_factors': np.zeros(atom_count),
            'atom_names': np.empty(atom_count, dtype='<U4'),
            'residue_names': np.empty(atom_count, dtype='<U3'),
            'chain_ids': np.empty(atom_count, dtype='<U1'),
            'residue_numbers': np.zeros(atom_count, dtype=int),
            'elements': np.empty(atom_count, dtype='<U2')
        }

    def _process_structure(self, structure, arrays):
        """Process structure and fill arrays with atom data."""
        aatype_list = []
        residue_index_list = []
        current_atom = 0

        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname()
                    if resname not in self.restype_3to1_dict:
                        continue

                    aatype_list.append(
                        self.restype_order_dict[self.restype_3to1_dict[resname]]
                    )
                    res_idx = residue.get_id()[1]
                    residue_index_list.append(res_idx)

                    for atom in residue:
                        atom_name = atom.get_name()
                        if atom_name not in self.atom_order:
                            continue

                        arrays['positions'][current_atom] = atom.get_coord()
                        arrays['atom_mask'][current_atom] = 1
                        arrays['atom_names'][current_atom] = atom_name
                        arrays['residue_names'][current_atom] = resname
                        arrays['chain_ids'][current_atom] = chain.get_id()
                        arrays['residue_numbers'][current_atom] = res_idx
                        arrays['elements'][current_atom] = atom.element
                        current_atom += 1

        return np.array(aatype_list), np.array(residue_index_list)

    def _calculate_atom_radii(self, residue_names, atom_names, elements):
        """Calculate atomic radii for all atoms."""
        return np.array([
            self.residue_library.get_radius(
                residue=res_name,
                atom=atom_name,
                element=element
            )
            for res_name, atom_name, element in zip(
                residue_names,
                atom_names,
                elements
            )
        ])

    def process_pdb(self, pdb_path: str | Path, selected_chains: Optional[list[str]] = None) -> JAXStructureData:
        """Process a PDB file into JAXStructureData format. """
        # Parse PDB file
        pdb_path = str(pdb_path)
        s_ext = pdb_path.split(".")[-1]
        if s_ext not in {"pdb", "ent", "cif"}:
            raise IOError(f"Structure format '{s_ext}' is not supported. Use '.pdb' or '.cif'")
            
        parser = PDB.PDBParser(QUIET=True) if s_ext in {"pdb", "ent"} else PDB.MMCIFParser()
        structure = parser.get_structure('protein', pdb_path)
        
        # Validate and clean structure
        structure = self._validate_structure(structure, selected_chains)
        
        # Count valid atoms and initialize arrays
        atom_count = self._count_valid_atoms(structure)
        arrays = self._initialize_arrays(atom_count)
        
        # Process structure
        aatype, residue_index = self._process_structure(structure, arrays)
        
        # Calculate atomic radii
        atom_radii = self._calculate_atom_radii(
            arrays['residue_names'],
            arrays['atom_names'],
            arrays['elements']
        )

        # Create JAXStructureData
        return JAXStructureData(
            atom_positions=jnp.array(arrays['positions']),
            atom_radii=jnp.array(atom_radii),
            atom_mask=jnp.array(arrays['atom_mask']),
            atom_names=arrays['atom_names'],
            residue_names=arrays['residue_names'],
            chain_ids=arrays['chain_ids'],
            residue_numbers=arrays['residue_numbers'],
            elements=arrays['elements'],
            b_factors=arrays['b_factors'],
            aatype=aatype,
            atom_types=np.array(self.atom_types),
            residue_index=residue_index,
            charges=np.array([]),
            structure_id="protein"
        )