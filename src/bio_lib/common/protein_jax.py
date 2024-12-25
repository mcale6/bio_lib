import numpy as np
import dataclasses
from typing import Optional
import jax.numpy as jnp
from . import residue_constants
from .residue_library import ResidueLibrary
from Bio import PDB

@dataclasses.dataclass
class JAXStructureData:
    """JAX-compatible structure representation matching StructureData.
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

def input_to_jax_structure(
   atom_positions: np.ndarray,  # [num_atoms, 3]
   atom_radii: np.ndarray,
   aatype: np.ndarray,        # [num_res]
   atom_mask: np.ndarray,     # [num_atoms]
   residue_index: np.ndarray, # [num_res]
   b_factors: Optional[np.ndarray] = None, # [num_atoms]
   atom_names: Optional[np.ndarray] = None,
   residue_names: Optional[np.ndarray] = None,
   chain_ids: Optional[np.ndarray] = None,
   residue_numbers: Optional[np.ndarray] = None,
   elements: Optional[np.ndarray] = None,
   structure_id: str = "protein",
) -> JAXStructureData:

   return JAXStructureData(
       atom_positions=jnp.array(atom_positions),
       atom_radii=jnp.array(atom_radii),
       atom_mask=jnp.array(atom_mask),
       atom_names=atom_names,
       residue_names=residue_names,
       chain_ids=chain_ids,
       residue_numbers=residue_numbers,
       elements=elements,
       b_factors=b_factors,
       aatype=aatype,
       atom_types=np.array(residue_constants.atom_types),
       residue_index=residue_index,
       charges=np.array([]),
       structure_id=structure_id,

   )

def process_pdb_to_arrays(pdb_file: str, residue_library: ResidueLibrary):
   atom_types = residue_constants.atom_types
   atom_order = residue_constants.atom_order
   restype_3to1_dict = residue_constants.restype_3to1
   restype_order_dict = residue_constants.restype_order

   parser = PDB.PDBParser(QUIET=True)
   structure = parser.get_structure('protein', pdb_file)

   # Pre-calculate valid atoms count
   valid_atom_count = 0
   for model in structure:
       for chain in model:
           for residue in chain:
               if residue.get_resname() not in restype_3to1_dict:
                   print(f"Skipping residue {residue.get_resname()} - not in lookup")
                   continue
               for atom in residue:
                   if atom.get_name() in atom_order:
                       valid_atom_count += 1
                   else:
                       print(f"Skipping atom {atom.get_name()} in {residue.get_resname()} - not in lookup")

   # Initialize arrays
   positions = np.zeros((valid_atom_count, 3))
   atom_mask = np.zeros(valid_atom_count)
   b_factors = np.zeros(valid_atom_count)
   atom_names = np.empty(valid_atom_count, dtype='<U4')
   residue_names = np.empty(valid_atom_count, dtype='<U3')
   chain_ids = np.empty(valid_atom_count, dtype='<U1')
   residue_numbers = np.zeros(valid_atom_count, dtype=int)
   elements = np.empty(valid_atom_count, dtype='<U2')

   aatype_list = []
   residue_index_list = []
   current_atom = 0

   for model in structure:
       for chain in model:
           for residue in chain:
               resname = residue.get_resname()
               if resname not in restype_3to1_dict:
                   continue

               aatype_list.append(restype_order_dict[restype_3to1_dict[resname]])
               res_idx = residue.get_id()[1]
               residue_index_list.append(res_idx)

               for atom in residue:
                   atom_name = atom.get_name()
                   if atom_name not in atom_order:
                       continue

                   positions[current_atom] = atom.get_coord()
                   atom_mask[current_atom] = 1
                   atom_names[current_atom] = atom_name
                   residue_names[current_atom] = resname
                   chain_ids[current_atom] = chain.get_id()
                   residue_numbers[current_atom] = res_idx
                   elements[current_atom] = atom.element
                   current_atom += 1

   # Calculate atomic radii
   atom_radii = np.array([
       residue_library.get_radius(
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
   return (
       positions,
       np.array(aatype_list),
       atom_mask,
       np.array(residue_index_list),
       b_factors,
       atom_names,  
       residue_names,
       chain_ids,
       residue_numbers,
       elements,
       atom_radii
   )