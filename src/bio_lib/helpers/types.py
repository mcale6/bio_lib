import json
import os
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from bio_lib.helpers.utils import NumpyEncoder

@dataclass
class ContactAnalysis:
    """Results from analyzing interface contacts.
    AA: aliphatic-aliphatic contacts
    CC: charged-charged contacts
    PP: polar-polar contacts
    AC: aliphatic-charged contacts
    AP: aliphatic-polar contacts
    CP: charged-polar contacts
    """
    values: list  # List containing exactly in this order [AA, CC, PP, AC, AP, CP]

    def __post_init__(self):
        """Convert input to float list if needed."""
        self.values = [float(v) for v in self.values]
        if len(self.values) != 6:
            raise ValueError("Contact values must be a list of length 6")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with basic contact counts, total contacts, and grouped contacts."""
        total_inter_atomic_contacts = sum(self.values)
        charged_contacts = self.values[1] + self.values[3] + self.values[5]  # CC + AC + CP
        polar_contacts = self.values[2] + self.values[4] + self.values[5]    # PP + AP + CP
        aliphatic_contacts = self.values[0] + self.values[3] + self.values[4]  # AA + AC + AP
        return {
            'AA': self.values[0],
            'CC': self.values[1],
            'PP': self.values[2],
            'AC': self.values[3],
            'AP': self.values[4],
            'CP': self.values[5],
            'IC': total_inter_atomic_contacts,
            'chargedC': charged_contacts,
            'polarC': polar_contacts,
            'aliphaticC': aliphatic_contacts,
        }

@dataclass
class ProdigyResults:
    contact_types: ContactAnalysis
    binding_affinity: np.float32
    dissociation_constant: np.float32
    nis_aliphatic: np.float32
    nis_charged: np.float32
    nis_polar: np.float32
    structure_id: str = "_"
    sasa_data: np.ndarray = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary including all data."""
        # Convert SASA data to a list of dictionaries
        sasa_list = []
        if self.sasa_data is not None:
            for row in self.sasa_data:
                sasa_list.append({
                    'chain': row['chain'],
                    'resname': row['resname'],
                    'resindex': row['resindex'],
                    'atomname': row['atomname'],
                    'atom_sasa': float(row['atom_sasa']),
                    'relative_sasa': float(row['relative_sasa'])
                })

        # Construct the final dictionary
        return {
            'structure_id': self.structure_id,
            'ba_val': float(self.binding_affinity),
            'kd': float(self.dissociation_constant),
            'contacts': self.contact_types.to_dict(),
            'nis': {
                'aliphatic': float(self.nis_aliphatic),
                'charged': float(self.nis_charged),
                'polar': float(self.nis_polar)
            },
            'sasa_data': sasa_list  # Include SASA data as a list of dictionaries
        }
    
    def save_results(self, output_dir: str) -> None:
        """Save all results as a single JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        results_dict = self.to_dict()
        output_path = os.path.join(output_dir, f"{self.structure_id}_results.json")
        print(f"Saving results to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyEncoder)

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