from typing import *
from pathlib import Path
from collections import defaultdict

class AtomInfo(NamedTuple):
    """Store atom information."""
    radius: float
    is_polar: bool

class ResidueLibrary:
    """Handles atom radii and polarity."""
    def __init__(self, library_input: Path = Path("./common/vdw.radii")):
        with open(library_input, 'r') as file: # care here
            library_text = file.read()
        self.residue_atoms = defaultdict(dict)
        self._parse_library(library_text)

    def _parse_library(self, text: str):
        current_residue = None
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('RESIDUE'):
                parts = line.split()
                current_residue = parts[2]
            elif line.startswith('ATOM'):
                if current_residue:
                    atom_name = line[5:9].strip()
                    parts = line[9:].strip().split()
                    radius = float(parts[0])
                    is_polar = bool(int(parts[1]))
                    self.residue_atoms[current_residue][atom_name] = AtomInfo(radius, is_polar)

    def get_radius(self, residue: str, atom: str, element: str = None) -> float:
        atom_info = self.residue_atoms.get(residue, {}).get(atom)
        if atom_info:
            return atom_info.radius

        element_radii = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
            'P': 1.80, 'FE': 1.47, 'ZN': 1.39, 'MG': 1.73
        }
        return element_radii.get(element.upper(), 1.80)

    def is_polar(self, residue: str, atom: str) -> bool:
        atom_info = self.residue_atoms.get(residue, {}).get(atom)
        return bool(atom_info and atom_info.is_polar)
    