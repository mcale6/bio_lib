import json
import os
import argparse
import numpy as np
import pkg_resources
from bio_lib.common.residue_classification import ResidueClassification, ResidueCharacter
from bio_lib.common.residue_library import default_library as residue_library

from prodigy_prot.predict_IC import *
from prodigy_prot.modules.freesasa_tools import *
from prodigy_prot.modules.models import *
from prodigy_prot.modules.parsers import *
from prodigy_prot.modules.utils import *

# Create a dictionary comprehension that reorganizes the data
REL_SASA = {res: asa.total for res, asa in ResidueClassification().rel_asa.items()}
NACCESS_CONFIG_PATH = pkg_resources.resource_filename('bio_lib', 'data/naccess.config')

def execute_freesasa_api2(structure):
    """
    Compute SASA using freesasa and return absolute and relative SASA differences.

    Args:
        structure: Input PDB structure.

    Returns:
        asa_data: Dictionary containing per-atom SASA values.
        rsa_data: Dictionary containing relative SASA values per residue.
        abs_diff_data: Dictionary containing absolute SASA differences per residue.
    """
    from freesasa import Classifier, calc, structureFromBioPDB

    asa_data, rsa_data, abs_diff_data = {}, {}, {}
    _rsa = REL_SASA
    classifier = Classifier(NACCESS_CONFIG_PATH)

    struct = structureFromBioPDB(structure, classifier)
    result = calc(struct)

    # Iterate over all atoms to get SASA and residue information.
    for idx in range(struct.nAtoms()):
        atname = struct.atomName(idx)
        resname = struct.residueName(idx)
        resid = struct.residueNumber(idx)
        chain = struct.chainLabel(idx)
        at_uid = (chain, resname, resid, atname)
        res_uid = (chain, resname, resid)

        asa = result.atomArea(idx)
        asa_data[at_uid] = asa # per atom
        rsa_data[res_uid] = rsa_data.get(res_uid, 0) + asa # per residue
        abs_diff_data[res_uid] = rsa_data.get(res_uid, 0) + asa # per residue

    rsa_data.update((res_uid, asa / _rsa[res_uid[1]]) for res_uid, asa in rsa_data.items()) # per residue 
    abs_diff_data.update((res_uid, abs(asa - _rsa[res_uid[1]])) for res_uid, asa in abs_diff_data.items()) # per residue 
    return asa_data, rsa_data, abs_diff_data

class CustomProdigy(Prodigy):
      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

      # overwrite the predict function to use dr-sasa or patched freeesa
      def predict(self, temp=None, distance_cutoff=5.5, acc_threshold=0.05):
        if temp is not None:
            self.temp = temp
        # Make selection dict from user option or PDB chains
        selection_dict = {}
        for igroup, group in enumerate(self.selection):
            chains = group.split(",")
            for chain in chains:
                if chain in selection_dict:
                    errmsg = (
                        "Selections must be disjoint sets: "
                        f"{chain} is repeated"
                    )
                    raise ValueError(errmsg)
                selection_dict[chain] = igroup

        # Contacts
        self.ic_network = calculate_ic(self.structure, d_cutoff=distance_cutoff, selection=selection_dict)
        self.bins = analyse_contacts(self.ic_network)
        # SASA
        self.asa_data, self.rsa_data, self.abs_diff_data = execute_freesasa_api2(self.structure)
        chain_sums_atm = lambda d: {'total': sum(d.values()), 'per_chain': {chain: sum(v for (c, _, _, _), v in d.items() if c == chain) for chain in {k[0] for k in d.keys()}}}
        print(chain_sums_atm(self.asa_data))
        #print(self.asa_data)

        self.nis_a, self.nis_c, self.nis_p = analyse_nis(self.rsa_data, acc_threshold=acc_threshold)
        # Affinity Calculation
        self.ba_val = IC_NIS(
            self.bins["CC"],
            self.bins["AC"],
            self.bins["PP"],
            self.bins["AP"],
            self.nis_a,
            self.nis_c,
        )
        self.kd_val = dg_to_kd(self.ba_val, self.temp)
        return

def predict_binding_affinity(
    struct_path,
    selection=None,
    temperature=25.0,
    distance_cutoff=5.5,
    acc_threshold=0.0,
    save_results=False,
    output_dir=None,
    quiet=False):
    """ Predict binding affinity using the custom PRODIGY method in python.
    care the relative bsa is sometiems higher than 1. in the codebase of prodigy
    Temperature in Celsius for Kd predictio, Distance cutoff to calculate ICs, Accessibility threshold from rel. SASA analysis
    """
    # Check and parse structure
    structure, n_chains, n_res = parse_structure(struct_path)
    print(f"[+] Parsed structure file {structure.id} ({n_chains} chains, {n_res} residues)")

    # Initialize Prodigy and predict
    prodigy = CustomProdigy(structure, selection, temperature)
    prodigy.predict(distance_cutoff=distance_cutoff, acc_threshold=acc_threshold)
    prodigy.print_prediction(quiet=quiet)
    results = prodigy.as_dict()

    if save_results:
        res_fname_json = os.path.basename(struct_path.replace(".pdb", "_ba_results.json"))
        res_fname_csv = os.path.basename(struct_path.replace(".pdb", "_sasa_atom_results.csv"))
        asa_csv_lines = "\n".join(["Chain,ResName,ResID,Atom,SASA"] + [f"{chain},{resname},{resid.strip()},{atom.strip()},{sasa:.3f}" for (chain, resname, resid, atom), sasa in prodigy.asa_data.items()])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path_json = os.path.join(output_dir, res_fname_json)
            output_path_csv = os.path.join(output_dir, res_fname_csv)
        else:
            output_path_json = os.path.join(".", res_fname_json)
            output_path_csv = os.path.join(".", res_fname_csv)
        
        with open(output_path_json, "w") as json_file:
            json.dump(results, json_file, indent=4)
        with open(output_path_csv, "w") as f:
            f.write(asa_csv_lines)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict binding affinity using the PRODIGY method.")
    
    parser.add_argument(
        "struct_path", 
        type=str, 
        help="Path to the input structure file."
    )
    parser.add_argument(
        "--selection", 
        type=str, 
        default=None, 
        help="Selection of atoms or residues (optional)."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=25.0, 
        help="Temperature in Celsius for Kd prediction (default: 25.0)."
    )
    parser.add_argument(
        "--distance_cutoff", 
        type=float, 
        default=5.5, 
        help="Distance cutoff for interface contacts (default: 5.5 Å)."
    )
    parser.add_argument(
        "--acc_threshold", 
        type=float, 
        default=0.05, 
        help="Accessibility threshold from rel. SASA (default: 0.05)."
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save the prediction results to a file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the results (optional)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Outputs only the predicted affinity value",
    )
    args = parser.parse_args()

    # Call the prediction function
    result = predict_binding_affinity(
        struct_path=args.struct_path,
        selection=args.selection,
        temperature=args.temperature,
        distance_cutoff=args.distance_cutoff,
        acc_threshold=args.acc_threshold,
        save_results=args.save_results,
        output_dir=args.output_dir,
        quiet=args.quiet
    )

    # Optionally print or save results
    print("Binding affinity prediction completed.")
    print(result)  # Customize based on the `CustomProdigy` output format.

if __name__ == "__main__":
    main()