import json
import os
import numpy as np
import pkg_resources
from bio_lib.common.residue_classification import ResidueClassification

from prodigy_prot.predict_IC import Prodigy, analyse_contacts, calculate_ic, analyse_nis
from prodigy_prot.modules.models import IC_NIS
from prodigy_prot.modules.parsers import parse_structure

# Create a dictionary comprehension that reorganizes the data
REL_SASA = {res: asa.total for res, asa in ResidueClassification().rel_asa.items()}
NACCESS_CONFIG_PATH = pkg_resources.resource_filename('bio_lib','data/naccess.config')

def execute_freesasa_api2(structure):
    """Compute SASA using freesasa and return absolute and relative SASA differences."""
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

    def as_dict(self):
            return_dict = {
                "structure": self.structure.id,
                "selection": ":".join(self.selection),
                "temp": self.temp,
                "ICs": len(self.ic_network),
                "nis_a": self.nis_a,
                "nis_c": self.nis_c,
                "nis_p": self.nis_p,
                "ba_val": self.ba_val,
                "kd_val": self.kd_val,
            }
            return_dict.update(self.bins)
            return return_dict

    def save_results(self, output_dir):
        """Save results to JSON and CSV files."""
        struct_path = self.structure.id  # Get structure ID/path
        
        # Generate filenames
        res_fname_json = os.path.basename(struct_path + "_binding_results.json")
        res_fname_csv = os.path.basename(struct_path + "_sasa_data.csv")

        # Format CSV data
        asa_csv_lines = "\n".join(["chain,resname,resid,atom,sasa,relative_sasa"] + 
            [f"{chain},{resname},{resid.strip()},{atom.strip()},{sasa:.3f},{self.rsa_data.get((chain,resname,resid), 0.0):.3f}" 
             for (chain, resname, resid, atom), sasa in self.asa_data.items()])

        # Set output paths
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path_json = os.path.join(output_dir, res_fname_json)
            output_path_csv = os.path.join(output_dir, res_fname_csv)
        else:
            output_path_json = os.path.join(".", res_fname_json)
            output_path_csv = os.path.join(".", res_fname_csv)
        
        # Save files
        with open(output_path_json, "w") as json_file:
            json.dump(self.as_dict(), json_file, indent=4)
        with open(output_path_csv, "w") as f:
            f.write(asa_csv_lines)

    # overwrite the predict function to use dr-sasa or patched freeesa
    def predict(self, temperature=25.0, distance_cutoff=5.5, acc_threshold=0.05):
        self.temperature = temperature
        self.distance_cutoff = distance_cutoff
        self.acc_threshold = acc_threshold
        # Make selection dict from user option or PDB chains
        selection_dict = {}
        self.selection = self.selection.split(",")
        if len(self.selection) != 2:
            raise ValueError("[-] Selection must be a list of two chains.")
        selection_dict = {chain: idx for idx, chain in enumerate(self.selection)}
        # Contacts
        self.ic_network = calculate_ic(self.structure, d_cutoff=self.distance_cutoff, selection=selection_dict)
        self.bins = analyse_contacts(self.ic_network)
        # SASA
        self.asa_data, self.rsa_data, self.abs_diff_data = execute_freesasa_api2(self.structure)
        chain_sums_atm = lambda d: {'total': sum(d.values()), 'per_chain': {chain: sum(v for (c, _, _, _), v in d.items() if c == chain) for chain in {k[0] for k in d.keys()}}}
        #print(chain_sums_atm(self.asa_data))

        self.nis_a, self.nis_c, self.nis_p = analyse_nis(self.rsa_data, acc_threshold=self.acc_threshold)
        # Affinity Calculation
        self.ba_val = IC_NIS(
            self.bins["CC"],
            self.bins["AC"],
            self.bins["PP"],
            self.bins["AP"],
            self.nis_a,
            self.nis_c,
        )
        self.kd_val = np.exp(self.ba_val / (0.0019858775 * (self.temperature + 273.15)))
        return
    
def predict_binding_affinity(
    struct_path,
    selection=None,
    temperature=25.0,
    distance_cutoff=5.5,
    acc_threshold=0.05,
    save_results=False,
    output_dir=".",
    quiet=False):
    """ Predict binding affinity using the custom PRODIGY method in python. Care the relative bsa is sometiems higher than 1. in the codebase of prodigy"""
    # Check and parse structure
    structure, n_chains, n_res = parse_structure(struct_path)
    print(f"[+] Parsed structure file {structure.id} ({n_chains} chains, {n_res} residues)")

    # Initialize Prodigy and predict
    prodigy = CustomProdigy(structure, selection, temperature)
    prodigy.predict(temperature=temperature, distance_cutoff=distance_cutoff, acc_threshold=acc_threshold)
    prodigy.print_prediction(quiet=quiet)

    if save_results:
        prodigy.save_results(output_dir)
    
    return prodigy.as_dict()