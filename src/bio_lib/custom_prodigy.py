from typing import Optional
import numpy as np
import pkg_resources
from bio_lib.common.residue_classification import ResidueClassification
from bio_lib.helpers.types import ContactAnalysis, ProdigyResults

from prodigy_prot.predict_IC import Prodigy, analyse_contacts, calculate_ic, analyse_nis
from prodigy_prot.modules.models import IC_NIS
from prodigy_prot.modules.parsers import parse_structure

# Create a dictionary comprehension that reorganizes the data
_REL_SASA = {res: asa.total for res, asa in ResidueClassification().rel_asa.items()}
_NACCESS_CONFIG_PATH = pkg_resources.resource_filename('bio_lib','data/naccess.config')

def execute_freesasa_api2(structure, sphere_points=None):
    """Compute SASA using freesasa and return absolute and relative SASA differences."""
    import freesasa
    from freesasa import Classifier, calc, structureFromBioPDB
    
    if sphere_points:
        parameters = freesasa.Parameters({
            'algorithm': freesasa.ShrakeRupley,
            'n-points': sphere_points,         # number of test points for Shrake-Rupley
        }) 
    else:
        parameters = freesasa.Parameters({
            'algorithm': freesasa.LeeRichards,
            'n-slices': 20,      # number of slices for Lee-Richards algorithm
        })

    classifier = Classifier(_NACCESS_CONFIG_PATH)
    struct = structureFromBioPDB(structure, classifier)
    result = calc(struct, parameters)

    asa_data, rsa_data, abs_diff_data = {}, {}, {}
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

    rsa_data.update((res_uid, asa / _REL_SASA[res_uid[1]]) for res_uid, asa in rsa_data.items()) # per residue 
    abs_diff_data.update((res_uid, abs(asa - _REL_SASA[res_uid[1]])) for res_uid, asa in abs_diff_data.items()) # per residue 
    return asa_data, rsa_data, abs_diff_data

class CustomProdigy(Prodigy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # overwrite the predict function to use dr-sasa or patched freeesa
    def predict(self, temperature=25.0, distance_cutoff=5.5, acc_threshold=0.05, sphere_points=100):
        self.temperature = temperature
        self.distance_cutoff = distance_cutoff
        self.acc_threshold = acc_threshold
        self.sphere_points = sphere_points
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
        self.asa_data, self.rsa_data, self.abs_diff_data = execute_freesasa_api2(self.structure, sphere_points=self.sphere_points)
        #chain_sums_atm = lambda d: {'total': sum(d.values()), 'per_chain': {chain: sum(v for (c, _, _, _), v in d.items() if c == chain) for chain in {k[0] for k in d.keys()}}}
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
    struct_path: str,
    selection: Optional[str] = None,
    temperature: float = 25.0,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    sphere_points: int = 100,
    save_results: bool = False,
    output_dir: str = ".",
    quiet: bool = False
) -> ProdigyResults:
    """Predict binding affinity using the custom PRODIGY method"""
    # Parse structure and run prediction as before
    structure, n_chains, n_res = parse_structure(struct_path)
    print(f"[+] Parsed structure file {structure.id} ({n_chains} chains, {n_res} residues)")

    prodigy = CustomProdigy(structure, selection, temperature)
    prodigy.predict(temperature=temperature, distance_cutoff=distance_cutoff, acc_threshold=acc_threshold, sphere_points=sphere_points)

    # Create ContactAnalysis object from prodigy bins
    contacts = ContactAnalysis([
        prodigy.bins['AA'],
        prodigy.bins['CC'],
        prodigy.bins['PP'],
        prodigy.bins['AC'],
        prodigy.bins['AP'],
        prodigy.bins['CP']
    ])

    # Create ProdigyResults object
    results = ProdigyResults(
        structure_id=structure.id,
        contact_types=contacts,
        binding_affinity=np.float32(prodigy.ba_val),
        dissociation_constant=np.float32(prodigy.kd_val),
        nis_aliphatic=np.float32(prodigy.nis_a),
        nis_charged=np.float32(prodigy.nis_c),
        nis_polar=np.float32(prodigy.nis_p),
        sasa_data=np.array([{
            'chain': chain,
            'resname': resname,
            'resindex': int(resid),
            'atomname': atom,
            'atom_sasa': sasa,
            'relative_sasa': prodigy.rsa_data.get((chain, resname, resid), 0.0)
        } for (chain, resname, resid, atom), sasa in prodigy.asa_data.items()])
    )

    if save_results:
        results.save_results(output_dir)
    if quiet == False:
        print(results)
    else:
        print(f' Predicted binding affinity (kcal.mol-1): {np.float32(prodigy.ba_val)}')
    
    return results