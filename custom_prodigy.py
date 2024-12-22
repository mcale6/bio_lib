def execute_freesasa_api(structure):
    """removed os.dup2"""

    asa_data, rsa_data = {}, {}
    _rsa = rel_asa["total"]

    config_path = "/content/prodigy/src/prodigy_prot/naccess.config"
    classifier = Classifier(config_path)

    struct = structureFromBioPDB(structure, classifier)
    result = calc(struct)

    # iterate over all atoms to get SASA and residue name
    for idx in range(struct.nAtoms()):
        atname = struct.atomName(idx)
        resname = struct.residueName(idx)
        resid = struct.residueNumber(idx)
        chain = struct.chainLabel(idx)
        at_uid = (chain, resname, resid, atname)
        res_uid = (chain, resname, resid)
        #
        asa = result.atomArea(idx)
        asa_data[at_uid] = asa
        # add asa to residue
        rsa_data[res_uid] = rsa_data.get(res_uid, 0) + asa

    # convert total asa ro relative asa
    rsa_data.update((res_uid, asa / _rsa[res_uid[1]]) for res_uid, asa in rsa_data.items())
    return asa_data, rsa_data

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

          self.sasa, self.cmplx_sasa = execute_freesasa_api(self.structure) ##asa_data, rsa_data #print(cmplx_sasa) #{('A', 'GLY', '1 '): 0.5120565264525995,
          self.nis_a, self.nis_c, _ = analyse_nis(self.cmplx_sasa, acc_threshold=acc_threshold)

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
    temperature = 25.0,
    distance_cutoff = 5.5,
    acc_threshold = 0.05):
    """ Predict binding affinity using the custom PRODIGY method in python.
    care the relative bsa is sometiems higher than 1. in the codebase of prodigy
    Temperature in Celsius for Kd predictio, Distance cutoff to calculate ICs, Accessibility threshold for BSA analysis
    """
    # Check and parse structure
    #struct_path = check_path(struct_path)
    structure, n_chains, n_res = parse_structure(struct_path)
    #print(f"[+] Parsed structure file {structure.id} ({n_chains} chains, {n_res} residues)")

    # Initialize Prodigy and predict
    prodigy = CustomProdigy(structure, selection, temperature)
    prodigy.predict(distance_cutoff=distance_cutoff, acc_threshold=acc_threshold)

    return prodigy