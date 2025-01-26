# Bio-Lib

A Python library for analyzing protein interactions, calculating Solvent Accessible Surface Area (SASA), predicting binding affinity (prodigy), and identifying residue contacts in JAX.

## Installation

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bio_lib==0.9.2
```

## Core Features
- **Input Processing in JAX**: Support for both AlphaFold2 (works) and custom JAX structure processing (in progress).
  - **Residue Classification**: Amino acid categorization (charged, polar, aliphatic) etc.
- **Binding Affinity Prediction in JAX**: ΔG and Kd estimation using interface contacts and surface properties, customized [PRODIGY](https://github.com/haddocking/prodigy)
  - **SASA Calculation**: JAX-based implementation of Shrake-Rupley algorithm for solvent-accessible surface area calculation
  - **Contact Analysis**: Distance-based residue-residue contact determination within protein complexes

## Benchmarking

```bash
run-prodigy-jax complex.pdb A B --format human --format json
run-prodigy-jax PRODIGYdataset/ # folder with pdb files, two chain names have to be all the same (A & B)
```

```python
# Process single structure
from bio_lib import run_prodigy_jax
results = run_prodigy_jax.run("complex.pdb", "A", "B")
print(results)

# Process directory of structures with timing
results = run_prodigy_jax.process_structures(
    "path/to/pdbs/",
    target_chain="A",
    binder_chain="B",
    use_jax_class=False  # Toggle between JAX/AlphaFold2 processing, AlphaFold2 is tested, JAX vesio nin progress
)
```

### Benchmark Results

| **Metric**                     | **Pearson r** | **p-value**       | **RMSE**   |
|---------------------------------|---------------|--------------------|------------|
| Binding Affinity               | 0.999808      | 1.605573e-131      | 0.040133   |
| Charged-Charged contacts       | 1.000000      | 0.000000e+00       | 0.000000   |
| Charged-Polar contacts         | 1.000000      | 0.000000e+00       | 0.000000   |
| Aliphatic-Charged contacts     | 1.000000      | 0.000000e+00       | 0.000000   |
| Polar-Polar contacts           | 1.000000      | 0.000000e+00       | 0.000000   |
| Aliphatic-Polar contacts       | 1.000000      | 0.000000e+00       | 0.000000   |
| Aliphatic-Aliphatic contacts   | 1.000000      | 0.000000e+00       | 0.000000   |
| NIS Polar                      | 0.999700      | 3.383779e-124      | 0.166575   |
| NIS Aliphatic                  | 0.998786      | 3.949009e-101      | 0.206809   |
| NIS Charged                    | 0.999774      | 7.695594e-129      | 0.139352   |

![Benchmark Analysis](benchmark_jax/corr_plots_org_vs_jax.png)

## Usage

### Command Line Interface

```bash
run-prodigy-jax complex.pdb A B --format human --format json
run-prodigy-jax PRODIGYdataset/ # folder with pdb files, two chain names have to be all the same (A & B)
```