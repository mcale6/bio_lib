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

The `run_prodigy_jax.py` module supports benchmarking protein complex analysis across multiple structures:

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
    use_jax_class=True  # Toggle between JAX/AlphaFold2 processing
)
```

### Benchmark Results

![Benchmark Analysis](benchmark_af/analysis_plots2.png)

![Benchmark Analysis](benchmark_jax/analysis_plots.png)

## Usage

### Command Line Interface

```bash
run-prodigy-jax complex.pdb A B --format human --format json
```

#### CLI Arguments
```
- pdb_path: Path to PDB file
- target_chain: Target protein chain ID
- binder_chain: Binder protein chain ID
- --cutoff: Contact distance cutoff (Å, default: 5.5)
- --acc_threshold: SASA threshold (default: 0.05)
- --output-dir: Output directory (default: ./results)
- --format: Output format [json|human|both] (default: both)