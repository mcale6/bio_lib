# Bio-Lib

A library for biological computations with JAX support, featuring PRODIGY binding affinity predictions in pure jax.

## Installation

```bash
!python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bio_lib==0.2.8
```
## Usage
from bio_lib import run_prodigy_jax

# Run PRODIGY analysis
results = run_prodigy_jax.run("complex.pdb", "A", "B")
print(results)
```

## Or from command line:
```
run-prodigy-jax complex.pdb A B --format human
```
