# bio_lib

## Setting up the Conda Environment

Create a new conda environment:
```sh
conda create -n bio_lib python=3.10
conda activate bio_lib
pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1
conda install -c conda-forge biopython
```