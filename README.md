# CoNi_SA_G
High throughput computational process of s.a transition metal catalysts on CoNi alloy for Hydrogen Evolution Reaction. 

# Version 1
## Included:
- Automatic structure generator and relaxer for any single atom transition metal placed upon CoNi alloy surfaces
- Automatic statistical graph plotter for analytical use

## Required Dependencies:
FAIRChem requires python 3.12, create an environment and install all required dependencies
```
conda create -n fairchem-uma python=3.12 -y
conda activate fairchem-uma
pip install fairchem-core pymatgen ase
```

Note: A Hugging Face auth token must be saved to access FAIRChem's pretrained models. Ensure you have a Hugging Face account, have applied for access to the [UMA Model Repository](https://huggingface.co/facebook/UMA), and have logged in via an access token. Once approved, save the token
```
hf auth login
```

Further information on FAIRChem can be found at the [fairchem repository](https://github.com/facebookresearch/fairchem).

## Statistical Analysis:
Statistical analysis code may be run on python 3.14, install required dependencies
```
pip install ase pandas openpyxl numpy matplotlib scikit-learn
```
