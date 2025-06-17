
<p align="left">
  <img src="docs/source/_static/logo.png" alt="logo" width="280"/>
</p>


[![PyPI version](https://img.shields.io/pypi/v/your-package-name.svg)](https://pypi.org/project/themap-tools/)
[![Downloads](https://static.pepy.tech/badge/your-package-name)](https://pepy.tech/project/themap-tools)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://themap-tools.readthedocs.io/)

# themap-tools: discovering TCR-HLA-Epitope interactions

<p align="center">
  <img src="docs/source/_static/diagram.png" alt="applications" width="1200"/>
</p>

**themap-tools** is a package for analysis of peptide-HLA presentation and TCR specificity. It is designed to help researchers understand the interactions between T cell receptors (TCRs) and peptides presented by human leukocyte antigen (HLA) molecules, which play a crucial role in the immune response. 

## Installation
**themap-tools** requires **Python 3.9** or later. It is available on PyPI and can be installed using pip:
```bash
pip install themap-tools
```
or by cloning the repository and installing it manually:
```bash
pip install git+https://github.com/uhlerlab/themap-tools.git@main
```

## Tutorials
For step-by-step guides on how to use **themap-tools**, including training HLA/peptide encoders and predicting TCR specificity, please refer to the [Documentation](https://themap-tools.readthedocs.io/) section.

## Key Features

- **Peptide representation learning with HLA context**  
  Learns latent embeddings of peptides while incorporating HLA background, enabling biologically informed modeling.

- **TCR specificity prediction using full receptor sequences**  
  Supports comprehensive modeling of TCR recognition by leveraging both α and β chain CDR regions.

- **Visualization of critical TCR residues**  
  Highlights key amino acid positions in TCRs that contribute to antigen recognition, aiding biological interpretation.

- **Discovery of disease-associated epitopes**  
  Identifies novel peptides potentially involved in disease by integrating small-scale disease-specific data with large-scale public datasets.

## Citation
If you use **themap-tools** in your research, please cite the following paper: