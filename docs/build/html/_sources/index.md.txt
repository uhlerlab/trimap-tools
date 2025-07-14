# Discovering TCR-Epitope-HLA Interactions

`trimap-tools` is a package for analysis of `peptide-HLA presentation` and `TCR specificity`. It is designed to help researchers understand the interactions between T cell receptors (TCRs) and peptides presented by human leukocyte antigen (HLA) molecules, which play a crucial role in the immune response.

<img src="./_static/diagram.png" alt="diagram" width="1000">

## 🚀 Key Features of `trimap-tools`

- **Peptide representation learning with HLA context**  
  Learns latent embeddings of peptides while incorporating HLA background, enabling biologically informed modeling.

- **TCR specificity prediction using full receptor sequences**  
  Supports comprehensive modeling of TCR recognition by leveraging both α and β chain CDR regions.

- **Visualization of critical TCR residues**  
  Highlights key amino acid positions in TCRs that contribute to antigen recognition, aiding biological interpretation.

- **Discovery of disease-associated epitopes**  
  Identifies novel peptides potentially involved in disease by integrating small-scale disease-specific data with large-scale public datasets.

## 🔧 Components

- `trimap.model.HLA_vae` — Variational Autoencoder for HLA sequences  
- `trimap.model.PEP_vae` — Variational Autoencoder for peptide sequences  
- `trimap.model.THE` — Main prediction module integrating TCRs with peptide or HLA targets

## 📁 Sections

- **Tutorials**  
  Step-by-step guides for training HLA/peptide encoders and predicting TCR specificity using example or custom data.

- **Discover Disease-Associated Epitopes**  
  Learn how to integrate small-scale experimental data with large public datasets to uncover novel epitope–TCR interactions relevant to specific diseases.


Explore the sidebar to get started 👉  
For installation instructions, pretrained models, or benchmarking datasets, refer to the corresponding sections.


<p align="center">
  <a href="https://mapmyvisitors.com/web/1bybt" title="Visit tracker">
    <img src="https://mapmyvisitors.com/map.png?d=Yyk7gzaOQndqLUKtJ_tf-WiyCHTadg_nHYdA0CSOFtI&cl=ffffff" />
  </a>
</p>



---

```{toctree}
:maxdepth: 2
:caption: Contents:
:hidden:

Installation
Turtorial/index
Application/index
API