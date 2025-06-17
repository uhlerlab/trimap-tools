# Discover disease-associated epitopes

To apply themap-tools for scanning disease-associated epitopes, users need access to a small set of disease-relevant data. This typically includes disease-enriched TCRs — for example, TRBV9+ TCRs known to be associated with ankylosing spondylitis — as well as a handful of experimentally validated peptide sequences capable of activating these TCRs.

By combining this minimal disease-specific dataset with large-scale public TCR-peptide-HLA datasets from other diseases, themap-tools enables users to train a robust and generalizable model for the disease of interest.

To explore new candidate epitopes that may activate these TCRs, users can first employ NetMHCpan to predict HLA presentation of candidate peptides. Then, themap-tools can be used to evaluate the likelihood of these peptides being recognized by the disease-associated TCRs.

---

```{toctree}
:maxdepth: 1

Train_model
Screen_peptides
Predict_epitopes