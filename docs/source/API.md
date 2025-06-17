# API Reference

## `HLA_vae` (in `themap.model`)

### `train_model(train_loader, lr, epochs, device)`

Train the HLA VAE model using MSE loss and KL divergence.

#### Parameters

| Name          | Type                            | Description                                         |
|---------------|----------------------------------|-----------------------------------------------------|
| `train_loader`| `torch.utils.data.DataLoader`   | Batches of input HLA tensors.                      |
| `lr`          | `float`                         | Learning rate.                                     |
| `epochs`      | `int`                           | Number of training epochs.                         |
| `device`      | `torch.device`                  | Device to run training (e.g., `cuda:0` or `cpu`).  |

#### Returns

None.

---

### `embed_hla(hla_loader, device)`

Generate latent embeddings for HLA sequences using the trained encoder.

#### Parameters

| Name          | Type                            | Description                                        |
|---------------|----------------------------------|----------------------------------------------------|
| `hla_loader`  | `torch.utils.data.DataLoader`   | Batches of input HLA sequences.                   |
| `device`      | `torch.device`                  | Computation device.                               |

#### Returns

`Tuple[np.ndarray, np.ndarray]`  
- `z_hla_mean`: latent mean vectors  
- `z_conv`: intermediate convolutional features

---

## `PEP_vae` (in `themap.model`)

### `train_model(train_loader, lr, epochs, device)`

Train peptide VAE using reconstruction + KL + alignment loss.

#### Parameters

| Name          | Type                            | Description                                       |
|---------------|----------------------------------|---------------------------------------------------|
| `train_loader`| `torch.utils.data.DataLoader`   | Batches of `(peptide, z_hla)` inputs.            |
| `lr`          | `float`                         | Learning rate.                                   |
| `epochs`      | `int`                           | Number of training epochs.                       |
| `device`      | `torch.device`                  | CUDA or CPU device.                              |

#### Returns

None.

---

### `embed_pep(data_loader, device)`

Embed peptides using the trained VAE encoder.

#### Parameters

| Name          | Type                            | Description                                       |
|---------------|----------------------------------|---------------------------------------------------|
| `data_loader` | `torch.utils.data.DataLoader`   | Peptide batches.                                 |
| `device`      | `torch.device`                  | Device for inference.                            |

#### Returns

`Tuple[np.ndarray, np.ndarray]`  
- `z_pep_mean`: latent mean embeddings  
- `z_conv`: intermediate conv features

---

## `THE` (in `themap.model`)

### `train_model(...)`

Train the full THE model using labeled TCR-target binding pairs.

#### Parameters

| Name           | Type              | Description |
|----------------|-------------------|-------------|
| `df`           | `pd.DataFrame`    | Training set with TCR, target, and label. |
| `num_epochs`   | `int`             | Number of training epochs. |
| `device`       | `torch.device`    | Training device. |
| `batch_size`   | `int`             | Batch size (default: 256). |
| `lr`           | `float`           | Learning rate (default: 1e-4). |
| `chain`        | `str`             | One of `'a'`, `'b'`, or `'ab'`. |
| `targets`      | `str`             | `'peptide'` or `'hla'`. |
| `hla_dict`     | `dict` or `None`  | Required if `targets='hla'`. |
| `df_add`       | `pd.DataFrame` or `None` | Additional data to append. |
| `neg_sampling` | `bool`            | Enable negative sampling (default: `True`). |
| `neg_resample` | `bool`            | Resample negatives each epoch (default: `True`). |
| `max_alpha`    | `int`             | Max alpha CDR3 length. |
| `max_beta`     | `int`             | Max beta CDR3 length. |
| `max_pep`      | `int`             | Max peptide length. |

#### Returns

None.

---

### `test_model(...)`

Evaluate trained THE model on a test set or test loader.

#### Parameters

| Name           | Type              | Description |
|----------------|-------------------|-------------|
| `df_test`      | `pd.DataFrame` or `None` | Test set DataFrame. |
| `batch_size`   | `int`             | Batch size. |
| `test_loader`  | `DataLoader` or `None` | If provided, skip DataFrame. |
| `hla_dict`     | `dict` or `None`  | Required for HLA targets. |
| `device`       | `str` or `torch.device` | CPU or CUDA. |
| `chain`        | `str`             | `'a'`, `'b'`, or `'ab'`. |
| `targets`      | `str`             | `'peptide'` or `'hla'`. |
| `max_alpha`    | `int`             | Max alpha CDR3 length. |
| `max_beta`     | `int`             | Max beta CDR3 length. |
| `max_pep`      | `int`             | Max peptide length. |

#### Returns

`Tuple[Tensor, Tensor, Tensor]`  
- `scores`: predicted binding scores  
- `alpha_attn`: attention for alpha  
- `beta_attn`: attention for beta