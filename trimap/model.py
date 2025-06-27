import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm
from trimap import net
from trimap import utils
import pandas as pd
import logging
import os
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
class HLA_vae(nn.Module):
    """
    Variational Autoencoder (VAE) model for HLA sequences.

    This model learns a latent representation of HLA amino acid sequences
    and reconstructs the original sequence through a decoder. It is trained
    to minimize reconstruction loss and KL divergence.

    Args:
        input_size (list[int]): Shape of input tensor (typically [channels, length]).
        latent_size (int): Dimension of latent representation.
    """
    def __init__(self, input_size, latent_size):
        super(HLA_vae, self).__init__()
        
        self.HLA_encoder = net.Encoder(input_size=input_size,hidden=latent_size)
        self.HLA_Linear = net.Linear_block(latent_size, latent_size, latent_size)
        self.HLA_decoder = net.Decoder(hidden=latent_size,output_size=input_size)
        
    def train_model(
        self,
        train_loader,
        lr,
        epochs,
        device,
    ):
        """
        Train the HLA VAE model using MSE loss and KL divergence.

        Args:
            train_loader (DataLoader): Dataloader providing batches of HLA input tensors.
            lr (float): Learning rate.
            epochs (int): Number of training epochs.
            device (torch.device): Device to run training on (CPU or GPU).
        """
        self.train()
        
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        MSELoss = nn.MSELoss()
        
        with tqdm(range(epochs), total=epochs, desc='Epochs') as tq:
            
            for epoch in tq:
                    
                epoch_loss = defaultdict(float)
                for i, hla in enumerate(train_loader):
                    
                    hla = hla[0].to(device)
                    
                    z_hla, _ = self.HLA_encoder(hla)
                    
                    z_hla, z_hla_mean, z_hla_logvar, x_hla = self.HLA_Linear(z_hla)
                    
                    hla_hat = self.HLA_decoder(x_hla)
                    
                    recon_loss = MSELoss(hla_hat, hla)
                    kl_loss = net.kl_div(z_hla_mean, z_hla_logvar)
                    
                    loss = {'KL_loss':1e-4*kl_loss, 'recon_loss':recon_loss}
                    
                    optimizer.zero_grad()
                    sum(loss.values()).backward()
                    
                    optimizer.step()
                    
                    for k,v in loss.items():
                        epoch_loss[k] += loss[k].item()
                        
                epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}
                epoch_info = ','.join(['{}={:.4f}'.format(k, v) for k,v in epoch_loss.items()])
                tq.set_postfix_str(epoch_info)
        
    def embed_hla(
        self,
        hla_loader,
        device
    ):
        """
        Generate latent embeddings for HLA sequences using the trained encoder.

        Args:
            hla_loader (DataLoader): Dataloader of input HLA sequences.
            device (torch.device): Device for model computation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - z_hla_mean: Latent mean vectors for HLA.
                - z_conv: Convolutional feature embeddings.
        """
        hla_embed = []
        conv_embed = []
        
        self.eval()

        with torch.no_grad():
            
            for i, data in enumerate(hla_loader):
                
                hla = data[0].to(device)
            
                z_hla, z_conv = self.HLA_encoder(hla)
                z_hla_mean, z_hla_logvar, x_hla = self.HLA_Linear(z_hla)
                
                hla_embed.append(z_hla_mean.detach().cpu().numpy())
                conv_embed.append(z_conv.detach().cpu().numpy())
                
        return np.concatenate(hla_embed, axis=0), np.concatenate(conv_embed, axis=0)
        
class PEP_vae(nn.Module):
    """
    Variational Autoencoder (VAE) for peptide (epitope) sequences.

    This model learns a latent encoding of peptide sequences and decodes them.
    It also supports alignment loss with HLA embeddings to enforce biological consistency.

    Args:
        input_size (list[int]): Shape of peptide input (e.g., [21, max_pep_length]).
        latent_size (int): Dimension of latent space.
    """
    def __init__(self, input_size, latent_size):
        super(PEP_vae, self).__init__()
        
        self.epitope_encoder = net.Encoder(input_size=input_size,hidden=latent_size)
        self.epitope_Linear = net.Linear_block(latent_size, latent_size, latent_size)
        self.epitope_decoder = net.Decoder(hidden=latent_size,output_size=input_size)
        
    def train_model(
        self,
        train_loader,
        lr,
        epochs,
        device,
    ):
        """
        Train the peptide VAE using reconstruction, KL, and alignment losses.

        Args:
            train_loader (DataLoader): Dataloader of (peptide, HLA) pairs.
            lr (float): Learning rate.
            epochs (int): Number of training epochs.
            device (torch.device): Target device for training.
        """
        self.train()
        
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        MSELoss = nn.MSELoss()
        
        with tqdm(range(epochs), total=epochs, desc='Epochs') as tq:
            
            for epoch in tq:
                    
                epoch_loss = defaultdict(float)
                
                for i, data in enumerate(train_loader):
                    
                    pep, z_hla = data
                    pep, z_hla = pep.to(device), z_hla.to(device)
                    
                    z_pep, _ = self.epitope_encoder(pep)
                    
                    z_pep, z_pep_mean, z_pep_logvar, x_pep = self.epitope_Linear(z_pep)
                    
                    pep_hat = self.epitope_decoder(x_pep)
                    
                    recon_loss = MSELoss(pep_hat, pep)
                    kl_loss = net.kl_div(z_pep_mean, z_pep_logvar)
                    
                    align_loss = MSELoss(z_pep_mean, z_hla)
                    
                    loss = {'KL_loss':1e-4*kl_loss, 'align_loss':5*align_loss, 'recon_loss':recon_loss}

                    optimizer.zero_grad()
                    sum(loss.values()).backward()
                    
                    optimizer.step()
                    
                    for k,v in loss.items():
                        epoch_loss[k] += loss[k].item()
                        
                epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}
                epoch_info = ','.join(['{}={:.4f}'.format(k, v) for k,v in epoch_loss.items()])
                tq.set_postfix_str(epoch_info) 
                    
    def embed_pep(
            self,
            data_loader,
            device
    ):
        """
        Embed peptides into latent space using the trained encoder.

        Args:
            data_loader (DataLoader): Input peptide batches.
            device (torch.device): Device for inference.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - z_pep_mean: Latent peptide embeddings.
                - z_conv: Intermediate convolutional features.
        """
        pep_embed = []
        conv_embed = []
        
        self.eval()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                
                data = data[0].to(device)
                
                z_pep, z_conv = self.epitope_encoder(data)
                
                z_pep_mean, _, _ = self.epitope_Linear(z_pep)
                
                pep_embed.append(z_pep_mean.detach().cpu().numpy())
                conv_embed.append(z_conv.detach().cpu().numpy())

        return np.concatenate(pep_embed, axis=0), np.concatenate(conv_embed, axis=0)

class TCRbind(nn.Module):
    """
    THE (TCR-HLA-Epitope) model for predicting T-cell epitope binding.

    This model integrates TCR alpha/beta chain embeddings with target embeddings (HLA or peptide),
    uses Bilinear Attention Network (BAN) layers, and outputs a classification score.

    Args:
        ban_heads (int): Number of attention heads in BAN.
        mlp (list[int]): MLP layer sizes [input, hidden, output].
        v_dim (int): Feature dimension of the target (e.g., peptide or HLA).
        q_dim (int): Feature dimension of TCR chains.
    """
    def __init__(self, ban_heads=8, mlp=[512, 128, 32], v_dim=256, q_dim=1024):
        super(TCRbind, self).__init__()
        self.bcn_alpha = weight_norm(net.BANLayer(v_dim=v_dim, q_dim=q_dim, h_dim=mlp[0], h_out=ban_heads), name='h_mat', dim=None)
        self.bcn_beta = weight_norm(net.BANLayer(v_dim=v_dim, q_dim=q_dim, h_dim=mlp[0], h_out=ban_heads), name='h_mat', dim=None)
        self.classifier = net.MLPDecoder(in_dim=mlp[0]*2, hidden_dim=mlp[1], out_dim=mlp[2], binary=1)
        self.alpha_dict = None
        self.beta_dict = None

    def load_dict(self, device, df_data, max_alpha, max_beta, re_embed):
        """
        Load or create TCR α/β embedding dictionaries using ProtBERT.

        Args:
            device (torch.device): Target device for embedding.
            df_data (pd.DataFrame): DataFrame with 'alpha' and 'beta' sequences.
            max_alpha (int): Maximum alpha sequence length.
            max_beta (int): Maximum beta sequence length.
            re_embed (bool): If True, force re-embedding even if cache exists.
        """
        from trimap.BERT import ProtBERT_embed as PB
        prot_bert = PB(device)

        # Generalized loader/creator for a chain ('alpha' or 'beta')
        def get_or_create_dict(chain, max_len):
            path = f'{chain}_dict.pt'
            if os.path.exists(path) and not re_embed:
                logger.info(f'Loading {path}')
                chain_dict = torch.load(path)
            else:
                logger.info(f'{path} not found, generating...')
                unique_seqs = df_data[chain].unique()
                embeddings = prot_bert.embed_seq(unique_seqs, batch_size=256, length=max_len).cpu()
                chain_dict = dict(zip(unique_seqs, embeddings))
                torch.save(chain_dict, path)
                logger.info(f'Saved {path}')
            return chain_dict

        # Add new sequences not in existing dict
        def update_with_new_seqs(chain_dict, chain, max_len):
            all_seqs = df_data[chain].unique()
            new_seqs = [s for s in all_seqs if s not in chain_dict]
            if not new_seqs:
                logger.info(f'No new {chain} sequences found')
                return chain_dict

            logger.info(f'Found new {chain} sequences, embedding...')
            new_embeddings = prot_bert.embed_seq(new_seqs, batch_size=256, length=max_len).cpu()
            for seq, emb in zip(new_seqs, new_embeddings):
                chain_dict[seq] = emb
            torch.save(chain_dict, f'{chain}_dict.pt')
            logger.info(f'Updated and saved {chain}_dict.pt')
            return chain_dict

        # Process alpha and beta chains
        self.alpha_dict = update_with_new_seqs(get_or_create_dict('alpha', max_alpha), 'alpha', max_alpha)
        self.beta_dict = update_with_new_seqs(get_or_create_dict('beta', max_beta), 'beta', max_beta)
        
    def create_dataloader(
        self, 
        df_data,
        batch_size,
        device, 
        target, 
        mode, 
        max_alpha,
        max_beta,
        max_pep,
        pep_model_dir='pep_model.pt',
        hla_model_dir='hla_model.pt',
        hla_dict=None,
        load_dict=True,
        re_embed=False
    ):
        """
        Create DataLoader for training or testing the THE model.

        Depending on `target`, the function embeds HLA or peptide inputs,
        and returns batched tensors including TCR and target embeddings.

        Args:
            df_data (pd.DataFrame): Data with columns for alpha, beta, epitope/HLA, etc.
            batch_size (int): Batch size for DataLoader.
            device (torch.device): Computation device.
            target (str): 'peptide' or 'hla'.
            mode (str): 'train' or 'test'.
            max_alpha (int): Maximum length for alpha CDR3 sequences.
            max_beta (int): Maximum length for beta CDR3 sequences.
            max_pep (int): Maximum length for peptide sequences (if target is peptide).
            pep_model_dir (str): Path to pre-trained peptide VAE model.
            hla_model_dir (str): Path to pre-trained HLA VAE model.
            hla_dict (dict, optional): Dictionary mapping HLA names to amino acid sequences.
            load_dict (bool): If True, load/create TCR α/β embedding dictionaries.
            re_embed (bool): If True, force re-embedding of TCR sequences.
        Raises:
            ValueError: If `target` is not 'peptide' or 'hla'.
            ValueError: If `mode` is not 'train' or 'test'.
        
        Returns:
            torch.utils.data.DataLoader: Batched model input.
        """
        
        if load_dict:
            self.load_dict(device, df_data, max_alpha, max_beta, re_embed)
        
        alpha = df_data['alpha'].values.tolist()
        beta = df_data['beta'].values.tolist()
        alpha = [self.alpha_dict[i] for i in alpha]
        beta = [self.beta_dict[i] for i in beta]
        
        alpha = torch.stack(alpha)
        beta = torch.stack(beta)

        if target == 'peptide':
            epi_train = utils.amino_acid_encode(df_data['Epitope'].tolist(),max_pep)
            epi_train = torch.from_numpy(epi_train).transpose(1,2).float()
            
            pep_model = PEP_vae(input_size=[21, max_pep],latent_size=256).to(device)
            pep_model.load_state_dict(torch.load(pep_model_dir))
            
            dataset = torch.utils.data.TensorDataset(epi_train)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            _, pep_embed = pep_model.embed_pep(dataloader, device=device)
            target_embed = torch.from_numpy(pep_embed).transpose(1, 2).float()
            
        elif target == 'hla':
            hla_aa = list(hla_dict.values())
            hla_name = list(hla_dict.keys())
            hla_name = ['HLA-'+i for i in hla_name]
            HLA_lib_array = utils.amino_acid_encode(hla_aa, len(hla_aa[0]))
            HLA_lib_tensor = torch.from_numpy(HLA_lib_array).transpose(1,2).float()
            HLA_dict = {}
            for i in range(len(hla_name)):
                HLA_dict[hla_name[i]] = HLA_lib_tensor[i]
                
            HLA_embed = [HLA_dict[i] for i in df_data['HLA'].tolist()]
            HLA_embed = torch.stack(HLA_embed).float()
            
            hla_dataset = torch.utils.data.TensorDataset(HLA_embed)
            hla_test_loader = torch.utils.data.DataLoader(hla_dataset, batch_size=batch_size, shuffle=False)
            hla_model = HLA_vae(input_size=[21, len(hla_aa[0])], latent_size=256).to(device)
            hla_model.load_state_dict(torch.load(hla_model_dir))
            _, hla_embed = hla_model.embed_hla(hla_test_loader, device)
            target_embed = torch.from_numpy(hla_embed).transpose(1, 2).float()
        
        else:
            raise ValueError('target must be peptide or hla!')
        
        if mode == 'train':
            labels = torch.from_numpy(df_data['label'].values)
            dataset = torch.utils.data.TensorDataset(alpha, beta, target_embed, labels)
            shuffle = True
        elif mode == 'test':
            dataset = torch.utils.data.TensorDataset(alpha, beta, target_embed)
            shuffle = False
        else:
            raise ValueError("mode must be 'train' or 'test'")

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
        return loader
    
    def _sampling(self, df_data):
        negative_samples = utils.negative_sampling(df_data)
        df_data['label'] = 1
        negative_samples['label'] = 0
        df_data = pd.concat([df_data, negative_samples])
        df_data.reset_index(drop=True, inplace=True)
        return df_data
    
    def train_model(
        self,
        df,
        num_epochs,
        device,
        batch_size=256,
        lr=1e-4,
        chain='ab',
        targets='peptide',
        hla_dict=None,
        df_add=None,
        neg_sampling=True,
        neg_resample=True,
        max_alpha=125,
        max_beta=127, 
        max_pep=14
    ):
        """
        Train the THE model to predict TCR-target binding.

        This function performs supervised training using TCR alpha/beta embeddings 
        and target embeddings (peptide or HLA). It optionally supports negative 
        sampling and dynamic resampling across epochs.

        Args:
            df (pd.DataFrame): Primary training dataset containing TCR sequences, 
                target peptides or HLAs, and binary labels.
            num_epochs (int): Number of training epochs.
            device (torch.device): Device to run training on (e.g., 'cuda:0' or 'cpu').
            batch_size (int): Batch size used in training. Default is 256.
            lr (float): Learning rate for the optimizer. Default is 1e-4.
            chain (str): Specifies which TCR chains to use in the model.
                Options are 'ab' (both), 'a', or 'b'. Default is 'ab'.
            targets (str): Specifies the type of target to embed — either 'peptide' or 'hla'.
            hla_dict (dict, optional): Mapping of HLA allele names to amino acid sequences.
                Required if targets='hla'.
            df_add (pd.DataFrame, optional): Additional training samples to be appended 
                (e.g., auxiliary positive data). Default is None.
            neg_sampling (bool): Whether to use negative sampling during training. Default is True.
            neg_resample (bool): If True, regenerates negative samples at each epoch. Default is True.
            max_alpha (int): Maximum length of alpha CDR3 sequences (for padding/embedding).
            max_beta (int): Maximum length of beta CDR3 sequences.
            max_pep (int): Maximum length of peptide sequences (if using peptide as target).

        Behavior:
            - If `neg_sampling` is True, negative samples are generated on-the-fly per epoch.
            - If `df_add` is provided, it is concatenated to the training set each epoch.
            - Model is trained using binary cross-entropy loss (`BCEWithLogitsLoss`).
            - ROC-AUC is computed each epoch for performance tracking.

        Returns:
            None. (Trained weights are updated in-place.)
        """

        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        loss_fct = nn.BCEWithLogitsLoss()
            
        df_data = df.copy()
        # If no negative sampling, prepare loader only once
        if not neg_sampling:
            if df_add is not None:
                df_data = pd.concat([df_data, df_add]).reset_index(drop=True)
                
            train_loader = self.create_dataloader(
                df_data, batch_size, device, targets, 'train',
                max_alpha=max_alpha, max_beta=max_beta, max_pep=max_pep, hla_dict=hla_dict,
            )

        logger.info('Training...')
        
        for epoch in range(num_epochs):
            # Prepare train loader if using negative sampling
            if neg_sampling and (epoch == 0 or neg_resample):
                df_data = self._sampling(df.copy())
                if df_add is not None:
                    df_data = pd.concat([df_data, df_add]).reset_index(drop=True)
                train_loader = self.create_dataloader(
                    df_data, batch_size, device, targets, 'train',
                    max_alpha=max_alpha, max_beta=max_beta, max_pep=max_pep, hla_dict=hla_dict,
                    load_dict=(epoch == 0)
                )

            scores_all, labels_all = [], []

            for data in tqdm(train_loader):
                if len(data[0]) == 1:
                    continue  # skip batch of size 1

                alpha, beta, target, labels = [x.float().to(device) for x in data]

                # Forward
                logits = self.forward_chain(target, alpha, beta, chain=chain, return_attn=False)
                scores = self.classifier(logits).squeeze()

                loss = loss_fct(scores, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()

                scores_all.append(scores.detach().cpu())
                labels_all.append(labels.cpu())

            # Evaluation
            scores = torch.cat(scores_all)
            labels = torch.cat(labels_all)
            roc = roc_auc_score(labels, scores)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, ROC: {roc:.4f}')
            torch.cuda.empty_cache()

    def forward_chain(self, target, alpha, beta, chain='ab', return_attn=False):
        """
        Perform forward computation for alpha/b with BAN attention.

        Args:
            target (Tensor): target embedding (e.g., peptide or HLA)
            alpha (Tensor): alpha chain embedding
            beta (Tensor): beta chain embedding
            chain (str): one of 'ab', 'a', or 'b'
            return_attn (bool): whether to return attention maps

        Returns:
            logits (Tensor): concatenated logits
            (Optional) attn_a, attn_b: attention weights if return_attn=True
        """
        attn_a = attn_b = None

        if chain == 'ab':
            a_logits, attn_a = self.bcn_alpha(target, alpha, softmax=True)
            b_logits, attn_b = self.bcn_beta(target, beta, softmax=True)
        elif chain == 'a':
            a_logits, attn_a = self.bcn_alpha(target, alpha, softmax=True)
            b_logits = torch.zeros_like(a_logits)
        elif chain == 'b':
            b_logits, attn_b = self.bcn_beta(target, beta, softmax=True)
            a_logits = torch.zeros_like(b_logits)
        else:
            raise ValueError("chain must be 'ab', 'a', or 'b'")

        logits = torch.cat([a_logits, b_logits], dim=1)

        if return_attn:
            return logits, attn_a, attn_b
        else:
            return logits
        
    def test_model(
        self,
        df_test=None,
        batch_size=256,
        test_loader=None,
        hla_dict=None,
        device='cuda:0',
        chain='ab',
        targets='peptide',
        max_alpha=125,
        max_beta=127,
        max_pep=14
    ):
        """
        Evaluate the THE model on a held-out test dataset.

        This function performs inference using the trained model and returns predicted 
        binding scores as well as attention weights (if applicable). Input can be provided 
        either as a DataFrame (`df_test`) or a pre-built `test_loader`.

        Args:
            df_test (pd.DataFrame, optional): Test dataset containing columns for 'alpha', 
                'beta', 'Epitope' or 'HLA'. Required if `test_loader` is not provided.
            batch_size (int): Batch size used during inference. Default is 256.
            test_loader (DataLoader, optional): Precomputed DataLoader for testing.
                If provided, `df_test` is ignored.
            hla_dict (dict, optional): Dictionary mapping HLA names to amino acid sequences.
                Required if `targets='hla'`.
            device (str or torch.device): Computation device (e.g., 'cuda:0' or 'cpu').
            chain (str): Specifies which TCR chain(s) to use: 'ab', 'a', or 'b'.
            targets (str): Specifies target type: 'peptide' or 'hla'.
            max_alpha (int): Maximum sequence length for alpha chains (for padding).
            max_beta (int): Maximum sequence length for beta chains.
            max_pep (int): Maximum peptide length (for encoding peptide targets).

        Returns:
            Tuple:
                - scores (torch.Tensor): Predicted probabilities (after sigmoid).
                - alpha_attn (torch.Tensor or None): Attention weights for alpha chain (if used).
                - beta_attn (torch.Tensor or None): Attention weights for beta chain (if used).
        """
        self.eval()

        if test_loader is None:
            test_loader = self.create_dataloader(
                df_test,
                batch_size=batch_size,
                device=device,
                target=targets,
                mode='test',
                max_alpha=max_alpha,
                max_beta=max_beta,
                max_pep=max_pep,
                 hla_dict=hla_dict,
            )

        logger.info('Predicting...')
        
        score_all = []
        attn_dict = {'a': [], 'b': []}

        with torch.no_grad():
            for data in tqdm(test_loader):
                alpha, beta, target = [x.float().to(device) for x in data[:3]]

                # Forward
                logits, attn_a, attn_b = self.forward_chain(target, alpha, beta, chain=chain, return_attn=True)
                scores = torch.sigmoid(self.classifier(logits).squeeze())

                score_all.append(scores.cpu())
                if attn_a is not None:
                    attn_dict['a'].append(attn_a.cpu())
                if attn_b is not None:
                    attn_dict['b'].append(attn_b.cpu())

        scores = torch.cat(score_all)

        alpha_attn = torch.cat(attn_dict['a'], dim=0) if attn_dict['a'] else None
        beta_attn = torch.cat(attn_dict['b'], dim=0) if attn_dict['b'] else None

        return scores, alpha_attn, beta_attn
    