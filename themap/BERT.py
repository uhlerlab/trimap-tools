import torch
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
import re
from tqdm import tqdm
import csv

class ProtBERT_embed:
    def __init__(self, device):
        self.device = device
        self.protbert_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    
    def embed_seq(self, seq, batch_size, length=None):
        # use protBert to encode epitopeseq
        if length is None:
            max_length = len(max(seq, key=lambda x: len(x)))
        else:
            max_length = length
        seq_embed = torch.zeros(len(seq), max_length, 1024).float()
        epoch = np.ceil(len(seq)/batch_size).astype(int)
        for i in tqdm(range(epoch)):
            sequence_examples = seq[batch_size*i:batch_size*(i+1)]
            # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
            sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
            # tokenize sequences and pad up to the longest sequence in the batch
            ids = self.tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(self.device)
            attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
            # generate embeddings
            with torch.no_grad():
                embedding_rpr = self.protbert_model(input_ids=input_ids,attention_mask=attention_mask)
            embed = embedding_rpr.last_hidden_state.cpu().detach()
            for j in range(len(embed)):
                seq_embed[batch_size*i+j,0:len(seq[batch_size*i+j]),:] = embed[j, 0:len(seq[batch_size*i+j])]
            torch.cuda.empty_cache()
        return seq_embed

class Atchley_embed:
    def __init__(self):
        pass

    def aamapping_TCR(self, peptideSeq,aa_dict,encode_dim):
        #Transform aa seqs to Atchley's factors.                                                                                              
        peptideArray = []
        if len(peptideSeq)>encode_dim:
            print('Length: '+str(len(peptideSeq))+' over bound!')
            peptideSeq=peptideSeq[0:encode_dim]
        for aa_single in peptideSeq:
            try:
                peptideArray.append(aa_dict[aa_single])
            except KeyError:
                print('Not proper aaSeqs: '+peptideSeq)
                peptideArray.append(np.zeros(5,dtype='float64'))
        for i in range(0,encode_dim-len(peptideSeq)):
            peptideArray.append(np.zeros(5,dtype='float64'))
        return np.asarray(peptideArray)

    def TCRMap(self, dataset,aa_dict,encode_dim):
        #Wrapper of aamapping                                                                                                                 
        for i in range(0,len(dataset)):
            if i==0:
                TCR_array=self.aamapping_TCR(dataset[i],aa_dict,encode_dim).reshape(1,encode_dim,5)
            else:
                TCR_array=np.append(TCR_array,self.aamapping_TCR(dataset[i],aa_dict,encode_dim).reshape(1,encode_dim,5),axis=0)
        print('TCRMap done!')
        return TCR_array
    
    def embed_seq(self, seq, encode_dim):
        aa_dict_dir='../Data/Atchley_factors.csv'
        aa_dict_atchley={}
        with open(aa_dict_dir,'r') as aa:
            aa_reader=csv.reader(aa)
            next(aa_reader, None)
            for rows in aa_reader:
                aa_name=rows[0]
                aa_factor=rows[1:len(rows)]
                aa_dict_atchley[aa_name]=np.asarray(aa_factor,dtype='float')
                
        seq_embed = self.TCRMap(seq, aa_dict_atchley, encode_dim)
        return torch.from_numpy(seq_embed).float()
        