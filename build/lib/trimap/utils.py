import pandas as pd
import csv
import numpy as np
from io import StringIO
import torch
import re

def _file2dict(filename,key_fields,store_fields,delimiter='\t'):
    """Read file to a dictionary.
    key_fields: fields to be used as keys
    store_fields: fields to be saved as a list
    delimiter: delimiter used in the given file."""
    dictionary={}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile,delimiter=delimiter)
        for row in reader:
            keys = [row[k] for k in key_fields]
            store= [row[s] for s in store_fields]

            sub_dict = dictionary
            for key in keys[:-1]:
                if key not in sub_dict:
                    sub_dict[key] = {}
                sub_dict = sub_dict[key]
            key = keys[-1]
            if key not in sub_dict:
                sub_dict[key] = []
            sub_dict[key].append(store)
    return dictionary

def _get_protseqs_ntseqs(chain='B', library_dir='library/'):
    """returns sequence dictioaries for genes: protseqsV, protseqsJ, nucseqsV, nucseqsJ"""
    seq_dicts=[]
    for gene,type in zip(['v','j','v','j'],['aa','aa','nt','nt']):
        name = library_dir+'tr'+chain.lower()+gene+'s_'+type+'.tsv'
        sdict = _file2dict(name,key_fields=['Allele'],store_fields=[type+'_seq'])
        for g in sdict:
            sdict[g]=sdict[g][0][0]
        seq_dicts.append(sdict)
    return seq_dicts

def determine_tcr_seq_vj(cdr3, V, J, chain, guess01=False, library_dir='library/'):
    if chain not in ['A', 'B']:
        raise ValueError("Chain must be 'A' or 'B'")

    protV, protJ = _get_protseqs_ntseqs(chain=chain, library_dir=library_dir)[:2]

    tcr_list = []

    for cdr3_, V_, J_ in zip(cdr3, V, J):
        if guess01:
            if '*' not in V_:
                V_ += '*01'
            if '*' not in J_:
                J_ += '*01'
        try:
            pv = protV[V_]
            pj = protJ[J_]
        except KeyError as e:
            raise KeyError(f"Cannot find {e.args[0]} in library for chain {chain}.")

        v_prefix = pv[:pv.rfind('C')]

        match = re.search(r'[FW]G.G', pj)
        if not match:
            raise ValueError(f"Pattern [FW]G.G not found in J protein {J_}.")
        j_suffix = pj[match.start() + 1:]

        tcr_seq = v_prefix + cdr3_ + j_suffix
        tcr_list.append(tcr_seq)

    return tcr_list

        
########################### One Hot ##########################   
aa_dict_one_hot = {'A': 0,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'K': 8,'L': 9,
           'M': 10,'N': 11,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'V': 17,
           'W': 18,'Y': 19,'X': 20} 
########################### Blosum ########################## 
BLOSUM50_MATRIX = pd.read_table(StringIO(u"""                                                                                      
    A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  J  Z  X  *                                                           
    A  5 -2 -1 -2 -1 -1 -1  0 -2 -1 -2 -1 -1 -3 -1  1  0 -3 -2  0 -2 -2 -1 -1 -5                                                           
    R -2  7 -1 -2 -4  1  0 -3  0 -4 -3  3 -2 -3 -3 -1 -1 -3 -1 -3 -1 -3  0 -1 -5                                                           
    N -1 -1  7  2 -2  0  0  0  1 -3 -4  0 -2 -4 -2  1  0 -4 -2 -3  5 -4  0 -1 -5                                                           
    D -2 -2  2  8 -4  0  2 -1 -1 -4 -4 -1 -4 -5 -1  0 -1 -5 -3 -4  6 -4  1 -1 -5                                                           
    C -1 -4 -2 -4 13 -3 -3 -3 -3 -2 -2 -3 -2 -2 -4 -1 -1 -5 -3 -1 -3 -2 -3 -1 -5                                                           
    Q -1  1  0  0 -3  7  2 -2  1 -3 -2  2  0 -4 -1  0 -1 -1 -1 -3  0 -3  4 -1 -5                                                           
    E -1  0  0  2 -3  2  6 -3  0 -4 -3  1 -2 -3 -1 -1 -1 -3 -2 -3  1 -3  5 -1 -5                                                           
    G  0 -3  0 -1 -3 -2 -3  8 -2 -4 -4 -2 -3 -4 -2  0 -2 -3 -3 -4 -1 -4 -2 -1 -5                                                           
    H -2  0  1 -1 -3  1  0 -2 10 -4 -3  0 -1 -1 -2 -1 -2 -3  2 -4  0 -3  0 -1 -5                                                          
    I -1 -4 -3 -4 -2 -3 -4 -4 -4  5  2 -3  2  0 -3 -3 -1 -3 -1  4 -4  4 -3 -1 -5                                                           
    L -2 -3 -4 -4 -2 -2 -3 -4 -3  2  5 -3  3  1 -4 -3 -1 -2 -1  1 -4  4 -3 -1 -5                                                           
    K -1  3  0 -1 -3  2  1 -2  0 -3 -3  6 -2 -4 -1  0 -1 -3 -2 -3  0 -3  1 -1 -5                                                           
    M -1 -2 -2 -4 -2  0 -2 -3 -1  2  3 -2  7  0 -3 -2 -1 -1  0  1 -3  2 -1 -1 -5                                                           
    F -3 -3 -4 -5 -2 -4 -3 -4 -1  0  1 -4  0  8 -4 -3 -2  1  4 -1 -4  1 -4 -1 -5                                                           
    P -1 -3 -2 -1 -4 -1 -1 -2 -2 -3 -4 -1 -3 -4 10 -1 -1 -4 -3 -3 -2 -3 -1 -1 -5                                                           
    S  1 -1  1  0 -1  0 -1  0 -1 -3 -3  0 -2 -3 -1  5  2 -4 -2 -2  0 -3  0 -1 -5                                                           
    T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  2  5 -3 -2  0  0 -1 -1 -1 -5                                                           
    W -3 -3 -4 -5 -5 -1 -3 -3 -3 -3 -2 -3 -1  1 -4 -4 -3 15  2 -3 -5 -2 -2 -1 -5                                                           
    Y -2 -1 -2 -3 -3 -1 -2 -3  2 -1 -1 -2  0  4 -3 -2 -2  2  8 -1 -3 -1 -2 -1 -5                                                           
    V  0 -3 -3 -4 -1 -3 -3 -4 -4  4  1 -3  1 -1 -3 -2  0 -3 -1  5 -3  2 -3 -1 -5                                                           
    B -2 -1  5  6 -3  0  1 -1  0 -4 -4  0 -3 -4 -2  0  0 -5 -3 -3  6 -4  1 -1 -5                                                           
    J -2 -3 -4 -4 -2 -3 -3 -4 -3  4  4 -3  2  1 -3 -3 -1 -2 -1  2 -4  4 -3 -1 -5                                                           
    Z -1  0  0  1 -3  4  5 -2  0 -3 -3  1 -1 -4 -1  0 -1 -2 -2 -3  1 -3  5 -1 -5                                                           
    X -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -5                                                           
    * -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5  1                                                           
"""), sep='\s+').loc[list(aa_dict_one_hot.keys()), list(aa_dict_one_hot.keys())]
assert (BLOSUM50_MATRIX == BLOSUM50_MATRIX.T).all().all()

ENCODING_DATA_FRAMES = {
    "BLOSUM50": BLOSUM50_MATRIX
}

def _aa_encode(sequence, maxlen, encoding_method):
    sequence = sequence.replace(u'\xa0', u'').upper()
    if len(sequence) > maxlen:
        raise ValueError(f"Sequence {sequence} is longer than maxlen {maxlen}.")

    # convert amino acids to indices
    indices = [aa_dict_one_hot.get(aa, 20) for aa in sequence]  # 20 = 'X' padding
    indices += [20] * (maxlen - len(indices))  # padding
    result = ENCODING_DATA_FRAMES[encoding_method].iloc[indices]
    return np.asarray(result)

def amino_acid_encode(dataset, maxlen, encoding_method='BLOSUM50', encode_func=_aa_encode):
    pos = 0
    array = np.zeros((len(dataset), maxlen, 21), dtype=np.float32)
    cache = {}
    for item in dataset:
        if item not in cache:
            array[pos] = encode_func(item, maxlen, encoding_method).reshape(1, maxlen, 21)
            cache[item] = array[pos]
        else:
            array[pos] = cache[item]
        pos += 1
    return array

def process_phmc(df, pep_max_len=14, hla_max_len=34):

    epitope_array = amino_acid_encode(df['Epitope'].tolist(), pep_max_len)
    HLA_array = amino_acid_encode(df['HLA_aa'].tolist(), hla_max_len)
    
    epitope_tensor = torch.from_numpy(epitope_array).transpose(1,2).float()
    HLA_tensor = torch.from_numpy(HLA_array).transpose(1,2).float()
    
    return df, epitope_tensor, HLA_tensor

def negative_sampling(df, negative_ratio=3):
    original_tuples = set(map(tuple, df.to_numpy()))
    cols = ['alpha', 'beta', 'V_alpha', 'J_alpha', 'V_beta', 'J_beta']
    negative_samples = []

    for _ in range(negative_ratio):
        shuffled = df[cols].sample(frac=1).reset_index(drop=True)
        shuffled_df = df.copy()
        shuffled_df[cols] = shuffled.values

        new_samples = shuffled_df[~shuffled_df.apply(tuple, axis=1).isin(original_tuples)]
        negative_samples.append(new_samples)

    return pd.concat(negative_samples, ignore_index=True)
