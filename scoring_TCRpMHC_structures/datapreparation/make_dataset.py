## PACKAGES ##
import os
# set device to cuda
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
import re

import esm
import esm.inverse_folding

from Bio import SeqIO
import numpy as np
import pandas as pd
import glob
import torch 
import torch.nn.functional as F

## PATHS ##
dotenv_path = find_dotenv()
ROOTPATH = Path(find_dotenv()).parent
PROCESSEDPATH = Path(ROOTPATH, 'data/ESM-IF1')

## FUNCTIONS ##

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception("Relaxed argument not boolean")

def cmdline_args():
    # Make parser object
    usage = f"""
    # Make dataset for test as example
    python3 make_dataset.py \
    --input_dir data/raw/TCRpMHC_structures/solved/ \
    --output_dir data/ESM-IF1/peptide_features/solved/ \
    --raw_data data/raw/nettcr_train_swapped_peptide_ls_3_full_tcr.csv \
    --fasta_dir data/raw/fastafiles/solved_WT/ \
    --annotation multichain \
    --relaxed False
    """
    p = ArgumentParser(
        description="Make dataset",
        formatter_class=RawTextHelpFormatter,
        usage=usage,
    )
    p.add_argument(
        "--input_dir",
        required=True,
        help="Input directory that contains TCRpMHCmodels generated pdbs and csvs",
        metavar="FOLDER",
    )
    p.add_argument(
        "--output_dir",
        default=f"{PROCESSEDPATH}",
        required=False,
        help="Job output directory",
        metavar="FOLDER",
    )
    p.add_argument(
        "--raw_data",
        default=None,
        required=False,
        help="CSV file containing binder information",
        metavar="FILE",
    )
    p.add_argument(
        "--fasta_dir",
        required=True,
        help="Folder containing fasta files for complexes",
        metavar="FILE",
    )
    p.add_argument(
        "--annotation",
        default="multichain",
        required=False,
        help="Annotation strategy for ESM-IF1 embeddings (multichain or padding)",
    )
    p.add_argument(
        "--relaxed",
        default=False,
        required=False,
        type=str2bool,
        help="Relaxed structures (True or False)",
    )

    return p.parse_args()

#@jit
def get_logits(fpath, chain_ids, target_chain_id, model, alphabet, annotation):
    """ Function extracting ESM-IF1 predicted logits from structures """
    # input: pdb file path, all chain ids, target chain id, model (ESM IF1), alphabet (ESM), annotation (multichain or padding)
    # return: target chain logits and amino acid sequence

    device = torch.device('cuda')

    structure = esm.inverse_folding.util.load_structure(fpath, chain_ids)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

    # padding annotation
    if annotation == 'padding':
        target_seq = 'X' + native_seqs[target_chain_id] + 'X'
        native_seqs[target_chain_id] = 'X' + native_seqs[target_chain_id] + 'X'
        #add nan values 
        nan_coords = np.empty((1,3,3,))
        nan_coords[:] = np.nan
        coords[target_chain_id] = np.vstack([nan_coords, coords[target_chain_id], nan_coords])
        
    # multichain annotation
    else:
        target_seq = native_seqs[target_chain_id]

    # get logits 
    all_coords = esm.inverse_folding.multichain_util._concatenate_coords(coords, target_chain_id)

    batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
    batch = [(all_coords, None, target_seq)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(batch)
    coords, confidence, padding_mask = coords.float().to(device), confidence.to(device), padding_mask.to(device)

    prev_output_tokens = tokens[:, :-1].to(device)
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)

    # return logits and target seq based on annotation strategy 
    if annotation == 'padding':
        return logits[0].T[1:-1], target_seq[1:-1] 
    else:
        return logits[0].T, target_seq

def get_probs(logits):
    """ Function extracting probabilities from logits """
    # input: ESM logits 
    # return: probabilities 

    AA_pos_in_tok = {"A": 5, "R": 10, "N": 17, "D": 13, "C": 23, "Q": 16, "E": 9, "G": 6, "H": 21, "I": 12, "L": 4, "K": 15, "M": 20, "F": 18, "P": 14, "S": 8, "T": 11, "W": 22, "Y": 19, "V": 7}
    
    #get each positions logits/probabilities for the 20 AAs
    all_pos_probs = []
    for pos in logits: 
        pos_toks = []
        for AA in sorted(AA_pos_in_tok): 
            pos_toks.append(float(pos[AA_pos_in_tok[AA]]))
        pos_probs = F.softmax(torch.FloatTensor(pos_toks), dim=0)
        all_pos_probs.append(pos_probs)
    
    return all_pos_probs

def write_prob_tsv(tsv_out_dir, id_name, all_pos_probs):
    AA_pos_in_tok = {"A": 5, "R": 10, "N": 17, "D": 13, "C": 23, "Q": 16, "E": 9, "G": 6, "H": 21, "I": 12, "L": 4, "K": 15, "M": 20, "F": 18, "P": 14, "S": 8, "T": 11, "W": 22, "Y": 19, "V": 7}
    # write to tsv 
    with open(tsv_out_dir+id_name+'.tsv','w') as prob_tsv: 
        prob_tsv.write("\t" + "\t".join(sorted(AA_pos_in_tok)) + "\n")
        for i, pos in enumerate(all_pos_probs):
            freq = [str(x) for x in pos.tolist()]
            prob_tsv.write(str(i+1) + "\t" + "\t".join(freq) + "\n")

def compute_IF1_embeddings(pdb_path, chain_ids, target_chain_id, model, alphabet, tsv_out_dir, id_name, annotation='multichain'):
    """ Function extracting AA probs for structures from ESM-IF1 calculated logits """
    # input: pdb file path, all chain ids, target chain id, model (ESM-IF1), alphabet (ESM), annotation strategy (multichain or padding)
    # return: writes probabilities to tsv file and returns target chain AA sequence and probilities
    
    # assert annotation strategy is either multichain or padding 
    annotation = annotation.lower()
    if annotation not in ['multichain','padding']:
        log.error(f'Not valid annotation strategy')
        sys.exit(1)
    
    #get logits and probabilities 
    logits, target_seq = get_logits(pdb_path, chain_ids, target_chain_id, model, alphabet, annotation)
    all_pos_probs = get_probs(logits)
    
    #save/write all pos probs to tsv in interim
    write_prob_tsv(tsv_out_dir, id_name, all_pos_probs)
    
    return target_seq, all_pos_probs

def collect_embeddings_from_tsv(tsv_file, fasta_file):
    # get probabilities
    all_pos_probs = pd.read_csv(tsv_file, sep='\t', index_col=0).values.tolist()
    #all_pos_probs = cudf.read_csv(tsv_file, sep='\t', index_col=0).values.tolist()
    # get peptide sequence
    for chain in SeqIO.parse(fasta_file, "fasta"):
        if chain.id[-1] == 'P':
            target_seq = str(chain.seq)
        
    return target_seq, all_pos_probs

def get_peptide_scores(target_seq, all_pos_probs):
    """ Function caluclating position prob """
    # input: target seq = peptide sequence, all_pos_probs = probability matrix of all amino acids per position
    # return: the ESM-IF predicted probability of the correct amino acids (list)
    
    #initialize
    sorted_AA_idx = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    target_seq = target_seq.upper()
    pos_scores = []

    #calculate scores 
    for i, letter in enumerate(target_seq):
        pos_prob = float(all_pos_probs[i][sorted_AA_idx[letter]]) #extract prob
        pos_scores.append(pos_prob)

    return pos_scores

def extract_template_iden(file):
    """ Function averaging templates' sequence identity from TCRpMHCmodels """
    # input: TCRpMHCmodels csv file with template selection and sequence identities 
    # return: Average of peptide-, MHC-, TCRalpha, TCRbeta- and total template sequence identity

    seqid_df = pd.read_csv(file, header=0)
    #seqid_df = cudf.read_csv(file, header=0)
    pep_iden_avg = (sum(list(seqid_df["pep_identity"]))/len(list(seqid_df["pep_identity"])))/100
    mhc_iden_avg = (sum(list(seqid_df["mhc_identity"]))/len(list(seqid_df["mhc_identity"])))/100
    tcrA_iden_avg = (sum(list(seqid_df["tcrA_identity"]))/len(list(seqid_df["tcrA_identity"])))/100
    tcrB_iden_avg = (sum(list(seqid_df["tcrB_identity"]))/len(list(seqid_df["tcrB_identity"])))/100
    total_iden_avg = (sum(list(seqid_df["total_identity"]))/len(list(seqid_df["total_identity"])))/100
    
    return pep_iden_avg, mhc_iden_avg, tcrA_iden_avg, tcrB_iden_avg, total_iden_avg

def check_binders_partitions(raw_df, complex_info):
    """ Function checking binder status and partition for complex """
    # input: dataframe containing raw data information, dict of complex info 
    # return: updated complex info dict 

    for index,row in raw_df.iterrows():
        complex_name = 'complex_'+str(index)
        if complex_name in complex_info.keys():
            complex_info[complex_name] = complex_info[complex_name] + [row['binder'], row['partition']]
    return complex_info


def save_features_tsv(complex_info, output_dir, annotation="multichain", raw_csv=None):
    if raw_csv != None:
        features_df = pd.DataFrame.from_dict(complex_info, orient='index', 
                                    columns=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 
                                    'pep_iden_avg', 'mhc_iden_avg', 'tcrA_iden_avg', 'tcrB_iden_avg', 'total_iden_avg', 'binder', 'partition'])
    else:
        features_df = pd.DataFrame.from_dict(complex_info, orient='index', 
                                    columns=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 
                                    'pep_iden_avg', 'mhc_iden_avg', 'tcrA_iden_avg', 'tcrB_iden_avg', 'total_iden_avg'])
    
    #save in processed
    if annotation.lower() == "multichain":
        out_path = output_dir + 'features_multichain.tsv'
    elif annotation.lower() == "padding":
        out_path = output_dir + 'features_padding.tsv'
    features_df.to_csv(out_path, sep='\t')

def main(args):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    
    raw data: TCRpMHC structure models
    """
    # set device to cuda
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    # check device is cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device (CPU/GPU):", device)

    log.info('making final data set from raw data')

    # load model and alphabet 
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().cuda()
    log.info('IF1 model loaded...')
    
    #input arguments 
    input_dir = args.input_dir
    output_dir = args.output_dir
    raw_csv = args.raw_data
    fasta_dir = args.fasta_dir
    annotation = args.annotation
    relaxation = args.relaxed

    #files 
    if raw_csv != None: 
        raw_df = pd.read_csv(raw_csv, header=0) 
    
    if relaxation == True:
        split_string = "_TCR-pMHC_Repair\.pdb" 
        pdb_files = glob.glob(input_dir + '*_TCR-pMHC_Repair.pdb')
    else:  
        split_string = '_TCR-pMHC\.pdb'
        pdb_files = glob.glob(input_dir + '*_TCR-pMHC.pdb')
   
    no_files = len(pdb_files)
    log.info(no_files)

    #make sure directories exists 
    os.makedirs(output_dir, exist_ok=True)
    #tsv_out_dir = output_dir + 'tsv_files/'
    if annotation.lower() == "multichain":
        tsv_out_dir = output_dir + 'tsv_files_multichain/'
    elif annotation.lower() == "padding":
        tsv_out_dir = output_dir + 'tsv_files_padding/'
    os.makedirs(tsv_out_dir, exist_ok=True)

    ## iterate over each complex in directory
    complex_info = dict()
    count = 0
    for pdb_file in pdb_files:
        complex_name = re.split(split_string, pdb_file.split('/')[-1])[0]

        log.info(f'Starting {complex_name}...')
        csv_file = input_dir + complex_name + '-complex-templates.csv'
        
        if os.path.isfile(tsv_out_dir + complex_name + '.tsv'):
            # collect probs from existing .tsv file
            tsv_file = tsv_out_dir + complex_name + '.tsv'
            fasta_file = fasta_dir + complex_name + '.fasta'
            try:
                target_seq, all_pos_probs = collect_embeddings_from_tsv(tsv_file, fasta_file)
            # compute embs if .tsv file is empty
            except:
                target_seq, all_pos_probs = compute_IF1_embeddings(pdb_file, ['A','B','M','P'], 'P', 
                                                model, alphabet, tsv_out_dir, complex_name, annotation)
        else:
            #embeddings 
            target_seq, all_pos_probs = compute_IF1_embeddings(pdb_file, ['A','B','M','P'], 'P', 
                                                model, alphabet, tsv_out_dir, complex_name, annotation) 

        #score peptide 
        pos_scores = get_peptide_scores(target_seq, all_pos_probs)
        #get template sequence identities 
        pep_iden_avg, mhc_iden_avg, tcrA_iden_avg, tcrB_iden_avg, total_iden_avg = extract_template_iden(csv_file)
        #save 
        complex_info[complex_name] = pos_scores + [pep_iden_avg, mhc_iden_avg, tcrA_iden_avg, tcrB_iden_avg, total_iden_avg]
        count += 1
        log.info(f'{count}/{no_files} complexes done...')

    if raw_csv != None: 
        #get binder information
        complex_info = check_binders_partitions(raw_df, complex_info)

    #save features 
    save_features_tsv(complex_info, output_dir, annotation, raw_csv)
    
## MAIN ##
logging.basicConfig(level=logging.INFO, format="[{asctime}] {message}", style="{")
log = logging.getLogger(__name__)
if __name__ == '__main__':
    args = cmdline_args()
    main(args)
    log.info('End of program..')
