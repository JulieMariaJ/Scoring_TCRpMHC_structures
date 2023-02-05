import os
import torch

# set device to cuda
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# check device is cuda
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device (CPU/GPU):", device)

import re
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

import random 
import copy

import glob

import esm
import esm.inverse_folding

from scipy import stats
from scipy.optimize import linprog

import Bio
from Bio import SeqIO
from Bio import PDB
from Bio.PDB import *
#from Bio.PDB import NeighborSearch

import sys
import torch.nn.functional as F

from sklearn import metrics

from prody import parsePDB, fetchPDB, Contacts, getCoords, Chain

# load model and alphabet 
esm_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
esm_model = esm_model.eval()#.cuda()

def get_logits(fpath, chain_ids, target_chain_id, model, alphabet, annotation):
    """ Function extracting ESM-IF1 predicted logits from structures """
    # input: pdb file path, all chain ids, target chain id, model (ESM IF1), alphabet (ESM), annotation (multichain or padding)
    # return: target chain logits and amino acid sequence

    device = torch.device('cpu') 
    #device = torch.device('cuda')

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

def get_single_prob(AA_logit):
    #get probabilities for the 20 AAs for single residue
    AA_pos_in_tok = {"A": 5, "R": 10, "N": 17, "D": 13, "C": 23, "Q": 16, "E": 9, "G": 6, "H": 21, "I": 12, "L": 4, "K": 15, "M": 20, "F": 18, "P": 14, "S": 8, "T": 11, "W": 22, "Y": 19, "V": 7}
    pos_toks = []
    for AA in sorted(AA_pos_in_tok): 
        pos_toks.append(float(AA_logit[AA_pos_in_tok[AA]]))
    pos_probs = F.softmax(torch.FloatTensor(pos_toks), dim=0)
    return pos_probs

def index_conversion(chain_structure):
    ''' returns dict w. res identifier as key and index as val '''
    idx_convert = {}
    for index, residue in enumerate(chain_structure):
        idx_convert[residue.id[1]] = index
    return idx_convert

def extract_TCR_neighbours(pdb_structure):
    # get model with prody
    pdb_structure_model = parsePDB(pdb_structure)
    tcr_chains = ['A', 'B']

    # collect num TCR neighbours + idx 
    TCR_num_neighbors = {"A": [], "B": []}
    TCR_inter_res_idx = {"A": [], "B": []}
    # iteration over peptide positions
    for i, res in enumerate(pdb_structure_model.getHierView()['P']):
        # iterate over tcr chains
        for tcr_chain in tcr_chains:
            inter_atoms = pdb_structure_model.select('same residue as chain ' + tcr_chain + ' within 4 of somepoint', somepoint = np.array(res.getCoords()))
            
            if inter_atoms != None:
                inter_res_idx = set()
                for atom in inter_atoms:
                    #res_idx = atom.getResindex()
                    res_idx = atom.getResnum()
                    res_name = atom.getResname()
                    inter_res_idx.add((res_idx,res_name))
            
                    #print(tcr_chain, res_idx, atom.getResname())
                TCR_num_neighbors[tcr_chain].append(len(inter_res_idx))
                TCR_inter_res_idx[tcr_chain].append(inter_res_idx)
            else:
                TCR_num_neighbors[tcr_chain].append(0)      # add 0 neighbours + None idx if no hit
                TCR_inter_res_idx[tcr_chain].append(None)
    
    return TCR_num_neighbors, TCR_inter_res_idx

## dict for AA conversion from 3-letter to 1-letter code
AA_conversion = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'HOH': ''} # remove water


##########

# get logits
chain_ids = ['A', 'B', 'M', 'P']
tcr_chains = ['A', 'B']
#neighbour_outpath = '/scratch/s174595/master_project/data/processed/features_w66_t95/'
#neighbour_outpath = '/scratch/s174595/master_project/data/processed/features_GLC_w66_t95/'
#neighbour_outpath = '/scratch/s174595/master_project/data/processed/features_NLV_w66_t95/'
neighbour_outpath = '/scratch/s174595/master_project/data/processed/features_YLQPRTFLL_w66_t95/'
#all_structures = glob.glob('/scratch/s174595/TCRpMHCmodels_new/tcrpmhc_models/GIL_peptides_StitchR/pdb_models_w66_t95/complex_*_TCR-pMHC.pdb')
#all_structures = glob.glob('/scratch/s174564/TCRpMHCmodels_new/tcrpmhc_models/notGIL_peptides_StitchR/pdb_models_GLC_w66_t95/complex_*_TCR-pMHC.pdb')
#all_structures = glob.glob('/scratch/s174564/TCRpMHCmodels_new/tcrpmhc_models/notGIL_peptides_StitchR/pdb_models_NLV/complex_*_TCR-pMHC.pdb')
all_structures = glob.glob('/scratch/s174564/TCRpMHCmodels_new/tcrpmhc_models/notGIL_peptides_StitchR/pdb_models_YLQPRTFLL_w66_t95/complex_*_TCR-pMHC.pdb')
no_files = len(all_structures)

print('Starting...')

#iterate over all structures get structure
for count, pdb_file in enumerate(all_structures):
    complex_name = re.split('_TCR-pMHC\.pdb', pdb_file.split('/')[-1])[0]
    print(complex_name, count+1, '/', no_files)
    #skip if files exist
    if os.path.isfile(neighbour_outpath + 'all_nearN_cdr3A_features/' + complex_name + '.tsv') and os.path.isfile(neighbour_outpath + 'all_nearN_cdr3B_features/' + complex_name + '.tsv'):
        print('File exists.')
        continue

    # get neighbours 
    #TCR_num_neighbors, TCR_inter_res_idx = extract_TCR_neighbours(pdb_file)
    _, TCR_inter_res_idx = extract_TCR_neighbours(pdb_file)

    # get true TCR indexes
    parser = PDBParser()
    structure = parser.get_structure("example", pdb_file)
    true_indexes = {"A": index_conversion(structure[0]["A"]), "B": index_conversion(structure[0]["B"])}

    for chain, indexes in TCR_inter_res_idx.items():
        nn_filepath = neighbour_outpath + 'all_nearN_cdr3' + str(chain) + '_features/' + complex_name + '.tsv'
        
        #skip if files exist
        if os.path.isfile(nn_filepath):
            print('File exists.')
            continue
        
        #write tsv file with neighbour information 
        with open(nn_filepath, 'a+') as neighbour_file: 
            #print(chain, indexes)
            logits, target_seq = get_logits(pdb_file, chain_ids, chain, esm_model, alphabet, "padding")
            # get idx of TCR neighbour
            for i, pos in enumerate(indexes):
                if pos != None:
                    for idx in pos:
                        #print(idx)
                        #print(target_seq[true_indexes[chain][idx]])
                        AA_logit = logits[true_indexes[chain][idx[0]]]
                        pos_prob = get_single_prob(AA_logit)
                        letter = AA_conversion[idx[1]]
                        neighbour_file.write('pos ' + str(i+1) + '\t' + '\t'.join([str(x.item()) for x in list(pos_prob)]) + '\t' + letter + '\n')
