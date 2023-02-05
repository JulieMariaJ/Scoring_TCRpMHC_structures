import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import glob 
from sklearn import metrics
import scipy.stats as stats
from Bio import SeqIO

### FUNCTIONS ###

def avg_df_from_tsvs(complex_names_list, tsv_path):
    """average probability across all complexes"""
    avg_df = pd.DataFrame(columns=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
       'R', 'S', 'T', 'V', 'W', 'Y'], index=[1,2,3,4,5,6,7,8,9])
    
    #iterate over each complex 
    for comp in complex_names_list:
        comp_tsv = pd.read_csv(tsv_path + comp + '.tsv', sep='\t', index_col=0)
        avg_df = avg_df.add(comp_tsv, fill_value=0)

    avg_df = avg_df.div(len(complex_names_list))
    return avg_df

def combine_avg_probs_df(dataframes):
    """average probability across list of dataframes"""
    combined_avg_df = dataframes[0]
    #iterate over each df
    for i in range(len(dataframes)-1):
        combined_avg_df = combined_avg_df.add(dataframes[i+1], fill_value=0)

    combined_avg_df = combined_avg_df.div(len(dataframes))
    return combined_avg_df

def avg_cdr3_probs_df(complex_names, tsv_path):
    """average probability across whole peptide type of binders"""
    all_AA_letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'letter']
    avg_df = pd.DataFrame(data=0, index=all_AA_letters[:-1], columns=['avg_score'])
    
    #iterate over each complex 
    for comp in complex_names:
        comp_tsv = pd.read_csv(tsv_path + comp + '.tsv', sep='\t', header=None, names=all_AA_letters)
        if comp_tsv.empty:
            continue
        cdr3_avg_df = pd.DataFrame(comp_tsv.drop(['letter'], axis=1).mean(axis=0), columns=['avg_score'])
        avg_df = avg_df.add(cdr3_avg_df, axis='index')
    
    avg_df = avg_df.div(len(complex_names))
    return avg_df


##### PATHS 
# out paths 
bg_path = '/scratch/s174595/master_project/data/processed/bg_files/' 
cdr3_bg_path = '/scratch/s174595/master_project/data/processed/bg_files/cdr3/'

# GIL path 
GIL_df = pd.read_csv('/scratch/s174595/master_project/data/processed/features_w66_t95/features_padding.tsv', sep='\t', index_col=0)
GIL_tsv_path = '/scratch/s174595/master_project/data/processed/features_w66_t95/tsv_files_padding/'
GIL_binder_names = list(GIL_df[GIL_df['binder'] == 1].index)
GIL_nonbinder_names = list(GIL_df[GIL_df['binder'] == 0].index)
# GLC path 
GLC_df = pd.read_csv('/scratch/s174564/master_project/data/processed/features_GLC_w66_t95/features_padding.tsv', sep='\t', index_col=0)
GLC_tsv_path = '/scratch/s174564/master_project/data/processed/features_GLC_w66_t95/tsv_files_padding/'
GLC_binder_names = list(GLC_df[GLC_df['binder'] == 1].index)
GLC_nonbinder_names = list(GLC_df[GLC_df['binder'] == 0].index)
# YLQ path 
YLQ_df = pd.read_csv('/scratch/s174595/master_project/data/processed/features_YLQPRTFLL_w66_t95/features_padding.tsv', sep='\t', index_col=0)
YLQ_tsv_path = '/scratch/s174595/master_project/data/processed/features_YLQPRTFLL_w66_t95/tsv_files_padding/'
YLQ_binder_names = list(YLQ_df[YLQ_df['binder'] == 1].index)
YLQ_nonbinder_names = list(YLQ_df[YLQ_df['binder'] == 0].index)
# NLV path 
NLV_df = pd.read_csv('/scratch/s174595/master_project/data/processed/features_NLV_w66_t95/features_padding.tsv', sep='\t', index_col=0)
NLV_tsv_path = '/scratch/s174595/master_project/data/processed/features_NLV_w66_t95/tsv_files_padding/'
NLV_binder_names = list(NLV_df[NLV_df['binder'] == 1].index)
NLV_nonbinder_names = list(NLV_df[NLV_df['binder'] == 0].index)
# WT SW path 
wt_fasta_fp = '/scratch/s174595/TCRpMHCmodels_new/tcrpmhc_models/CleanPDBfiles/data/cleaned_data/fasta/'
WT_df = pd.read_csv('/scratch/s174564/master_project/data/processed/features_WT_SW/features_WT/features_padding.tsv', sep='\t', index_col=0)
WT_tsv_path = '/scratch/s174564/master_project/data/processed/features_WT_SW/features_WT/tsv_files_padding/'
WT_names = list(WT_df.index)
WT_names = [x for x in WT_names if x not in ['2p5e', '3d39', '3d3v', '3o4l', '5jhd']]
sw_fasta_fp = '/scratch/s174564/TCRpMHCmodels_new/tcrpmhc_models/CleanPDBfiles/data/cleaned_data/fasta_swapped/'
SW_df = pd.read_csv('/scratch/s174564/master_project/data/processed/features_WT_SW/features_SW/features_padding.tsv', sep='\t', index_col=0)
SW_tsv_path = '/scratch/s174564/master_project/data/processed/features_WT_SW/features_SW/tsv_files_padding/'
SW_names = list(SW_df.index)
SW_names = [x for x in SW_names if x not in ['2p5e_SW', '3d39_SW', '3d3v_SW', '3o4l_SW', '5jhd_SW']]
# solved path 
solved_tsv_path = '/scratch/s174595/master_project/data/interim/solved_structures/padding/'

## CDR3 paths
#GIL
GIL_cdr3A_tsv_path = '/scratch/s174595/master_project/data/processed/features_w66_t95/all_nearN_cdr3A_features/'
GIL_cdr3B_tsv_path = '/scratch/s174595/master_project/data/processed/features_w66_t95/all_nearN_cdr3B_features/'
#GLC
GLC_cdr3A_tsv_path = '/scratch/s174595/master_project/data/processed/features_GLC_w66_t95/all_nearN_cdr3A_features/'
GLC_cdr3B_tsv_path = '/scratch/s174595/master_project/data/processed/features_GLC_w66_t95/all_nearN_cdr3B_features/'
#YLQ
YLQ_cdr3A_tsv_path = '/scratch/s174595/master_project/data/processed/features_YLQPRTFLL_w66_t95/all_nearN_cdr3A_features/'
YLQ_cdr3B_tsv_path = '/scratch/s174595/master_project/data/processed/features_YLQPRTFLL_w66_t95/all_nearN_cdr3B_features/'
#NLV
NLV_cdr3A_tsv_path = '/scratch/s174595/master_project/data/processed/features_NLV_w66_t95/all_nearN_cdr3A_features/'
NLV_cdr3B_tsv_path = '/scratch/s174595/master_project/data/processed/features_NLV_w66_t95/all_nearN_cdr3B_features/'


############################ make bg files #######################################

### GGYN ###

# GIL avg 
GIL_avg_df = avg_df_from_tsvs(GIL_binder_names, GIL_tsv_path)
GIL_nonbind_avg_df = avg_df_from_tsvs(GIL_nonbinder_names, GIL_tsv_path)
# GLC avg 
GLC_avg_df = avg_df_from_tsvs(GLC_binder_names, GLC_tsv_path)
GLC_nonbind_avg_df = avg_df_from_tsvs(GLC_nonbinder_names, GLC_tsv_path)
# YLQ avg 
YLQ_avg_df = avg_df_from_tsvs(YLQ_binder_names, YLQ_tsv_path)
YLQ_nonbind_avg_df = avg_df_from_tsvs(YLQ_nonbinder_names, YLQ_tsv_path)
# NLV avg 
NLV_avg_df = avg_df_from_tsvs(NLV_binder_names, NLV_tsv_path)
NLV_nonbind_avg_df = avg_df_from_tsvs(NLV_nonbinder_names, NLV_tsv_path)
# GGYN bg 
GGYN_avg_df = combine_avg_probs_df([GIL_avg_df, GIL_nonbind_avg_df, 
                                GLC_avg_df, GLC_nonbind_avg_df,
                                YLQ_avg_df, YLQ_nonbind_avg_df,
                                NLV_avg_df, NLV_nonbind_avg_df])

#write to outdir 
GIL_avg_df.to_csv(bg_path + 'GIL_avg_df.tsv', sep='\t')
GLC_avg_df.to_csv(bg_path + 'GLC_avg_df.tsv', sep='\t')
YLQ_avg_df.to_csv(bg_path + 'YLQ_avg_df.tsv', sep='\t')
NLV_avg_df.to_csv(bg_path + 'NLV_avg_df.tsv', sep='\t')
GGYN_avg_df.to_csv(bg_path + 'GGYN_avg_df.tsv', sep='\t')
GIL_nonbind_avg_df.to_csv(bg_path + 'GIL_nonbind_avg_df.tsv', sep='\t')
GLC_nonbind_avg_df.to_csv(bg_path + 'GLC_nonbind_avg_df.tsv', sep='\t')
YLQ_nonbind_avg_df.to_csv(bg_path + 'YLQ_nonbind_avg_df.tsv', sep='\t')
NLV_nonbind_avg_df.to_csv(bg_path + 'NLV_nonbind_avg_df.tsv', sep='\t')

### solved and WT-SW ###

# get sequences of WT and SW from fastas
wt_seqs = dict()
sw_seqs = dict()
for wt_file in glob.glob(wt_fasta_fp + '*.fasta'): 
    pdb_name = wt_file.split('/')[-1][:4]
    
    wt_fasta = wt_fasta_fp + pdb_name + '.fasta'
    sw_fasta = sw_fasta_fp + pdb_name + '_SW.fasta'
    for chain in SeqIO.parse(wt_fasta, "fasta"):
        if chain.id[-1] == 'P':
            orig_peptide = str(chain.seq)
            wt_seqs[pdb_name] = orig_peptide
    for chain in SeqIO.parse(sw_fasta, "fasta"):
        if chain.id[-1] == 'P':
            swap_peptide = str(chain.seq)
            sw_seqs[pdb_name + '_SW'] = swap_peptide

# WT SW avg
WT_avg_df = avg_df_from_tsvs(WT_names, WT_tsv_path)
SW_avg_df = avg_df_from_tsvs(SW_names, SW_tsv_path)
# solved avg
solved_avg_df = avg_df_from_tsvs(WT_names, solved_tsv_path)

#write to outdir 
WT_avg_df.to_csv(bg_path + 'WT_avg_df.tsv', sep='\t')
SW_avg_df.to_csv(bg_path + 'SW_avg_df.tsv', sep='\t')
solved_avg_df.to_csv(bg_path + 'solved_avg_df.tsv', sep='\t')

# modeled combined 

# avg of all binders / non-binders across peptide modeled 
all_pos_avg_df = combine_avg_probs_df([GIL_avg_df, NLV_avg_df, YLQ_avg_df, GLC_avg_df, WT_avg_df])
all_neg_avg_df = combine_avg_probs_df([GIL_nonbind_avg_df, NLV_nonbind_avg_df, YLQ_nonbind_avg_df, GLC_nonbind_avg_df, SW_avg_df])
# all combined 
total_avg_df = combine_avg_probs_df([all_pos_avg_df, all_neg_avg_df])


#write to outdir
all_pos_avg_df.to_csv(bg_path + 'pos_modeled_avg_df.tsv', sep='\t')
all_neg_avg_df.to_csv(bg_path + 'neg_modeled_avg_df.tsv', sep='\t')
total_avg_df.to_csv(bg_path + 'all_modeled_avg_df.tsv', sep='\t')

###########################################################################

############################## make CDR3 bgs #####################################

##  CDR3 backgrounds ##
# GIL
GIL_bind_avg_cdr3A_df = avg_cdr3_probs_df(GIL_binder_names, GIL_cdr3A_tsv_path)
GIL_nonbind_avg_cdr3A_df = avg_cdr3_probs_df(GIL_nonbinder_names, GIL_cdr3A_tsv_path)
GIL_bind_avg_cdr3B_df = avg_cdr3_probs_df(GIL_binder_names, GIL_cdr3B_tsv_path)
GIL_nonbind_avg_cdr3B_df = avg_cdr3_probs_df(GIL_nonbinder_names, GIL_cdr3B_tsv_path)
# GLC
GLC_bind_avg_cdr3A_df = avg_cdr3_probs_df(GLC_binder_names, GLC_cdr3A_tsv_path)
GLC_nonbind_avg_cdr3A_df = avg_cdr3_probs_df(GLC_nonbinder_names, GLC_cdr3A_tsv_path)
GLC_bind_avg_cdr3B_df = avg_cdr3_probs_df(GLC_binder_names, GLC_cdr3B_tsv_path)
GLC_nonbind_avg_cdr3B_df = avg_cdr3_probs_df(GLC_nonbinder_names, GLC_cdr3B_tsv_path)
# YLQ
YLQ_bind_avg_cdr3A_df = avg_cdr3_probs_df(YLQ_binder_names, YLQ_cdr3A_tsv_path)
YLQ_nonbind_avg_cdr3A_df = avg_cdr3_probs_df(YLQ_nonbinder_names, YLQ_cdr3A_tsv_path)
YLQ_bind_avg_cdr3B_df = avg_cdr3_probs_df(YLQ_binder_names, YLQ_cdr3B_tsv_path)
YLQ_nonbind_avg_cdr3B_df = avg_cdr3_probs_df(YLQ_nonbinder_names, YLQ_cdr3B_tsv_path)
# NLV
NLV_bind_avg_cdr3A_df = avg_cdr3_probs_df(NLV_binder_names, NLV_cdr3A_tsv_path)
NLV_nonbind_avg_cdr3A_df = avg_cdr3_probs_df(NLV_nonbinder_names, NLV_cdr3A_tsv_path)
NLV_bind_avg_cdr3B_df = avg_cdr3_probs_df(NLV_binder_names, NLV_cdr3B_tsv_path)
NLV_nonbind_avg_cdr3B_df = avg_cdr3_probs_df(NLV_nonbinder_names, NLV_cdr3B_tsv_path)

## write to tsvs
#GIL
GIL_bind_avg_cdr3A_df.to_csv(cdr3_bg_path + 'GIL_bind_cdr3A_avg_df.tsv', sep='\t')
GIL_nonbind_avg_cdr3A_df.to_csv(cdr3_bg_path + 'GIL_nonbind_cdr3A_avg_df.tsv', sep='\t')
GIL_bind_avg_cdr3B_df.to_csv(cdr3_bg_path + 'GIL_bind_cdr3B_avg_df.tsv', sep='\t')
GIL_nonbind_avg_cdr3B_df.to_csv(cdr3_bg_path + 'GIL_nonbind_cdr3B_avg_df.tsv', sep='\t')
#GLC
GLC_bind_avg_cdr3A_df.to_csv(cdr3_bg_path + 'GLC_bind_cdr3A_avg_df.tsv', sep='\t')
GLC_nonbind_avg_cdr3A_df.to_csv(cdr3_bg_path + 'GLC_nonbind_cdr3A_avg_df.tsv', sep='\t')
GLC_bind_avg_cdr3B_df.to_csv(cdr3_bg_path + 'GLC_bind_cdr3B_avg_df.tsv', sep='\t')
GLC_nonbind_avg_cdr3B_df.to_csv(cdr3_bg_path + 'GLC_nonbind_cdr3B_avg_df.tsv', sep='\t')
#YLQ
YLQ_bind_avg_cdr3A_df.to_csv(cdr3_bg_path + 'YLQ_bind_cdr3A_avg_df.tsv', sep='\t')
YLQ_nonbind_avg_cdr3A_df.to_csv(cdr3_bg_path + 'YLQ_nonbind_cdr3A_avg_df.tsv', sep='\t')
YLQ_bind_avg_cdr3B_df.to_csv(cdr3_bg_path + 'YLQ_bind_cdr3B_avg_df.tsv', sep='\t')
YLQ_nonbind_avg_cdr3B_df.to_csv(cdr3_bg_path + 'YLQ_nonbind_cdr3B_avg_df.tsv', sep='\t')
#NLV
NLV_bind_avg_cdr3A_df.to_csv(cdr3_bg_path + 'NLV_bind_cdr3A_avg_df.tsv', sep='\t')
NLV_nonbind_avg_cdr3A_df.to_csv(cdr3_bg_path + 'NLV_nonbind_cdr3A_avg_df.tsv', sep='\t')
NLV_bind_avg_cdr3B_df.to_csv(cdr3_bg_path + 'NLV_bind_cdr3B_avg_df.tsv', sep='\t')
NLV_nonbind_avg_cdr3B_df.to_csv(cdr3_bg_path + 'NLV_nonbind_cdr3B_avg_df.tsv', sep='\t')