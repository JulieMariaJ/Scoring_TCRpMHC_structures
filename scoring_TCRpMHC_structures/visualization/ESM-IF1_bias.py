import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics
import scipy.stats as stats
from Bio import SeqIO
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

### init paths ###

dotenv_path = find_dotenv()
ROOTPATH = Path(find_dotenv()).parent
RAWPATH = Path(ROOTPATH, 'data/raw')
ESMPATH = Path(ROOTPATH, 'data/ESM-IF1_predictions')
RESULTSPATH = Path(ROOTPATH, 'result_figures') 

### filepaths ###
bg_path = Path(ESMPATH, 'background_models')
solved_tsv_path = Path(ESMPATH, 'peptide_features/solved')
modeled_tsv_path = Path(ESMPATH, 'peptide_features/WT/tsv_files_padding')
wt_seqs = pd.read_csv(Path(RAWPATH, 'fastafiles/WT_seqs.tsv'), sep='\t', index_col=0)

# all bg files 
solved_avg_df = pd.read_csv(Path(bg_path, 'solved_avg_df.tsv'), sep='\t', index_col=0)
WT_avg_df = pd.read_csv(Path(bg_path, 'WT_avg_df.tsv'), sep='\t', index_col=0)
SW_avg_df = pd.read_csv(Path(bg_path, 'SW_avg_df.tsv'), sep='\t', index_col=0)
GIL_avg_df = pd.read_csv(Path(bg_path, 'GIL_avg_df.tsv'), sep='\t', index_col=0)
GIL_nonbind_avg_df = pd.read_csv(Path(bg_path, 'GIL_nonbind_avg_df.tsv'), sep='\t', index_col=0)
GLC_avg_df = pd.read_csv(Path(bg_path, 'GLC_avg_df.tsv'), sep='\t', index_col=0)
GLC_nonbind_avg_df = pd.read_csv(Path(bg_path, 'GLC_nonbind_avg_df.tsv'), sep='\t', index_col=0)
YLQ_avg_df = pd.read_csv(Path(bg_path, 'YLQ_avg_df.tsv'), sep='\t', index_col=0)
YLQ_nonbind_avg_df = pd.read_csv(Path(bg_path, 'YLQ_nonbind_avg_df.tsv'), sep='\t', index_col=0)
NLV_avg_df = pd.read_csv(Path(bg_path, 'NLV_avg_df.tsv'), sep='\t', index_col=0)
NLV_nonbind_avg_df = pd.read_csv(Path(bg_path, 'NLV_nonbind_avg_df.tsv'), sep='\t', index_col=0)
all_pos_avg_df = pd.read_csv(Path(bg_path, 'pos_modeled_avg_df.tsv'), sep='\t', index_col=0)
all_neg_avg_df = pd.read_csv(Path(bg_path, 'neg_modeled_avg_df.tsv'), sep='\t', index_col=0)
total_avg_df = pd.read_csv(Path(bg_path, 'all_modeled_avg_df.tsv'), sep='\t', index_col=0)


########################## Bell plot ###################################
### requires: 
    # - tsvpath for solved ESM-IF1 predictions
    # - tsvpath for modeled ESM-IF1 predictions
    # - tsvpath containing sequences for solved or WT 
###

#initialize dfs with uniform bg
uni_bg_solved_df = pd.DataFrame(index=range(1,10))
uni_bg_modeled_df = pd.DataFrame(index=range(1,10))
q = 0.05 

#iterate over each complex
for complex_name, seq in wt_seqs.iterrows():
    #leave one out of background 
    solved_tsv = pd.read_csv(Path(solved_tsv_path, complex_name + '.tsv'), sep='\t', index_col=0)
    modeled_tsv = pd.read_csv(Path(modeled_tsv_path, complex_name + '.tsv'), sep='\t', index_col=0)

    #log odds ratio of correct AA 
    solved_scores = []
    modeled_scores = []
    for i, AA in enumerate(seq.values[0]):
        #solved
        solved_scores.append(np.log2(solved_tsv[AA][i+1]/q)) 
        #modeled
        modeled_scores.append(np.log2(modeled_tsv[AA][i+1]/q)) 

    #append scores to df 
    uni_bg_solved_df[complex_name] = solved_scores
    uni_bg_modeled_df[complex_name] = modeled_scores

#plot 
all_dfs = {'no_bg_solved': uni_bg_solved_df.T, 'no_bg_modeled': uni_bg_modeled_df.T} 
all_names = ['Solved', 'Modeled']

fig, axes = plt.subplots(1,2, figsize=(15,5)) 
sns.set_palette("pastel")
sns.set_style("whitegrid")

for i, (ax, (k, d)) in enumerate(zip(axes.ravel(), all_dfs.items())):
    d = pd.melt(d.reset_index(), id_vars = 'index', value_vars=[1,2,3,4,5,6,7,8,9], var_name='position', value_name='Score')
    
    ax.hlines(y=[0,4.3], xmin=-10, xmax=20, colors='blue', linestyles='--', lw=1, zorder=-1) #bring back = zorder=-1
    #ax.hlines(y=0, xmin=-1, xmax=10, colors='gray', linestyles='--', lw=2)
    
    sns.boxplot(data=d, 
            x="position", 
            y="Score",
            showmeans=True,
            ax=ax) 

    ax.set_ylim([-8,5])
    ax.set_title(all_names[i], fontsize=16)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Position', fontsize=14)
    
plt.savefig(Path(RESULTSPATH, 'bellplot.pdf'), format='pdf')


########################## Heatmap plot ###################################
### requires: 
    # - tsvpaths for average ESM-IF1 prediction for: 
                # Solved', 'WT', 'SW', 
                # 'GIL (+)', 'GIL (-)', 'GLC (+)', 'GLC (-)', 'YLQ (+)', 'YLQ (-)', 'NLV (+)', 'NLV (-)', 
                # 'All modeled (+)', 'All modeled (-)', 'All modeled'
###

#initiate heatmap
combination_names = ['Solved', 
                        'WT', 'SW', 
                        'GIL (+)', 'GIL (-)', 
                        'GLC (+)', 'GLC (-)', 
                        'YLQ (+)', 'YLQ (-)', 
                        'NLV (+)', 'NLV (-)', 
                        'All modeled (+)', 'All modeled (-)', 
                        'All modeled']
heatmap_df = pd.DataFrame(columns=combination_names, 
            index=combination_names)
all_dfs = [solved_avg_df, 
            WT_avg_df, SW_avg_df, 
            GIL_avg_df, GIL_nonbind_avg_df, 
            GLC_avg_df, GLC_nonbind_avg_df, 
            YLQ_avg_df, YLQ_nonbind_avg_df, 
            NLV_avg_df, NLV_nonbind_avg_df, 
            all_pos_avg_df, all_neg_avg_df, 
            total_avg_df]

# generate heatmap df of pccs
for i in range(len(heatmap_df)):
    col_name = combination_names[i]
    col_df = all_dfs[i]
    for j in range(len(heatmap_df)):
        row_name = combination_names[j]
        row_df = all_dfs[j]
        #pearson's corr coef
        pcc = stats.pearsonr(col_df.values.flatten(), row_df.values.flatten()).statistic
        #add to df 
        heatmap_df[col_name][row_name] = float(pcc)


#plot heatmap 
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(heatmap_df.astype(float), annot=True, vmin=0.60, vmax=1.0, cmap='flare') 
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.tick_params(axis='x', rotation=90)
plt.savefig(Path(RESULTSPATH, 'heatmap.pdf'), format='pdf')

###########################################################################