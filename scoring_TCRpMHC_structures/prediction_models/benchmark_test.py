## PACKAGES ##
import re
import glob
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from pathlib import Path
from sklearn import metrics
import scipy.stats as stats
import matplotlib.pyplot as plt
from dotenv import find_dotenv
from statannot import add_stat_annotation
from matplotlib import collections  as mc


## PATHS ##

dotenv_path = find_dotenv()
ROOTPATH = Path(find_dotenv()).parent
RAWPATH = Path(ROOTPATH, 'data/raw')
ESMPATH = Path(ROOTPATH, 'data/ESM-IF1_predictions')
RESULTSPATH = Path(ROOTPATH, 'result_figures') 

## FUNCTIONS ##

def combine_avg_probs_df(dataframes):
    """combine multiple average dataframes to one average df"""

    combined_avg_df = dataframes[0]
    for i in range(len(dataframes)-1):
        combined_avg_df = combined_avg_df.add(dataframes[i+1], fill_value=0)

    combined_avg_df = combined_avg_df.div(len(dataframes))

    return combined_avg_df


def compute_logodds(orig_probs_df, background_df):
    """compute log2(odds) df from orig probs and bg probs"""

    odds_df = orig_probs_df/background_df
    logodds_df = np.log2(odds_df.astype(float))

    return logodds_df


def add_peptide_score_bg(fastafiles, input_df, probs_path, background_df, which_score='WT_score'):

    features_df = (input_df).copy()
    features_df[['peptide', which_score]] = ''

    for fasta in fastafiles:
        id_name = fasta.split("/")[-1] 
        id_name = re.sub(r"(.fasta)$", "", id_name)

        if id_name in features_df.index:
            for record in SeqIO.parse(fasta, "fasta"):
                if record.id[-1] == 'P':
                    # add target sequence
                    target_seq = str(record.seq)
                    features_df['peptide'][id_name] = target_seq
                   
                    prob_file = probs_path + id_name + '.tsv'
                    all_pos_probs_df = pd.read_csv(prob_file, sep='\t', index_col=0)
                    
                    # get sum log2(odds) score
                    logodds_df = compute_logodds(all_pos_probs_df, background_df)
                    sum_log_odds = 0
                    for i, letter in enumerate(target_seq):
                        sum_log_odds += logodds_df[letter][i+1]
                    features_df[which_score][id_name] = sum_log_odds

    return features_df

def plot_scores_boxplot(scores_df, plot_title='', filename='', plot_lines=False):

    fig, ax = plt.subplots()
    sns.set_style("whitegrid")
    sns.set_palette("Paired")

    # plot boxplot
    p_value_stud = stats.ttest_rel(scores_df["WT"], scores_df["SW"]).pvalue
    sns.boxplot(data = scores_df)

    ## AUC for WT-SW
    y_preds = list(scores_df['WT'].values) + list(scores_df['SW'].values)
    y_true_binary = [1]*len(scores_df) + [0]*len(scores_df)
    auc = metrics.roc_auc_score(y_true_binary, y_preds)

    # plot lines for pairwise WT-SW    
    lines = []
    delta_values = []
    for i, wt_dot in enumerate(scores_df["WT"]):
        lines.append([(0, wt_dot), (1, scores_df["SW"][i])])
        delta_values.append(wt_dot - scores_df["SW"][i])
    lc_pos = mc.LineCollection([x for (j, x) in enumerate(lines) if delta_values[j] >= 0], colors='blue', linewidths=1.5, alpha=0.5)
    lc_neg = mc.LineCollection([x for (j, x) in enumerate(lines) if delta_values[j] < 0], colors='red', linewidths=1.5, alpha=0.5)    
    
    if plot_lines == True:
        ax = sns.stripplot(data = scores_df, color="black", jitter=False)
        ax.add_collection(lc_pos)
        ax.add_collection(lc_neg)

    plt.ylabel("Peptide score")
    plt.ylim([-24,14])
    plt.suptitle(plot_title, fontsize=14)
    plt.title(f"Delta avg.: {sum(delta_values)/len(delta_values):.2f}, p-value: {p_value_stud:.3f}, AUC: {auc:.3f}")
    
    fig.savefig(filename, format="pdf", dpi=1200)


def plot_corr_scatter(score_df, xname='', yname='', plot_title='', xlab='', ylab='', filename=''):
    pdb_ids_sorted = sorted(list(score_df.index))
    sns.set_palette('colorblind')

    fig, ax = plt.subplots()
    pearson_cor, pval = stats.pearsonr(score_df[xname], score_df[yname])

    ax.scatter(score_df[xname], score_df[yname], s=12, label=f"PCC: {pearson_cor:.3f}, p-value: {pval:.3f}")
    for i, name in enumerate(pdb_ids_sorted): 
        ax.annotate(name, (score_df[xname][i], score_df[yname][i]))

    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(plot_title)
    plt.legend(loc="upper left")
    plt.savefig(filename, format="pdf", dpi=1200)


################################## Collect scores ##################################

### Load average dfs for GGYN
total_background_df = pd.read_csv(Path(ESMPATH, 'background_models/GGYN_avg_df.tsv'), sep='\t', index_col=0)

### Collect scores from WT-SW data
WT_features_file = Path(ESMPATH, 'peptide_features/WT/features_padding.tsv')
WT_features_df = pd.read_csv(WT_features_file, sep='\t', index_col=0)
WT_fasta_files = glob.glob(str(Path(RAWPATH, 'fastafiles/solved_WT'))  + '/*.fasta')
WT_probs_path = str(ESMPATH) + "/peptide_features/WT/tsv_files_padding/"

SW_features_file = Path(ESMPATH, 'peptide_features/SW/features_padding.tsv')
SW_features_df = pd.read_csv(SW_features_file, sep='\t', index_col=0)
SW_fasta_files = glob.glob(str(Path(RAWPATH, 'fastafiles/SW')) + '/*_SW.fasta')
SW_probs_path = str(ESMPATH) + "/peptide_features/SW/tsv_files_padding/"

WT_features_df = add_peptide_score_bg(WT_fasta_files, WT_features_df, WT_probs_path, total_background_df, which_score='WT_score')
SW_features_df = add_peptide_score_bg(SW_fasta_files, SW_features_df, SW_probs_path, total_background_df, which_score='SW_score')

### dataframe for GGYN background ###
WTSW_score_df = pd.DataFrame()
WTSW_score_df['WT'] = WT_features_df['WT_score'].copy()
WTSW_score_df['SW'] = ''
for idx, score in SW_features_df.iterrows():
    WTSW_score_df['SW'][idx[:4]] = SW_features_df['SW_score'][idx].copy()

# delta score
WTSW_score_df['delta_score'] = ''
for idx, row in WTSW_score_df.iterrows():
    WTSW_score_df['delta_score'][idx] = row['WT'] - row['SW']

# delta identity score
WTSW_score_df[['delta_tot_iden', 'delta_mhc_iden', 'delta_pep_iden', 'delta_tcrA_iden', 'delta_tcrB_iden']] = ''
for idx, row in WTSW_score_df.iterrows():
    WTSW_score_df['delta_tot_iden'][idx] = WT_features_df['total_iden_avg'][idx] - SW_features_df['total_iden_avg'][idx + '_SW']
    WTSW_score_df['delta_mhc_iden'][idx] = WT_features_df['mhc_iden_avg'][idx] - SW_features_df['mhc_iden_avg'][idx + '_SW']
    WTSW_score_df['delta_pep_iden'][idx] = WT_features_df['pep_iden_avg'][idx] - SW_features_df['pep_iden_avg'][idx + '_SW']
    WTSW_score_df['delta_tcrA_iden'][idx] = WT_features_df['tcrA_iden_avg'][idx] - SW_features_df['tcrA_iden_avg'][idx + '_SW']
    WTSW_score_df['delta_tcrB_iden'][idx] = WT_features_df['tcrB_iden_avg'][idx] - SW_features_df['tcrB_iden_avg'][idx + '_SW']

# drop special cases
specialcases = ['3d39','3d3v']
WTSW_score_df = WTSW_score_df.drop(specialcases, axis=0)

score_uniform_bg = pd.DataFrame(index=WTSW_score_df.index)
score_uniform_bg['WT'] = ''
score_uniform_bg['SW'] = ''

### dataframe using uniform background ###
specialcases = ['3d39','3d3v']
for i, row in WT_features_df.iterrows():
    if i not in specialcases:
        score_uniform_bg['WT'][i] = np.log2(np.prod(row[['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']]/0.05))

for i, row in SW_features_df.iterrows():
    if i[:4] not in specialcases:
        score_uniform_bg['SW'][i[:4]] = np.log2(np.prod(row[['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']]/0.05))

# re_uniform_bg.index)
# specialcases = ['3d39','3d3v']
# score_uniform_bg = score_uniform_bg.drop(specialcases, axis=0)

####################### Get top identities and RMSDs ###############################

### Collect pymol RMSDs
pymol_rmsds = pd.read_csv(Path(RAWPATH, 'pymol_rmsd_solved_model_wt.csv'), index_col=0)

### Collect top template identities
temp_iden_csv = glob.glob(str(Path(RAWPATH, 'template_identities/WT')) + '/*complex-templates.csv')

## get top templates
top_templates = {}
specialcases = ['3d39','3d3v']
for file in temp_iden_csv:
    id_name = file.split("/")[-1].split("-")[0] 
    if id_name not in specialcases:
        dataframe = pd.read_csv(file, header=0)
        # add only top template
        top_templates[id_name] = dataframe["total_identity"][0]

# collect data
top_templates = pd.DataFrame.from_dict(top_templates, orient='index', columns=['top_total_identity'])
chothia_lesk_df = top_templates.join(pymol_rmsds)


################################## Visualizations ##################################

### Boxplots ###
sns.reset_defaults()
sns.set_style("whitegrid")

## Plot 1: WT and SW scores (BG: uniform)
plot_scores_boxplot(score_uniform_bg, 
                    plot_title='WT and SW scores using uniform background',
                    filename=Path(RESULTSPATH, 'WTSW_scores_uniformbg.pdf'),
                    plot_lines=False)

## Plot 2: WT and SW scores (BG: GGYN)
plot_scores_boxplot(WTSW_score_df.drop(['delta_score', 'delta_mhc_iden', 'delta_tot_iden', 'delta_pep_iden', 'delta_tcrA_iden', 'delta_tcrB_iden'], axis=1), 
                    plot_title='WT and SW scores using GGYN background',
                    filename=Path(RESULTSPATH, 'WTSW_scores_GGYNbg.pdf'),
                    plot_lines=False)

# with lines
plot_scores_boxplot(score_uniform_bg, 
                    plot_title='WT and SW scores using uniform background',
                    filename=Path(RESULTSPATH, 'WTSW_scores_uniformbg_lines.pdf'),
                    plot_lines=True)
plot_scores_boxplot(WTSW_score_df.drop(['delta_score', 'delta_mhc_iden', 'delta_tot_iden', 'delta_pep_iden', 'delta_tcrA_iden', 'delta_tcrB_iden'], axis=1), 
                    plot_title='WT and SW scores using GGYN background',
                    filename=Path(RESULTSPATH, 'WTSW_scores_GGYNbg_lines.pdf'),
                    plot_lines=True)

### Supplementary plots ###

## Plot 3: Correlation bt. total identity and score
plot_corr_scatter(WTSW_score_df, 
                    xname='delta_tot_iden', 
                    yname='delta_score', 
                    plot_title='Total sequence identity', 
                    xlab='∆ Total identity', 
                    ylab='∆ Peptide score',
                    filename=Path(RESULTSPATH, 'deltascore_tot_iden_corr.pdf'))

## Plot 3: Correlation bt. TCRbeta identity and score
plot_corr_scatter(WTSW_score_df, 
                    xname='delta_tcrB_iden', 
                    yname='delta_score', 
                    plot_title='TCRβ identity', 
                    xlab='∆ TCRβ identity', 
                    ylab='∆ Peptide score',
                    filename=Path(RESULTSPATH, 'deltascore_tcrB_iden_corr.pdf'))


## Plot 4: Correlation of orig values instead of delta (WT)
pdb_ids_sorted = sorted(list(WTSW_score_df.index))
sns.set_palette('colorblind')

fig, axes = plt.subplots(1,2, figsize=(16,6))

# WT
data = WT_features_df.drop(specialcases, axis=0)
pearson_cor, pval = stats.pearsonr(data['tcrB_iden_avg'], data['WT_score'])
axes[0].scatter(data['tcrB_iden_avg'], data['WT_score'], s=12, label=f"PCC: {pearson_cor:.3f}, p-value: {pval:.3f}")
for i, name in enumerate(pdb_ids_sorted): 
    axes[0].annotate(name, (data['tcrB_iden_avg'][i], data['WT_score'][i]))
axes[0].set_ylabel('Peptide score')
axes[0].set_xlabel('TCRβ identity')
axes[0].set_title(f'Binders (WT)')
axes[0].legend(loc="upper left")

# SW
specialcases_SW = ['3d39_SW','3d3v_SW']
data = SW_features_df.drop(specialcases_SW, axis=0)
pearson_cor, pval = stats.pearsonr(data['tcrB_iden_avg'], data['SW_score'])
axes[1].scatter(data['tcrB_iden_avg'], data['SW_score'], s=12,  label=f"PCC: {pearson_cor:.3f}, p-value: {pval:.3f}")
for i, name in enumerate(pdb_ids_sorted): 
    axes[1].annotate(name, (data['tcrB_iden_avg'][i], data['SW_score'][i]))
axes[1].set_ylabel('Peptide score')
axes[1].set_xlabel('TCRβ identity')
axes[1].set_title(f'Non-binders (SW)')
axes[1].legend(loc="upper right")

plt.savefig(Path(RESULTSPATH, 'score_tcrB_iden_corr.pdf'), format="pdf", dpi=1200)


## Plot 5: Correlation plots for remaining chain identities
pdb_ids_sorted = sorted(list(WTSW_score_df.index))
sns.set_palette('colorblind')
sns.set_style("whitegrid")

fig, axes = plt.subplots(2,2, figsize=(18,12))
identities = ['delta_mhc_iden', 'delta_pep_iden', 'delta_tcrA_iden', 'delta_tcrB_iden']
titles = ['MHC identity', 'Peptide identity', 'TCRα identity', 'TCRβ identity']

# plot all four
count = 0
for i in range(2):
    for j in range(2):
        pearson_cor, pval = stats.pearsonr(WTSW_score_df[identities[count]], WTSW_score_df['delta_score'])
        axes[i,j].scatter(WTSW_score_df[identities[count]], WTSW_score_df['delta_score'], s=12, label=f"PCC: {pearson_cor:.3f}, p-value: {pval:.3f}")
        for idx, name in enumerate(pdb_ids_sorted): 
            axes[i,j].annotate(name, (WTSW_score_df[identities[count]][idx], WTSW_score_df['delta_score'][idx]))
        
        axes[i,j].set_title(f"{titles[count]}")
        axes[i,j].set_xlabel(f"∆ {titles[count]}")
        axes[i,j].set_ylabel("∆ Peptide score")
        axes[i,j].set_ylim([-10, 30])
        axes[i,j].legend(loc="upper left")
        count += 1

plt.savefig(Path(RESULTSPATH, 'deltascore_chain_iden_corr.pdf'), format="pdf", dpi=1200)


### Plot 6: Chothia-Lesk plot
fig = plt.subplots(figsize=(8, 6))
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# TCRpMHC
pearson_cor, pval = stats.pearsonr(chothia_lesk_df['top_total_identity'], chothia_lesk_df['RMSD total'])
sns.scatterplot(x=chothia_lesk_df['top_total_identity'], y=chothia_lesk_df['RMSD total'], label=f"TCR-pMHC RMSD, PCC: {pearson_cor:.3f}, p-val: {pval:.3f}")

# pMHC
pearson_cor, pval = stats.pearsonr(chothia_lesk_df['top_total_identity'], chothia_lesk_df['RMSD pMHC'])
sns.scatterplot(x=chothia_lesk_df['top_total_identity'], y=chothia_lesk_df['RMSD pMHC'], label=f"pMHC RMSD, PCC = {pearson_cor:.3f}, p-val: {pval:.3f}")

# peptide
pearson_cor, pval = stats.pearsonr(chothia_lesk_df['top_total_identity'], chothia_lesk_df['RMSD peptide'])
sns.scatterplot(x=chothia_lesk_df['top_total_identity'], y=chothia_lesk_df['RMSD peptide'], label=f"peptide RMSD, PCC = {pearson_cor:.3f}, p-val: {pval:.3f}")

plt.title("RMSD(solved, model) vs. similarity to top template", fontsize=16)
plt.xlabel("Sequence similarity to top template (%)")
plt.ylabel(("RMSD"))
plt.legend()
plt.savefig(Path(RESULTSPATH, 'chothia_lesk.pdf'), format="pdf", dpi=1200)
