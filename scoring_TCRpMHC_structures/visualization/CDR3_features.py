## PACKAGES ##

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import scipy.stats as stats
from sklearn import metrics
import matplotlib.pyplot as plt 
from statannot import add_stat_annotation
from dotenv import find_dotenv, load_dotenv

## PATHS ##

dotenv_path = find_dotenv()
ROOTPATH = Path(find_dotenv()).parent
RAWPATH = Path(ROOTPATH, 'data/raw')
ESMPATH = Path(ROOTPATH, 'data/ESM-IF1_predictions')
RESULTSPATH = Path(ROOTPATH, 'result_figures')

## FUNCTIONS ##

def combine_avg_probs_df(dataframes):

    combined_avg_df = dataframes[0]
    for i in range(len(dataframes)-1):
        combined_avg_df = combined_avg_df.add(dataframes[i+1], fill_value=0)

    combined_avg_df = combined_avg_df.div(len(dataframes))

    return combined_avg_df


def collect_cdr3_scores(complex_names, tsv_path, background_df):

    all_AA_letters = ['Pos', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'letter']
    cdr3_scores = {}

    for complex in complex_names:
        complex_df = pd.read_csv(tsv_path + complex + '.tsv', sep='\t', index_col=None, header=None, names=all_AA_letters)
        if complex_df.empty:
            cdr3_scores[complex] = np.nan

        else:
            score_sum = 0
            for pep_pos, row in complex_df.iterrows():
                pos = row['Pos'][-1]
                letter = row['letter']
                prob = complex_df[letter][pep_pos]
                log_odds = np.log2(prob/background_df['avg_score'][str(letter)])
                score_sum += log_odds

            # add score
            cdr3_scores[complex] = score_sum/len(complex_df)

    return cdr3_scores


def combine_scores(bind_scores_dict, nonbind_scores_dict, peptide=''):

    # drop nans
    bind_scores_dict = {k:v for (k,v) in bind_scores_dict.items() if not np.isnan(v)}
    nonbind_scores_dict = {k:v for (k,v) in nonbind_scores_dict.items() if not np.isnan(v)}

    # binders
    bind_score_df = pd.DataFrame.from_dict(data=bind_scores_dict, orient='index', columns=['score'])
    bind_score_df['binder'] = 1

    # non binders
    nonbind_score_df = pd.DataFrame.from_dict(data=nonbind_scores_dict, orient='index', columns=['score'])
    nonbind_score_df['binder'] = 0

    combined_scores_df = pd.concat([bind_score_df, nonbind_score_df])
    combined_scores_df['peptide'] = peptide
    
    return combined_scores_df


def plot_cdr3_histogram(pepnames, average_dfs, ylim=[0,1], plot_title='', filename=''):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.set_style("whitegrid")

    count = 0
    for i in range(2):
        for j in range(2):
            sns.barplot(x=average_dfs[count].index, 
                    y=average_dfs[count]['avg_score'],
                    ax=axes[i,j]
            )
            axes[i,j].set_ylim(ylim)
            axes[i,j].set_xlabel("")
            axes[i,j].set_ylabel("Avg. score")
            axes[i,j].set_title(f"{pepnames[count]}")
            count += 1

    plt.suptitle(plot_title, fontsize=18)
    plt.savefig(filename, format="pdf", dpi=1200)


def plot_cdr3_heatmap_corr(all_dfs, combination_names, plot_title='', filename=''):

    heatmap_df = pd.DataFrame(columns=combination_names, 
                                index=combination_names)

    # calculate PCCs
    for i in range(len(heatmap_df)):
        col_name = combination_names[i]
        col_df = all_dfs[i]
        for j in range(len(heatmap_df)):
            row_name = combination_names[j]
            row_df = all_dfs[j]
            pcc = stats.pearsonr(list(col_df.values.flatten()), list(row_df.values.flatten()))[0]
            heatmap_df[col_name][row_name] = float(pcc)

    # plot heatmap
    fig, ax = plt.subplots(figsize=(8,6))
    ax = sns.heatmap(heatmap_df.astype(float), annot=True, vmin=0.90, vmax=1.0, cmap='flare') 
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', rotation=90)

    plt.title(plot_title, fontsize=16)
    plt.savefig(filename, format="pdf", dpi=1200)
        

#################### Load average dataframes ######################

### Load average df for GGYN
total_background_df = pd.read_csv(Path(ESMPATH, 'background_models/GGYN_avg_df.tsv'), sep='\t', index_col=0)

# GIL 
GIL_bind_cdr3A_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/GIL_bind_cdr3A_avg_df.tsv'), sep='\t', index_col=0)
GIL_bind_cdr3B_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/GIL_bind_cdr3B_avg_df.tsv'), sep='\t', index_col=0)
GIL_nonbind_cdr3A_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/GIL_nonbind_cdr3A_avg_df.tsv'), sep='\t', index_col=0)
GIL_nonbind_cdr3B_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/GIL_nonbind_cdr3B_avg_df.tsv'), sep='\t', index_col=0)

# GLC
GLC_bind_cdr3A_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/GLC_bind_cdr3A_avg_df.tsv'), sep='\t', index_col=0)
GLC_bind_cdr3B_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/GLC_bind_cdr3B_avg_df.tsv'), sep='\t', index_col=0)
GLC_nonbind_cdr3A_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/GLC_nonbind_cdr3A_avg_df.tsv'), sep='\t', index_col=0)
GLC_nonbind_cdr3B_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/GLC_nonbind_cdr3B_avg_df.tsv'), sep='\t', index_col=0)

# YLQ
YLQ_bind_cdr3A_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/YLQ_bind_cdr3A_avg_df.tsv'), sep='\t', index_col=0)
YLQ_bind_cdr3B_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/YLQ_bind_cdr3B_avg_df.tsv'), sep='\t', index_col=0)
YLQ_nonbind_cdr3A_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/YLQ_nonbind_cdr3A_avg_df.tsv'), sep='\t', index_col=0)
YLQ_nonbind_cdr3B_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/YLQ_nonbind_cdr3B_avg_df.tsv'), sep='\t', index_col=0)

# NLV
NLV_bind_cdr3A_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/NLV_bind_cdr3A_avg_df.tsv'), sep='\t', index_col=0)
NLV_bind_cdr3B_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/NLV_bind_cdr3B_avg_df.tsv'), sep='\t', index_col=0)
NLV_nonbind_cdr3A_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/NLV_nonbind_cdr3A_avg_df.tsv'), sep='\t', index_col=0)
NLV_nonbind_cdr3B_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/cdr3/NLV_nonbind_cdr3B_avg_df.tsv'), sep='\t', index_col=0)

#### Paths ####
GIL_df = pd.read_csv(Path(ESMPATH, 'peptide_features/GIL/features_padding.tsv'), sep='\t', index_col=0)
GIL_binder_names = list(GIL_df[GIL_df['binder'] == 1].index)
GIL_nonbinder_names = list(GIL_df[GIL_df['binder'] == 0].index)
GIL_cdr3A_tsv_path = str(Path(ESMPATH, 'CDR3_features/GIL/all_nearN_cdr3A_features')) + '/'
GIL_cdr3B_tsv_path = str(Path(ESMPATH, 'CDR3_features/GIL/all_nearN_cdr3B_features')) + '/'

GLC_df = pd.read_csv(Path(ESMPATH, 'peptide_features/GLC/features_padding.tsv'), sep='\t', index_col=0)
GLC_binder_names = list(GLC_df[GLC_df['binder'] == 1].index)
GLC_nonbinder_names = list(GLC_df[GLC_df['binder'] == 0].index)
GLC_cdr3A_tsv_path = str(Path(ESMPATH, 'CDR3_features/GLC/all_nearN_cdr3A_features')) + '/'
GLC_cdr3B_tsv_path = str(Path(ESMPATH, 'CDR3_features/GLC/all_nearN_cdr3B_features')) + '/'

YLQ_df = pd.read_csv(Path(ESMPATH, 'peptide_features/YLQ/features_padding.tsv'), sep='\t', index_col=0)
YLQ_binder_names = list(YLQ_df[YLQ_df['binder'] == 1].index)
YLQ_nonbinder_names = list(YLQ_df[YLQ_df['binder'] == 0].index)
YLQ_cdr3A_tsv_path = str(Path(ESMPATH, 'CDR3_features/YLQ/all_nearN_cdr3A_features')) + '/'
YLQ_cdr3B_tsv_path = str(Path(ESMPATH, 'CDR3_features/YLQ/all_nearN_cdr3B_features')) + '/'

NLV_df = pd.read_csv(Path(ESMPATH, 'peptide_features/NLV/features_padding.tsv'), sep='\t', index_col=0)
NLV_binder_names = list(NLV_df[NLV_df['binder'] == 1].index)
NLV_nonbinder_names = list(NLV_df[NLV_df['binder'] == 0].index)
NLV_cdr3A_tsv_path = str(Path(ESMPATH, 'CDR3_features/NLV/all_nearN_cdr3A_features')) + '/'
NLV_cdr3B_tsv_path = str(Path(ESMPATH, 'CDR3_features/NLV/all_nearN_cdr3B_features')) + '/'

#### Calculate scores using bg ####

## GIL alpha
GIL_background_df = combine_avg_probs_df([NLV_bind_cdr3A_avg_df, NLV_nonbind_cdr3A_avg_df, YLQ_bind_cdr3A_avg_df, YLQ_nonbind_cdr3A_avg_df, GLC_bind_cdr3A_avg_df, GLC_nonbind_cdr3A_avg_df])
cdr3A_bind_scores = collect_cdr3_scores(GIL_binder_names, GIL_cdr3A_tsv_path, GIL_background_df)
cdr3A_nonbind_scores = collect_cdr3_scores(GIL_nonbinder_names, GIL_cdr3A_tsv_path, GIL_background_df)
GIL_cdr3A_combined_scores_df = combine_scores(cdr3A_bind_scores, cdr3A_nonbind_scores, peptide='GIL')
## GIL beta
GIL_background_df = combine_avg_probs_df([NLV_bind_cdr3B_avg_df, NLV_nonbind_cdr3B_avg_df, YLQ_bind_cdr3B_avg_df, YLQ_nonbind_cdr3B_avg_df, GLC_bind_cdr3B_avg_df, GLC_nonbind_cdr3B_avg_df])
cdr3B_bind_scores = collect_cdr3_scores(GIL_binder_names, GIL_cdr3B_tsv_path, GIL_background_df)
cdr3B_nonbind_scores = collect_cdr3_scores(GIL_nonbinder_names, GIL_cdr3B_tsv_path, GIL_background_df)
GIL_cdr3B_combined_scores_df = combine_scores(cdr3B_bind_scores, cdr3B_nonbind_scores, peptide='GIL')

## GLC alpha
GLC_background_df = combine_avg_probs_df([NLV_bind_cdr3A_avg_df, NLV_nonbind_cdr3A_avg_df, YLQ_bind_cdr3A_avg_df, YLQ_nonbind_cdr3A_avg_df, GIL_bind_cdr3A_avg_df, GIL_nonbind_cdr3A_avg_df])
cdr3A_bind_scores = collect_cdr3_scores(GLC_binder_names, GLC_cdr3A_tsv_path, GLC_background_df)
cdr3A_nonbind_scores = collect_cdr3_scores(GLC_nonbinder_names, GLC_cdr3A_tsv_path, GLC_background_df)
GLC_cdr3A_combined_scores_df = combine_scores(cdr3A_bind_scores, cdr3A_nonbind_scores, peptide='GLC')

## GLC beta
GLC_background_df = combine_avg_probs_df([NLV_bind_cdr3B_avg_df, NLV_nonbind_cdr3B_avg_df, YLQ_bind_cdr3B_avg_df, YLQ_nonbind_cdr3B_avg_df, GIL_bind_cdr3B_avg_df, GIL_nonbind_cdr3B_avg_df])
cdr3B_bind_scores = collect_cdr3_scores(GLC_binder_names, GLC_cdr3B_tsv_path, GLC_background_df)
cdr3B_nonbind_scores = collect_cdr3_scores(GLC_nonbinder_names, GLC_cdr3B_tsv_path, GLC_background_df)
GLC_cdr3B_combined_scores_df = combine_scores(cdr3B_bind_scores, cdr3B_nonbind_scores, peptide='GLC')

## YLQ alpha
YLQ_background_df = combine_avg_probs_df([NLV_bind_cdr3A_avg_df, NLV_nonbind_cdr3A_avg_df, GIL_bind_cdr3A_avg_df, GIL_nonbind_cdr3A_avg_df, GLC_bind_cdr3A_avg_df, GLC_nonbind_cdr3A_avg_df])
cdr3A_bind_scores = collect_cdr3_scores(YLQ_binder_names, YLQ_cdr3A_tsv_path, YLQ_background_df)
cdr3A_nonbind_scores = collect_cdr3_scores(YLQ_nonbinder_names, YLQ_cdr3A_tsv_path, YLQ_background_df)
YLQ_cdr3A_combined_scores_df = combine_scores(cdr3A_bind_scores, cdr3A_nonbind_scores, peptide='YLQ')

## YLQ beta
YLQ_background_df = combine_avg_probs_df([NLV_bind_cdr3B_avg_df, NLV_nonbind_cdr3B_avg_df, GIL_bind_cdr3B_avg_df, GIL_nonbind_cdr3B_avg_df, GLC_bind_cdr3B_avg_df, GLC_nonbind_cdr3B_avg_df])
cdr3B_bind_scores = collect_cdr3_scores(YLQ_binder_names, YLQ_cdr3B_tsv_path, YLQ_background_df)
cdr3B_nonbind_scores = collect_cdr3_scores(YLQ_nonbinder_names, YLQ_cdr3B_tsv_path, YLQ_background_df)
YLQ_cdr3B_combined_scores_df = combine_scores(cdr3B_bind_scores, cdr3B_nonbind_scores, peptide='YLQ')

## NLV alpha
NLV_background_df = combine_avg_probs_df([YLQ_bind_cdr3A_avg_df, YLQ_nonbind_cdr3A_avg_df, GIL_bind_cdr3A_avg_df, GIL_nonbind_cdr3A_avg_df, GLC_bind_cdr3A_avg_df, GLC_nonbind_cdr3A_avg_df])
cdr3A_bind_scores = collect_cdr3_scores(NLV_binder_names, NLV_cdr3A_tsv_path, NLV_background_df)
cdr3A_nonbind_scores = collect_cdr3_scores(NLV_nonbinder_names, NLV_cdr3A_tsv_path, NLV_background_df)
NLV_cdr3A_combined_scores_df = combine_scores(cdr3A_bind_scores, cdr3A_nonbind_scores, peptide='NLV')

## NLV beta
NLV_background_df = combine_avg_probs_df([YLQ_bind_cdr3B_avg_df, YLQ_nonbind_cdr3B_avg_df, GIL_bind_cdr3B_avg_df, GIL_nonbind_cdr3B_avg_df, GLC_bind_cdr3B_avg_df, GLC_nonbind_cdr3B_avg_df])
cdr3B_bind_scores = collect_cdr3_scores(NLV_binder_names, NLV_cdr3B_tsv_path, NLV_background_df)
cdr3B_nonbind_scores = collect_cdr3_scores(NLV_nonbinder_names, NLV_cdr3B_tsv_path, NLV_background_df)
NLV_cdr3B_combined_scores_df = combine_scores(cdr3B_bind_scores, cdr3B_nonbind_scores, peptide='NLV')

## combine all scores to df
all_scores_cdr3A_df = pd.concat([GIL_cdr3A_combined_scores_df, GLC_cdr3A_combined_scores_df, YLQ_cdr3A_combined_scores_df, NLV_cdr3A_combined_scores_df])
all_scores_cdr3B_df = pd.concat([GIL_cdr3B_combined_scores_df, GLC_cdr3B_combined_scores_df, YLQ_cdr3B_combined_scores_df, NLV_cdr3B_combined_scores_df])
#################### Visualize ######################

sns.set_style("whitegrid")

### Plot 1: Histogram of average CDR3α preds (binders)
plot_cdr3_histogram(pepnames=['GIL', 'GLC', 'YLQ', 'NLV'], 
                        average_dfs=[GIL_bind_cdr3A_avg_df, GLC_bind_cdr3A_avg_df, YLQ_bind_cdr3A_avg_df, NLV_bind_cdr3A_avg_df], 
                        ylim=[0,0.190], 
                        plot_title='CDR3α (Binders)',
                        filename=Path(RESULTSPATH, 'cdr3A_avg_distribution.pdf'))

### Plot 2: Histogram of average CDR3β preds (binders)
plot_cdr3_histogram(pepnames=['GIL', 'GLC', 'YLQ', 'NLV'], 
                        average_dfs=[GIL_bind_cdr3B_avg_df, GLC_bind_cdr3B_avg_df, YLQ_bind_cdr3B_avg_df, NLV_bind_cdr3B_avg_df], 
                        ylim=[0,0.29], 
                        plot_title='CDR3β (Binders)',
                        filename=Path(RESULTSPATH, 'cdr3B_avg_distribution.pdf'))

### Plot 3: Correlation heatmap CDR3 alpha
all_pos_avg_df = combine_avg_probs_df([GIL_bind_cdr3A_avg_df, GLC_bind_cdr3A_avg_df, YLQ_bind_cdr3A_avg_df, NLV_bind_cdr3A_avg_df])
all_neg_avg_df = combine_avg_probs_df([GIL_nonbind_cdr3A_avg_df, GLC_nonbind_cdr3A_avg_df, YLQ_nonbind_cdr3A_avg_df, NLV_nonbind_cdr3A_avg_df])
total_avg_df = combine_avg_probs_df([all_pos_avg_df, all_neg_avg_df])

all_dfs = [GIL_bind_cdr3A_avg_df, GIL_nonbind_cdr3A_avg_df, 
            GLC_bind_cdr3A_avg_df, GLC_nonbind_cdr3A_avg_df, 
            YLQ_bind_cdr3A_avg_df, YLQ_nonbind_cdr3A_avg_df, 
            NLV_bind_cdr3A_avg_df, NLV_nonbind_cdr3A_avg_df,
            all_pos_avg_df, all_neg_avg_df, total_avg_df]
combination_names = ['GIL (+)', 'GIL (-)', 
                        'GLC (+)', 'GLC (-)', 
                        'YLQ (+)', 'YLQ (-)', 
                        'NLV (+)', 'NLV (-)', 
                        'All (+)', 'All (-)', 'All']

plot_cdr3_heatmap_corr(all_dfs, 
                        combination_names, 
                        plot_title='CDR3α', 
                        filename=Path(RESULTSPATH, 'cdr3A_heatmap.pdf'))

### Plot 4: Correlation heatmap CDR3 beta
all_pos_avg_df = combine_avg_probs_df([GIL_bind_cdr3B_avg_df, GLC_bind_cdr3B_avg_df, YLQ_bind_cdr3B_avg_df, NLV_bind_cdr3B_avg_df])
all_neg_avg_df = combine_avg_probs_df([GIL_nonbind_cdr3B_avg_df, GLC_nonbind_cdr3B_avg_df, YLQ_nonbind_cdr3B_avg_df, NLV_nonbind_cdr3B_avg_df])
total_avg_df = combine_avg_probs_df([all_pos_avg_df, all_neg_avg_df])

all_dfs = [GIL_bind_cdr3B_avg_df, GIL_nonbind_cdr3B_avg_df, 
            GLC_bind_cdr3B_avg_df, GLC_nonbind_cdr3B_avg_df, 
            YLQ_bind_cdr3B_avg_df, YLQ_nonbind_cdr3B_avg_df, 
            NLV_bind_cdr3B_avg_df, NLV_nonbind_cdr3B_avg_df,
            all_pos_avg_df, all_neg_avg_df, total_avg_df]
combination_names = ['GIL (+)', 'GIL (-)', 
                        'GLC (+)', 'GLC (-)', 
                        'YLQ (+)', 'YLQ (-)', 
                        'NLV (+)', 'NLV (-)', 
                        'All (+)', 'All (-)', 'All']

plot_cdr3_heatmap_corr(all_dfs, 
                        combination_names, 
                        plot_title='CDR3β', 
                        filename=Path(RESULTSPATH, 'cdr3B_heatmap.pdf'))


### Plot 5: Boxplots of CDR3 scores (alpha)
fig = plt.subplots(figsize=(10, 5))
sns.set_style("whitegrid")
sns.set_palette('Paired')

ax = sns.boxplot(x=all_scores_cdr3A_df["peptide"], 
                    y=all_scores_cdr3A_df["score"], 
                    hue=all_scores_cdr3A_df['binder'],
                    hue_order=[1,0], 
                    showmeans=True)

add_stat_annotation(ax, data=all_scores_cdr3A_df, x='peptide', y='score', hue='binder',
                    box_pairs=[(("GIL", 0), ("GIL", 1)),
                                (("GLC", 0), ("GLC", 1)),
                                (("YLQ", 0), ("YLQ", 1)),
                                (("NLV", 0), ("NLV", 1))],
                    test='t-test_ind', text_format='star', loc='inside', verbose=1, comparisons_correction=None)

plt.suptitle("CDR3α", fontsize=20)
plt.ylabel("Score")
plt.xlabel("")
plt.savefig(Path(RESULTSPATH, 'CDRalpha_boxplots.pdf'), format="pdf", dpi=1200)


### Plot 6: Boxplots of CDR3 scores (beta)
fig = plt.subplots(figsize=(10, 5))
sns.set_style("whitegrid")
sns.set_palette('Paired')

ax = sns.boxplot(x=all_scores_cdr3B_df["peptide"], 
                    y=all_scores_cdr3B_df["score"], 
                    hue=all_scores_cdr3B_df['binder'], 
                    hue_order=[1,0],
                    showmeans=True)

add_stat_annotation(ax, data=all_scores_cdr3B_df, x='peptide', y='score', hue='binder',
                    box_pairs=[(("GIL", 0), ("GIL", 1)),
                                (("GLC", 0), ("GLC", 1)),
                                (("NLV", 0), ("NLV", 1)),
                                (("YLQ", 0), ("YLQ", 1))],
                    test='t-test_ind', text_format='star', loc='inside', verbose=1, comparisons_correction=None)

plt.suptitle("CDR3β", fontsize=20)
plt.ylabel("Score")
plt.xlabel("")
plt.savefig(Path(RESULTSPATH, 'CDRbeta_boxplots.pdf'), format="pdf", dpi=1200)