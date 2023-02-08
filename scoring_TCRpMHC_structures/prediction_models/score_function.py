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

# dotenv_path = find_dotenv()
# ROOTPATH = Path(find_dotenv()).parent
# PROCESSEDPATH = Path(ROOTPATH, 'data/processed')
# RESULTSPATH = Path(ROOTPATH, 'reports/figures') ## change to new git structures
dotenv_path = find_dotenv()
ROOTPATH = Path(find_dotenv())
ESMPATH = Path(ROOTPATH, 'data/ESM-IF1_predictions')
RESULTSPATH = Path(ROOTPATH, 'result_figures') ## change to new git structures

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


def collect_scores_bg(pep_seq, complex_names, tsv_path, background_df):
    """calculate log-odds peptide scores for all complexes of same peptide"""

    sum_logodds = {}
    position_logodds = {}

    for complex in complex_names:
        complex_df = pd.read_csv(str(tsv_path) + '/' + complex + '.tsv', sep='\t', index_col=0)
        logodds_df = compute_logodds(complex_df, background_df)

        pos_scores = [0]*len(pep_seq)
        for i, letter in enumerate(pep_seq):
            pos_scores[i] = logodds_df[letter][i+1]
        # add peptide score
        position_logodds[complex] = pos_scores
        sum_logodds[complex] = sum(pos_scores)

    return sum_logodds, position_logodds


def combine_position_scores(bind_pos_scores_dict, nonbind_pos_scores_dict):
    """combine binder + non-binder scores to one df"""
    # binders
    bind_score_df = pd.DataFrame.from_dict(data=bind_pos_scores_dict, orient='index', columns=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'])
    bind_score_df['binder'] = 1

    # non binders
    nonbind_score_df = pd.DataFrame.from_dict(data=nonbind_pos_scores_dict, orient='index', columns=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'])
    nonbind_score_df['binder'] = 0

    position_scores_df = pd.concat([bind_score_df, nonbind_score_df])
    
    return position_scores_df


def combine_sum_scores(bind_scores_dict, nonbind_scores_dict, peptide=''):
    """Combine binder and non-binder sum log2(odds) scores"""
    # binders
    bind_score_df = pd.DataFrame.from_dict(data=bind_scores_dict, orient='index', columns=['score'])
    bind_score_df['binder'] = 1

    # non binders
    nonbind_score_df = pd.DataFrame.from_dict(data=nonbind_scores_dict, orient='index', columns=['score'])
    nonbind_score_df['binder'] = 0

    combined_scores_df = pd.concat([bind_score_df, nonbind_score_df])
    combined_scores_df['peptide'] = peptide
    
    return combined_scores_df


def peptide_score_performance(bind_scores, nonbind_scores, peptidename, plot=True, y_pred_threshold=None):
    """visualize binder vs non-binder scores in boxplots"""

    plot_df = pd.DataFrame.from_dict({'bind':bind_scores, 'non-bind':nonbind_scores})
    # pval
    p_val = stats.ttest_ind(list(bind_scores.values()), list(nonbind_scores.values())).pvalue

    # auc 
    y_preds = list(bind_scores.values()) + list(nonbind_scores.values())
    y_true_binary = [1]*len(bind_scores) + [0]*len(nonbind_scores)
    auc = metrics.roc_auc_score(y_true_binary, y_preds)

    # calculate optimal threshold
    if y_pred_threshold == None:
        fpr_, tpr_, thresholds = metrics.roc_curve(y_true_binary, y_preds)
        optimal_idx = np.argmax(tpr_ - fpr_)
        y_pred_threshold = thresholds[optimal_idx]
    
    # Convert binary
    y_pred_binary = y_preds >= y_pred_threshold

    # threshold-dependent measures
    precision = np.round(metrics.precision_score(y_true_binary, y_pred_binary), 3)
    mcc = np.round(metrics.matthews_corrcoef(y_true_binary, y_pred_binary), 3)

    # Confusion matrix
    conf = metrics.confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = conf.ravel()
    tnr = np.round(tn / (tn+fp), 3)
    tpr = np.round(tp / (tp+fn), 3)
    fpr = np.round(fp / (fp+tn), 3)

    # plot 
    if plot == True:
        sns.boxplot(plot_df) 
        plt.title(f'Peptide scores for {peptidename}\n P-val: {p_val}, auc: {auc:.3f}')
        plt.ylabel('Score')
        plt.show()

    return auc, mcc, conf, y_pred_threshold


def plot_aucs(pep_names, all_aucs, ylim=[0,1], results_fp=''):
    """ Plots AUCs for models """

    sns.set_style("whitegrid")
    sns.barplot(x=pep_names, y=all_aucs, palette='colorblind')
    plt.xticks()
    plt.ylim(ylim) 
    plt.ylabel('AUC')

    for i in range(len(pep_names)):
        plt.annotate(str(round(all_aucs[i],3)), xy=(i,all_aucs[i]), ha='center', va='bottom')

    plt.savefig(results_fp, format="pdf", dpi=600)


################################ Get scores ################################

print("Loading average scores ...")

# GIL 
GIL_df = pd.read_csv(Path(ESMPATH, 'peptide_features/GIL/features_padding.tsv'), sep='\t', index_col=0)
GIL_tsv_path = Path(ESMPATH, 'peptide_features/GIL/tsv_files_padding/')
GIL_binder_names = list(GIL_df[GIL_df['binder'] == 1].index)
GIL_nonbinder_names = list(GIL_df[GIL_df['binder'] == 0].index)
GIL_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/GIL_avg_df.tsv'), sep='\t', index_col=0)
GIL_nonbind_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/GIL_nonbind_avg_df.tsv'), sep='\t', index_col=0)

# GLC
GLC_df = pd.read_csv(Path(ESMPATH, 'peptide_features/GLC/features_padding.tsv'), sep='\t', index_col=0)
GLC_tsv_path = Path(ESMPATH, 'peptide_features/GLC/tsv_files_padding/')
GLC_binder_names = list(GLC_df[GLC_df['binder'] == 1].index)
GLC_nonbinder_names = list(GLC_df[GLC_df['binder'] == 0].index)
GLC_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/GLC_avg_df.tsv'), sep='\t', index_col=0)
GLC_nonbind_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/GLC_nonbind_avg_df.tsv'), sep='\t', index_col=0)

# YLQ
YLQ_df = pd.read_csv(Path(ESMPATH, 'peptide_features/YLQ/features_padding.tsv'), sep='\t', index_col=0)
YLQ_tsv_path = Path(ESMPATH, 'peptide_features/YLQ/tsv_files_padding/')
YLQ_binder_names = list(YLQ_df[YLQ_df['binder'] == 1].index)
YLQ_nonbinder_names = list(YLQ_df[YLQ_df['binder'] == 0].index)
YLQ_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/YLQ_avg_df.tsv'), sep='\t', index_col=0)
YLQ_nonbind_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/YLQ_nonbind_avg_df.tsv'), sep='\t', index_col=0)

# NLV
NLV_df = pd.read_csv(Path(ESMPATH, 'peptide_features/NLV/features_padding.tsv'), sep='\t', index_col=0)
NLV_tsv_path = Path(ESMPATH, 'peptide_features/NLV/tsv_files_padding/')
NLV_binder_names = list(NLV_df[NLV_df['binder'] == 1].index)
NLV_nonbinder_names = list(NLV_df[NLV_df['binder'] == 0].index)
NLV_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/NLV_avg_df.tsv'), sep='\t', index_col=0)
NLV_nonbind_avg_df = pd.read_csv(Path(ESMPATH, 'background_models/NLV_nonbind_avg_df.tsv'), sep='\t', index_col=0)

# compute scores
print("Calculating scores ...")

GIL_background_df = combine_avg_probs_df([GLC_avg_df, GLC_nonbind_avg_df, YLQ_avg_df, YLQ_nonbind_avg_df, NLV_avg_df, NLV_nonbind_avg_df])
GIL_bind_scores, GIL_bind_position_scores = collect_scores_bg('GILGFVFTL', GIL_binder_names, GIL_tsv_path, GIL_background_df)
GIL_nonbind_scores, GIL_nonbind_position_scores = collect_scores_bg('GILGFVFTL', GIL_nonbinder_names, GIL_tsv_path, GIL_background_df)
GIL_auc, GIL_mcc, GIL_conf, GIL_pred_threshold = peptide_score_performance(GIL_bind_scores, GIL_nonbind_scores, 'GIL', plot=False, y_pred_threshold=None)

print("GIL done ...")

GLC_background_df = combine_avg_probs_df([YLQ_avg_df, YLQ_nonbind_avg_df, GIL_avg_df, GIL_nonbind_avg_df, NLV_avg_df, NLV_nonbind_avg_df])
GLC_bind_scores, GLC_bind_position_scores = collect_scores_bg('GLCTLVAML', GLC_binder_names, GLC_tsv_path, GLC_background_df)
GLC_nonbind_scores, GLC_nonbind_position_scores = collect_scores_bg('GLCTLVAML', GLC_nonbinder_names, GLC_tsv_path, GLC_background_df)
GLC_auc, GLC_mcc, GLC_conf, GLC_pred_threshold = peptide_score_performance(GLC_bind_scores, GLC_nonbind_scores, 'GLC', plot=False, y_pred_threshold=None)

print("GLC done ...")

YLQ_background_df = combine_avg_probs_df([GLC_avg_df, GLC_nonbind_avg_df, GIL_avg_df, GIL_nonbind_avg_df, NLV_avg_df, NLV_nonbind_avg_df])
YLQ_bind_scores, YLQ_bind_position_scores = collect_scores_bg('YLQPRTFLL', YLQ_binder_names, YLQ_tsv_path, YLQ_background_df)
YLQ_nonbind_scores, YLQ_nonbind_position_scores = collect_scores_bg('YLQPRTFLL', YLQ_nonbinder_names, YLQ_tsv_path, YLQ_background_df)
YLQ_auc, YLQ_mcc, YLQ_conf, YLQ_pred_threshold = peptide_score_performance(YLQ_bind_scores, YLQ_nonbind_scores, 'YLQ', plot=False, y_pred_threshold=None)

print("YLQ done ...")

NLV_background_df = combine_avg_probs_df([YLQ_avg_df, YLQ_nonbind_avg_df, GIL_avg_df, GIL_nonbind_avg_df, GLC_avg_df, GLC_nonbind_avg_df])
NLV_bind_scores, NLV_bind_position_scores = collect_scores_bg('NLVPMVATV', NLV_binder_names, NLV_tsv_path, NLV_background_df)
NLV_nonbind_scores, NLV_nonbind_position_scores = collect_scores_bg('NLVPMVATV', NLV_nonbinder_names, NLV_tsv_path, NLV_background_df)
NLV_auc, NLV_mcc, NLV_conf, NLV_pred_threshold = peptide_score_performance(NLV_bind_scores, NLV_nonbind_scores, 'NLV', plot=False, y_pred_threshold=None)

print("NLV done ...")

### Combine Summed log-odds scores ###
GIL_combined_scores = combine_sum_scores(GIL_bind_scores, GIL_nonbind_scores, peptide='GIL')
GLC_combined_scores = combine_sum_scores(GLC_bind_scores, GLC_nonbind_scores, peptide='GLC')
NLV_combined_scores = combine_sum_scores(NLV_bind_scores, NLV_nonbind_scores, peptide='NLV')
YLQ_combined_scores = combine_sum_scores(YLQ_bind_scores, YLQ_nonbind_scores, peptide='YLQ')

## Add identities
combined_all = [GIL_combined_scores, GLC_combined_scores, YLQ_combined_scores, NLV_combined_scores]
features_all = [GIL_df, GLC_df, YLQ_df, NLV_df]
for combined_df, features_df in zip(combined_all, features_all):
    for identity_avg in ['total_iden_avg', 'pep_iden_avg', 'mhc_iden_avg', 'tcrA_iden_avg', 'tcrB_iden_avg']:
        combined_df[identity_avg] = features_df[identity_avg]

# combine all peptide scores (summed log-odds)
all_combined_scores = pd.concat([GIL_combined_scores, GLC_combined_scores, YLQ_combined_scores, NLV_combined_scores])


### Combined positions scores ###
GIL_position_scores_df = combine_position_scores(GIL_bind_position_scores, GIL_nonbind_position_scores)
GLC_position_scores_df = combine_position_scores(GLC_bind_position_scores, GLC_nonbind_position_scores)
YLQ_position_scores_df = combine_position_scores(YLQ_bind_position_scores, YLQ_nonbind_position_scores)
NLV_position_scores_df = combine_position_scores(NLV_bind_position_scores, NLV_nonbind_position_scores)

################################ Visualizations ################################

print("Doing visualizations ...")

### Plot 1: peptide AUCs ###
plot_aucs(['GIL', 'GLC', 'YLQ', 'NLV'], 
            [GIL_auc, GLC_auc, YLQ_auc, NLV_auc], 
            ylim=[0.45,0.65],
            results_fp=Path(RESULTSPATH, 'SF_GGYN_AUCs.pdf')
            )

### Plot 2: boxplots of scores ###
fig = plt.subplots(figsize=(12, 6))
sns.set_style("whitegrid")
sns.set_palette('Paired')
ax = sns.boxplot(x=all_combined_scores["peptide"], 
                    y=all_combined_scores["score"], 
                    hue=all_combined_scores['binder'], 
                    hue_order=[1,0], 
                    showmeans=True)    

add_stat_annotation(ax, data=all_combined_scores, x='peptide', y='score', hue='binder',
                    box_pairs=[(("GIL", 0), ("GIL", 1)),
                                (("GLC", 0), ("GLC", 1)),
                                (("YLQ", 0), ("YLQ", 1)),
                                (("NLV", 0), ("NLV", 1))],
                    test='t-test_ind', text_format='star', loc='inside', verbose=1, comparisons_correction=None)

plt.ylabel("Peptide score")
plt.xlabel("")
plt.savefig(Path(RESULTSPATH, 'SF_GGYN_boxplots.pdf'), format="pdf", dpi=600)


### Plot 3: correlation between scores and identities (all peptides) ###
identities = ['total_iden_avg', 'pep_iden_avg', 'mhc_iden_avg', 'tcrA_iden_avg', 'tcrB_iden_avg']
identity_titles = ['Total', 'Peptide', 'MHC', 'TCRα', 'TCRβ']
non_binders = all_combined_scores[all_combined_scores['binder'] == 0]
binders = all_combined_scores[all_combined_scores['binder'] == 1]

sns.reset_defaults()

fig, axes = plt.subplots(1,5, figsize=(24, 6))

for i in range(5):
    pos_PCC, pos_pval = stats.pearsonr(binders[identities[i]], binders['score'])
    neg_PCC, neg_pval = stats.pearsonr(non_binders[identities[i]], non_binders['score'])
    print(f"{identity_titles[i]}: \nNegatives: PCC={neg_PCC:.2f}, p-value={neg_pval:.2e} \nPositives: PCC={pos_PCC:.2f}, p-value={pos_pval:.2e}")
    # plot
    sns.scatterplot(x=non_binders[identities[i]], y=non_binders['score'], ax=axes[i], s=14, label=f'Negatives: \nPCC={neg_PCC:.2f}')
    sns.scatterplot(x=binders[identities[i]], y=binders['score'], ax=axes[i], s=14, label=f'Positives: \nPCC={pos_PCC:.2f}')
    
    axes[i].set_title(f"{identity_titles[i]}", fontsize=18)
    axes[i].set_xlabel(f"Avg. sequence similarity", fontsize=14)
    axes[i].set_ylabel("")
    axes[i].legend(loc='lower right', markerscale=2, fontsize=14)

axes[0].set_ylabel("Peptide score", fontsize=14)
fig.savefig(Path(RESULTSPATH, 'SF_GGYN_identity_corr.png'), format="png")


### Plot 4: correlation between scores and identities (Peptide-stratified) ###
palette_in_order = ["#0173b2", "#d55e00", "#de8f05", "#029e73"]
sns.set_palette(palette_in_order)

dataframes = [binders, non_binders]
pos_or_neg = ['Positives', 'Negatives']

fig, axes = plt.subplots(2,5, figsize=(25, 13))

count = 0 
for i in range(2):
    for j in range(5):
        # peptide PCCs and p-values
        all_PCC, all_pval = stats.pearsonr(dataframes[i][identities[j]], dataframes[i]['score'])
        GIL_dataframe = dataframes[i][dataframes[i]['peptide'] == 'GIL']
        GIL_PCC, GIL_pval = stats.pearsonr(GIL_dataframe[identities[j]], GIL_dataframe['score'])
        GLC_dataframe = dataframes[i][dataframes[i]['peptide'] == 'GLC']
        GLC_PCC, GLC_pval = stats.pearsonr(GLC_dataframe[identities[j]], GLC_dataframe['score'])
        NLV_dataframe = dataframes[i][dataframes[i]['peptide'] == 'NLV']
        NLV_PCC, NLV_pval = stats.pearsonr(NLV_dataframe[identities[j]], NLV_dataframe['score'])
        YLQ_dataframe = dataframes[i][dataframes[i]['peptide'] == 'YLQ']
        YLQ_PCC, YLQ_pval = stats.pearsonr(YLQ_dataframe[identities[j]], YLQ_dataframe['score'])

        # plot
        sns.scatterplot(x=dataframes[i][identities[j]], y=dataframes[i]['score'], hue=dataframes[i]['peptide'], hue_order=['GIL', 'NLV', 'GLC', 'YLQ'], ax=axes[i,j], s=12) 
        axes[i,j].set_title(f"{identity_titles[j]} ({pos_or_neg[i]})", fontsize=18)
        axes[i,j].set_xlabel(f"Avg. sequence similarity", fontsize=14)
        axes[i,j].set_ylabel("")
        axes[i,j].legend(loc='lower right', fontsize=11, labels=[f'All: PCC={all_PCC:.2f}, p-val={all_pval:.2e}', 
                                                    f'GIL: PCC={GIL_PCC:.2f}, p-val={GIL_pval:.2e}', 
                                                    f'NLV: PCC={NLV_PCC:.2f}, p-val={NLV_pval:.2e}', 
                                                    f'GLC: PCC={GLC_PCC:.2f}, p-val={GLC_pval:.2e}', 
                                                    f'YLQ: PCC={YLQ_PCC:.2f}, p-val={YLQ_pval:.2e}'])
        count += 1

    axes[i,0].set_ylabel("Peptide score", fontsize=14)
fig.savefig(Path(RESULTSPATH, 'SF_GGYN_identity_corr_pepStratified.png'), format="png")


### Plot 5: Distribution of position log-odds scores
sns.reset_defaults()
sns.set_style("whitegrid")
sns.set_palette("Paired")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

pep_names = ['GILGFVFTL', 'GLCTLVAML', 'YLQPRTFLL', 'NLVPMVATV']
peptides = [GIL_position_scores_df, GLC_position_scores_df, YLQ_position_scores_df, NLV_position_scores_df]

sns.set_style("whitegrid")
count = 0
for i in range(2):
    for j in range(2):
        data_melted = peptides[count].melt(id_vars=['binder'], value_vars=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'],
                        var_name='Position', value_name='Score')
        box = sns.boxplot(
            data=data_melted,
            x='Position',
            y='Score',
            hue='binder',
            hue_order=[1,0],
            ax=axes[i,j],
            showmeans=True
        )
        add_stat_annotation(box, data=data_melted, x='Position', y='Score', hue='binder',
                    box_pairs=[(("S1", 0), ("S1", 1)),
                                (("S2", 0), ("S2", 1)),
                                (("S3", 0), ("S3", 1)),
                                (("S4", 0), ("S4", 1)),
                                (("S5", 0), ("S5", 1)),
                                (("S6", 0), ("S6", 1)),
                                (("S7", 0), ("S7", 1)),
                                (("S8", 0), ("S8", 1)),
                                (("S9", 0), ("S9", 1))],
                    test='t-test_ind', text_format='star', loc='inside', verbose=1, comparisons_correction=None)

        axes[i,j].set_ylim([-12.4, 7])
        axes[i,j].set_xlabel("")
        axes[i,j].set_xticklabels(np.arange(1,10, dtype=int))
        axes[i,j].set_title(f"{pep_names[count]}")
        axes[i,j].legend(loc='upper right')
        count += 1

plt.savefig(Path(RESULTSPATH, 'SF_GGYN_pos_scores.pdf'), format="pdf", dpi=1200)