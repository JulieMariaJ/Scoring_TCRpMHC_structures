import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import glob 
from sklearn import metrics
import scipy.stats as stats
from Bio import SeqIO
import copy
from matplotlib import collections  as mc
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

### PATHS ###

dotenv_path = find_dotenv()
ROOTPATH = Path(find_dotenv()).parent
RAWPATH = Path(ROOTPATH, 'data/raw')
ESMPATH = Path(ROOTPATH, 'data/ESM-IF1_predictions')
RESULTSPATH = Path(ROOTPATH, 'result_figures') 

### FUNCTIONS ###

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def avg_df_from_tsvs(complex_names_list, tsv_path):
    """average probability across all complexes"""
    avg_df = pd.DataFrame(columns=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
       'R', 'S', 'T', 'V', 'W', 'Y'], index=[1,2,3,4,5,6,7,8,9])
    
    #iterate over each complex 
    for comp in complex_names_list:
        comp_tsv = pd.read_csv(Path(tsv_path, comp + '.tsv'), sep='\t', index_col=0)
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
        comp_tsv = pd.read_csv(Path(tsv_path, comp + '.tsv'), sep='\t', header=None, names=all_AA_letters)
        if comp_tsv.empty:
            continue
        cdr3_avg_df = pd.DataFrame(comp_tsv.drop(['letter'], axis=1).mean(axis=0), columns=['avg_score'])
        avg_df = avg_df.add(cdr3_avg_df, axis='index')
    
    avg_df = avg_df.div(len(complex_names))
    return avg_df

def collect_cdr3_scores(complex_names, tsv_path, background_df):
    """ Get avg CDR3 scores """
    all_AA_letters = ['Pos', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'letter']
    cdr3_scores = {}
    
    #iterate over complexes 
    for complex in complex_names:
        complex_df = pd.read_csv(Path(tsv_path, complex + '.tsv'), sep='\t', index_col=None, header=None, names=all_AA_letters)
        if complex_df.empty:
            cdr3_scores[complex] = np.nan
        else:
            score_sum = 0
            for pep_pos, row in complex_df.iterrows():
                pos = row['Pos'][-1]
                letter = row['letter']
                prob = complex_df[letter][pep_pos]
                score_sum += np.log2(prob/background_df['avg_score'][str(letter)]) 

            # add avg score
            cdr3_scores[complex] = score_sum/len(complex_df) 

    return cdr3_scores

def extract_log_odds_bg(df_for_bg_list, pep_seq_dict, tsv_path, complex_names_list):
    """ get 9x1 and 9x20 prob dfs """
    #get background table
    q_df = combine_avg_probs_df(df_for_bg_list)
    complex_pos_scores = dict()
    complex_all_scores = dict()
    for comp in complex_names_list:
        pep_seq = pep_seq_dict[comp]
        #get log odds matrix
        comp_tsv = pd.read_csv(Path(tsv_path, comp + '.tsv'), sep='\t', index_col=0)
        odds_df = comp_tsv.div(q_df, fill_value=0)
        log_odds_df = odds_df.applymap(np.log2)
    
        #collect correct AA score 
        log_odds_score, all_log_odds = [], []
        for i, AA in enumerate(pep_seq):
            log_odds_score.append(log_odds_df[AA][i+1])
            all_log_odds.append(log_odds_df.iloc[i].values.tolist())

        complex_pos_scores[comp] = log_odds_score
        complex_all_scores[comp] = sum(all_log_odds, [])
    
    complex_pos_scores = pd.DataFrame.from_dict(complex_pos_scores, orient='index', columns=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'])
    complex_all_scores = pd.DataFrame.from_dict(complex_all_scores, orient='index')
    return complex_pos_scores, complex_all_scores

def prepare_logreg_features(df_for_bg, pep_df, pep_seq, tsv_path, cdr3A_scores, cdr3B_scores, replace_nan=0):
    """ save 6 different logreg feature dfs """
    # make 9x1 and 9x20 dfs
    pep_seq_dict = {pep_df.index[i]: pep_seq for i in range(len(pep_df))}
    pep_9x1_df, pep_9x20_df = extract_log_odds_bg(df_for_bg, pep_seq_dict, tsv_path, pep_df.index.tolist())
    
    # append binder status and partition
    pep_9x1_df['binder'] = pep_df['binder']
    pep_9x1_df['partition'] = pep_df['partition']
    pep_9x20_df['binder'] = pep_df['binder']
    pep_9x20_df['partition'] = pep_df['partition']
    
    # make 6x1 and 6x20 dfs 
    pep_6x1_df = copy.deepcopy(pep_9x1_df)
    pep_6x1_df = pep_6x1_df[['S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'binder', 'partition']]
    pep_6x20_df = copy.deepcopy(pep_9x20_df)
    pep_6x20_df = pep_6x20_df.drop(pep_6x20_df.iloc[:, 160:180],axis = 1)
    pep_6x20_df = pep_6x20_df.drop(pep_6x20_df.iloc[:, 0:40],axis = 1)

    # make 6x1+cdr3 and 6x20+cdr3 dfs 
    pep_9x1_cdr3_df = copy.deepcopy(pep_9x1_df)
    pep_9x1_cdr3_df['CDR3A_score'] = pd.DataFrame.from_dict(cdr3A_scores, orient='index',columns=['cdr3A_score'])['cdr3A_score']
    pep_9x1_cdr3_df['CDR3B_score'] = pd.DataFrame.from_dict(cdr3B_scores, orient='index',columns=['cdr3B_score'])['cdr3B_score']
    pep_9x1_cdr3_df['CDR3A_score'] = pep_9x1_cdr3_df['CDR3A_score'].fillna(replace_nan)
    pep_9x1_cdr3_df['CDR3B_score'] = pep_9x1_cdr3_df['CDR3B_score'].fillna(replace_nan)
    pep_9x20_cdr3_df = copy.deepcopy(pep_9x20_df)
    pep_9x20_cdr3_df['CDR3A_score'] = pd.DataFrame.from_dict(cdr3A_scores, orient='index',columns=['cdr3A_score'])['cdr3A_score']
    pep_9x20_cdr3_df['CDR3B_score'] = pd.DataFrame.from_dict(cdr3B_scores, orient='index',columns=['cdr3B_score'])['cdr3B_score']
    pep_9x20_cdr3_df['CDR3A_score'] = pep_9x20_cdr3_df['CDR3A_score'].fillna(replace_nan)
    pep_9x20_cdr3_df['CDR3B_score'] = pep_9x20_cdr3_df['CDR3B_score'].fillna(replace_nan)

    return pep_9x1_df, pep_9x20_df, pep_9x1_cdr3_df, pep_9x20_cdr3_df, pep_6x1_df, pep_6x20_df

def fit_logreg(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train.values.ravel())
    y_pred = logreg.predict(X_test)
    y_pred_proba = [x[1] for x in logreg.predict_proba(X_test)]
    # performance 
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
    return auc, fpr, tpr, logreg

def cv_logregmodel(features_df, feature_comb, cv_folds, plot_true=False):
    if plot_true == True:
        fig = plt.subplots()

    models = []
    for i in range(cv_folds):
        # Train and test split
        train_df = features_df[features_df['partition'] != i]
        val_df = features_df[features_df['partition'] == i]
        
        #define train and test X and y
        y_train = train_df['binder']
        X_train = train_df[train_df.columns.difference(['binder', 'partition'])] 
        y_val = val_df['binder']
        X_val = val_df[val_df.columns.difference(['binder', 'partition'])]
        
        #fit log reg
        auc, fpr, tpr, model = fit_logreg(X_train, y_train, X_val, y_val)
        models.append(model)
        
        # plot 
        if plot_true == True: 
            plt.plot(fpr, tpr, label = f"Fold {i+1}, AUC = {auc:.2f}")

    if plot_true == True:
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(f'LogReg using {feature_comb} as feature')
        plt.legend()
        plt.show()
    return models

def loo_test(test_df, models, feature_comb, plot_true):
    #define X and y in test 
    X_test = test_df[test_df.columns.difference(['binder', 'partition'])]
    y_test = test_df['binder']
    
    #test every model 
    collected_preds = []
    collected_y_test = []
    for i, model in enumerate(models):
        collected_y_test.append(y_test)
        y_pred_proba = [x[1] for x in model.predict_proba(X_test)]
        collected_preds.append(y_pred_proba)
    
    #ensemble preds 
    collected_y_test = np.array(collected_y_test).flatten()
    collected_preds = np.array(collected_preds).flatten() 
    prob_auc = metrics.roc_auc_score(collected_y_test, collected_preds)
    fpr, tpr, thresholds = metrics.roc_curve(collected_y_test, collected_preds)

    #plot
    if plot_true == True:
        plt.plot(fpr, tpr, label = f"Ensemble model on test, AUC = {prob_auc:.2f}")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(f'LogReg using {feature_comb} as feature')
        plt.legend()
        plt.show()
    
    return prob_auc, fpr, tpr

def cv_ds_logregmodel(features_df, feature_comb, downsampling=False, us_n=None):

    GIL_indexes_used = set()
    fig = plt.subplots()

    models = []
    for i in range(4):
        # Train and test split
        train_df = features_df[features_df['partition'] != i]
        test_df = features_df[features_df['partition'] == i]        # use if no downsampling
        GLC_train_df = train_df[train_df['peptide'] == 'GLC']
        #print(GLC_train_df)
        val_df = features_df[features_df['partition'] == i]
        
        #define GLC train and test X and y
        GLC_y_train = GLC_train_df['binder']
        GLC_X_train = GLC_train_df.drop(['binder', 'partition', 'peptide'], axis=1)
    
        # GIL
        GIL_train_df = train_df[train_df['peptide'] == 'GIL']
        GIL_y_train = GIL_train_df['binder']
        GIL_X_train = GIL_train_df.drop(['binder', 'partition','peptide'], axis=1)
        
        # downsampling of GIL
        if downsampling == True and us_n != None:
            GIL_X_train = GIL_X_train.sample(n = us_n)      #us_n = num to downsample to
            GIL_y_train = GIL_y_train.loc[list(GIL_X_train.index)]

        # combine GLC+GIL train and define val
        X_train = pd.concat([GLC_X_train, GIL_X_train])             
        y_train = pd.concat([GLC_y_train, GIL_y_train])   
        y_val = val_df['binder']
        X_val = val_df.drop(['binder', 'partition', 'peptide'], axis=1)
        
        GIL_indexes_used.update(set(GIL_X_train.index))

        # plot 
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train.values.ravel())
        models.append(logreg)
        y_pred = logreg.predict(X_val)
        y_pred_proba = [x[1] for x in logreg.predict_proba(X_val)]
        auc = metrics.roc_auc_score(y_val, y_pred)
        prob_auc = metrics.roc_auc_score(y_val, y_pred_proba)
        auc_prob_only = prob_auc
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred_proba)
        plt.plot(fpr, tpr, label = f"Fold {i+1}, AUC = {prob_auc:.2f}")

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(f'LogReg using {feature_comb} as feature')
    plt.legend()
    plt.show()
    return models, GIL_indexes_used

seed_everything(42)

### define GGYN paths ###

# GIL path 
GIL_df = pd.read_csv(Path(ESMPATH, 'peptide_features/GIL/features_padding.tsv'), sep='\t', index_col=0)
GIL_tsv_path = Path(ESMPATH, 'peptide_features/GIL/tsv_files_padding/')
GIL_binder_names = list(GIL_df[GIL_df['binder'] == 1].index)
GIL_nonbinder_names = list(GIL_df[GIL_df['binder'] == 0].index)

# GLC path 
GLC_df = pd.read_csv(Path(ESMPATH, 'peptide_features/GLC/features_padding.tsv'), sep='\t', index_col=0)
GLC_tsv_path = Path(ESMPATH, 'peptide_features/GLC/tsv_files_padding/')
GLC_binder_names = list(GLC_df[GLC_df['binder'] == 1].index)
GLC_nonbinder_names = list(GLC_df[GLC_df['binder'] == 0].index)

# YLQ path 
YLQ_df = pd.read_csv(Path(ESMPATH, 'peptide_features/YLQ/features_padding.tsv'), sep='\t', index_col=0)
YLQ_tsv_path = Path(ESMPATH, 'peptide_features/YLQ/tsv_files_padding/')
YLQ_binder_names = list(YLQ_df[YLQ_df['binder'] == 1].index)
YLQ_nonbinder_names = list(YLQ_df[YLQ_df['binder'] == 0].index)

# NLV path 
NLV_df = pd.read_csv(Path(ESMPATH, 'peptide_features/NLV/features_padding.tsv'), sep='\t', index_col=0)
NLV_tsv_path = Path(ESMPATH, 'peptide_features/NLV/tsv_files_padding/')
NLV_binder_names = list(NLV_df[NLV_df['binder'] == 1].index)
NLV_nonbinder_names = list(NLV_df[NLV_df['binder'] == 0].index)

### define CDR3 paths ###
#GIL
GIL_cdr3A_tsv_path = Path(ESMPATH,'CDR3_features/GIL/all_nearN_cdr3A_features')
GIL_cdr3B_tsv_path = Path(ESMPATH,'CDR3_features/GIL/all_nearN_cdr3B_features')

#GLC
GLC_cdr3A_tsv_path = Path(ESMPATH,'CDR3_features/GLC/all_nearN_cdr3A_features')
GLC_cdr3B_tsv_path = Path(ESMPATH,'CDR3_features/GLC/all_nearN_cdr3B_features')

#YLQ
YLQ_cdr3A_tsv_path = Path(ESMPATH,'CDR3_features/YLQ/all_nearN_cdr3A_features')
YLQ_cdr3B_tsv_path = Path(ESMPATH,'CDR3_features/YLQ/all_nearN_cdr3B_features')

#NLV
NLV_cdr3A_tsv_path = Path(ESMPATH,'CDR3_features/NLV/all_nearN_cdr3A_features')
NLV_cdr3B_tsv_path = Path(ESMPATH,'CDR3_features/NLV/all_nearN_cdr3B_features')

############################ make bg files #######################################

print("Generating average GGYN bg files ...")

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

############################## make CDR3 bgs #####################################

print("Generating average CDR3 bg files ...")

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


################## prep features ##################  

print("Generating log reg features table ...")
# features: 9x1, 9x20, 6x1, 6x20, 9x1+CDR3, and 9x20+CDR3 dataframes for each peptide (GIL, GLC, YLQ and NLV)

## GIL ##
# get cdr3 scores 
GIL_cdr3A_bg_df = combine_avg_probs_df([NLV_bind_avg_cdr3A_df, NLV_nonbind_avg_cdr3A_df, YLQ_bind_avg_cdr3A_df, YLQ_nonbind_avg_cdr3A_df, GLC_bind_avg_cdr3A_df, GLC_nonbind_avg_cdr3A_df])
GIL_cdr3A_scores = collect_cdr3_scores(GIL_df.index.tolist(), GIL_cdr3A_tsv_path, GIL_cdr3A_bg_df)
GIL_cdr3B_bg_df = combine_avg_probs_df([NLV_bind_avg_cdr3B_df, NLV_nonbind_avg_cdr3B_df, YLQ_bind_avg_cdr3B_df, YLQ_nonbind_avg_cdr3B_df, GLC_bind_avg_cdr3B_df, GLC_nonbind_avg_cdr3B_df])
GIL_cdr3B_scores = collect_cdr3_scores(GIL_df.index.tolist(), GIL_cdr3B_tsv_path, GIL_cdr3B_bg_df)
#bg for GIL
bg_for_gil = [GLC_avg_df, GLC_nonbind_avg_df, YLQ_avg_df, YLQ_nonbind_avg_df, NLV_avg_df, NLV_nonbind_avg_df]
#generate 6 feature dfs 
GIL_9x1_df, GIL_9x20_df, GIL_9x1_cdr3_df, GIL_9x20_cdr3_df, GIL_6x1_df, GIL_6x20_df = prepare_logreg_features(bg_for_gil, GIL_df, 'GILGFVFTL', GIL_tsv_path, GIL_cdr3A_scores, GIL_cdr3B_scores, replace_nan=0)

## GLC ##
# get cdr3 scores 
GLC_background_df = combine_avg_probs_df([NLV_bind_avg_cdr3A_df, NLV_nonbind_avg_cdr3A_df, YLQ_bind_avg_cdr3A_df, YLQ_nonbind_avg_cdr3A_df, GIL_bind_avg_cdr3A_df, GIL_nonbind_avg_cdr3A_df])
GLC_cdr3A_scores = collect_cdr3_scores(GLC_df.index.tolist(), GLC_cdr3A_tsv_path, GLC_background_df)
GLC_background_df = combine_avg_probs_df([NLV_bind_avg_cdr3B_df, NLV_nonbind_avg_cdr3B_df, YLQ_bind_avg_cdr3B_df, YLQ_nonbind_avg_cdr3B_df, GIL_bind_avg_cdr3B_df, GIL_nonbind_avg_cdr3B_df])
GLC_cdr3B_scores = collect_cdr3_scores(GLC_df.index.tolist(), GLC_cdr3B_tsv_path, GLC_background_df)
#bg for GLC
bg_for_glc = [GIL_avg_df, GIL_nonbind_avg_df, NLV_avg_df, NLV_nonbind_avg_df, YLQ_avg_df, YLQ_nonbind_avg_df]
#generate 6 feature dfs 
GLC_9x1_df, GLC_9x20_df, GLC_9x1_cdr3_df, GLC_9x20_cdr3_df, GLC_6x1_df, GLC_6x20_df = prepare_logreg_features(bg_for_glc, GLC_df, 'GLCTLVAML', GLC_tsv_path, GLC_cdr3A_scores, GLC_cdr3B_scores, replace_nan=0)

## YLQ ##
# get cdr3 scores
YLQ_background_df = combine_avg_probs_df([NLV_bind_avg_cdr3A_df, NLV_nonbind_avg_cdr3A_df, GIL_bind_avg_cdr3A_df, GIL_nonbind_avg_cdr3A_df, GLC_bind_avg_cdr3A_df, GLC_nonbind_avg_cdr3A_df])
YLQ_cdr3A_scores = collect_cdr3_scores(YLQ_df.index.tolist(), YLQ_cdr3A_tsv_path, YLQ_background_df)
YLQ_background_df = combine_avg_probs_df([NLV_bind_avg_cdr3B_df, NLV_nonbind_avg_cdr3B_df, GIL_bind_avg_cdr3B_df, GIL_nonbind_avg_cdr3B_df, GLC_bind_avg_cdr3B_df, GLC_nonbind_avg_cdr3B_df])
YLQ_cdr3B_scores = collect_cdr3_scores(YLQ_df.index.tolist(), YLQ_cdr3B_tsv_path, YLQ_background_df)
#bg for YLQ
bg_for_ylq = [GIL_avg_df, GIL_nonbind_avg_df, NLV_avg_df, NLV_nonbind_avg_df, GLC_avg_df, GLC_nonbind_avg_df]
#generate 6 feature dfs 
YLQ_9x1_df, YLQ_9x20_df, YLQ_9x1_cdr3_df, YLQ_9x20_cdr3_df, YLQ_6x1_df, YLQ_6x20_df = prepare_logreg_features(bg_for_ylq, YLQ_df, 'YLQPRTFLL', YLQ_tsv_path, YLQ_cdr3A_scores, YLQ_cdr3B_scores, replace_nan=0)

## NLV ##
# get cdr3 scores 
NLV_background_df = combine_avg_probs_df([YLQ_bind_avg_cdr3A_df, YLQ_nonbind_avg_cdr3A_df, GIL_bind_avg_cdr3A_df, GIL_nonbind_avg_cdr3A_df, GLC_bind_avg_cdr3A_df, GLC_nonbind_avg_cdr3A_df])
NLV_cdr3A_scores = collect_cdr3_scores(NLV_df.index.tolist(), NLV_cdr3A_tsv_path, NLV_background_df)
NLV_background_df = combine_avg_probs_df([YLQ_bind_avg_cdr3B_df, YLQ_nonbind_avg_cdr3B_df, GIL_bind_avg_cdr3B_df, GIL_nonbind_avg_cdr3B_df, GLC_bind_avg_cdr3B_df, GLC_nonbind_avg_cdr3B_df])
NLV_cdr3B_scores = collect_cdr3_scores(NLV_df.index.tolist(), NLV_cdr3B_tsv_path, NLV_background_df)
#bg for NLV
bg_for_nlv = [GIL_avg_df, GIL_nonbind_avg_df, YLQ_avg_df, YLQ_nonbind_avg_df, GLC_avg_df, GLC_nonbind_avg_df]
#generate 6 feature dfs 
NLV_9x1_df, NLV_9x20_df, NLV_9x1_cdr3_df, NLV_9x20_cdr3_df, NLV_6x1_df, NLV_6x20_df = prepare_logreg_features(bg_for_nlv, NLV_df, 'NLVPMVATV', NLV_tsv_path, NLV_cdr3A_scores, NLV_cdr3B_scores, replace_nan=0)


# ###### run logreg models ######
print("Running logreg models ...")

## 9x1 models ##
# GIL 
models = cv_logregmodel(GIL_9x1_df[GIL_9x1_df['partition'] != 4], 'GIL 9x1 log odds vector', 4, plot_true=False)
GIL_9x1_test_auc, _, _ = loo_test(GIL_9x1_df[GIL_9x1_df['partition'] == 4], models, 'GIL 9x1 log odds vector', plot_true=False)
# GLC
models = cv_logregmodel(GLC_9x1_df[GLC_9x1_df['partition'] != 4], 'GLC 9x1 log odds vector', 4, plot_true=False)
GLC_9x1_test_auc, _, _ = loo_test(GLC_9x1_df[GLC_9x1_df['partition'] == 4], models, 'GLC 9x1 log odds vector', plot_true=False)
# YLQ
models = cv_logregmodel(YLQ_9x1_df[YLQ_9x1_df['partition'] != 4], 'YLQ 9x1 log odds vector', 4, plot_true=False)
YLQ_9x1_test_auc, _, _ = loo_test(YLQ_9x1_df[YLQ_9x1_df['partition'] == 4], models, 'YLQ 9x1 log odds vector', plot_true=False)
# NLV
models = cv_logregmodel(NLV_9x1_df[NLV_9x1_df['partition'] != 4], 'NLV 9x1 log odds vector', 4, plot_true=False)
NLV_9x1_test_auc, _, _ = loo_test(NLV_9x1_df[NLV_9x1_df['partition'] == 4], models, 'NLV 9x1 log odds vector', plot_true=False)

## 9x20 models ## - also test on three other peptides
# GIL
models = cv_logregmodel(GIL_9x20_df[GIL_9x20_df['partition'] != 4], 'GIL 9x20 log odds vector', 4, plot_true=False)
GIL_9x20_test_auc, _, _ = loo_test(GIL_9x20_df[GIL_9x20_df['partition'] == 4], models, 'GIL 9x20 log odds vector', plot_true=False)
# test on three other peptides
test_auc_9x20_GIL, _, _ = loo_test(GLC_9x20_df.append(YLQ_9x20_df).append(NLV_9x20_df), models, 'fit GIL 9x20, test GLC+YLQ+NLV 9x20', plot_true=False)
# GLC 
models = cv_logregmodel(GLC_9x20_df[GLC_9x20_df['partition'] != 4], 'GLC 9x20 log odds vector', 4, plot_true=False)
GLC_9x20_test_auc, _, _ = loo_test(GLC_9x20_df[GLC_9x20_df['partition'] == 4], models, 'GLC 9x20 log odds vector', plot_true=False)
# test on new peptide
test_auc_9x20_GLC, _, _ = loo_test(GIL_9x20_df.append(YLQ_9x20_df).append(NLV_9x20_df), models, 'fit GLC 9x20, test GIL+YLQ+NLV 9x20', plot_true=False)
# YLQ
models = cv_logregmodel(YLQ_9x20_df[YLQ_9x20_df['partition'] != 4], 'YLQ 9x20 log odds vector', 4, plot_true=False)
YLQ_9x20_test_auc, _, _ = loo_test(YLQ_9x20_df[YLQ_9x20_df['partition'] == 4], models, 'YLQ 9x20 log odds vector', plot_true=False)
# test on new peptide
test_auc_9x20_YLQ, _, _ = loo_test(GIL_9x20_df.append(GLC_9x20_df).append(NLV_9x20_df), models, 'fit YLQ 9x20, test GIL+GLC+NLV 9x20', plot_true=False)
# NLV
models = cv_logregmodel(NLV_9x20_df[NLV_9x20_df['partition'] != 4], 'NLV 9x20 log odds vector', 4, plot_true=False)
NLV_9x20_test_auc, _, _ = loo_test(NLV_9x20_df[NLV_9x20_df['partition'] == 4], models, 'NLV 9x20 log odds vector', plot_true=False)
# test on new peptide
test_auc_9x20_NLV, _, _ = loo_test(GIL_9x20_df.append(GLC_9x20_df).append(YLQ_9x20_df), models, 'fit NLV 9x20, test GIL+GLC+YLQ 9x20', plot_true=False)


## 9x1 + CDR3 models ##
# GIL 
models = cv_logregmodel(GIL_9x1_cdr3_df[GIL_9x1_cdr3_df['partition'] != 4], 'GIL 9x1 + CDR3 log odds vector', 4, plot_true=False)
GIL_9x1_cdr3_test_auc, _, _ = loo_test(GIL_9x1_cdr3_df[GIL_9x1_cdr3_df['partition'] == 4], models, 'GIL 9x1 log odds vector', plot_true=False)
# GLC
models = cv_logregmodel(GLC_9x1_cdr3_df[GLC_9x1_cdr3_df['partition'] != 4], 'GLC 9x1 + CDR3 log odds vector', 4, plot_true=False)
GLC_9x1_cdr3_test_auc, _, _ = loo_test(GLC_9x1_cdr3_df[GLC_9x1_cdr3_df['partition'] == 4], models, 'GLC 9x1 log odds vector', plot_true=False)
# YLQ
models = cv_logregmodel(YLQ_9x1_cdr3_df[YLQ_9x1_cdr3_df['partition'] != 4], 'YLQ 9x1 + CDR3 log odds vector', 4, plot_true=False)
YLQ_9x1_cdr3_test_auc, _, _ = loo_test(YLQ_9x1_cdr3_df[YLQ_9x1_cdr3_df['partition'] == 4], models, 'YLQ 9x1 log odds vector', plot_true=False)
# NLV
models = cv_logregmodel(NLV_9x1_cdr3_df[NLV_9x1_cdr3_df['partition'] != 4], 'NLV 9x1 + CDR3 log odds vector', 4, plot_true=False)
NLV_9x1_cdr3_test_auc, _, _ = loo_test(NLV_9x1_cdr3_df[NLV_9x1_cdr3_df['partition'] == 4], models, 'NLV 9x1 + CDR3 log odds vector', plot_true=False)

## 9x20 models + CDR3 ## - also test on three other peptides
# GIL
models = cv_logregmodel(GIL_9x20_cdr3_df[GIL_9x20_cdr3_df['partition'] != 4], 'GIL 9x20 + CDR3 log odds vector', 4, plot_true=False)
GIL_9x20_cdr3_test_auc, _, _ = loo_test(GIL_9x20_cdr3_df[GIL_9x20_cdr3_df['partition'] == 4], models, 'GIL 9x20 + CDR3 log odds vector', plot_true=False)
# test on three other peptides
test_auc_9x20_cdr3_GIL, _, _ = loo_test(GLC_9x20_cdr3_df.append(YLQ_9x20_cdr3_df).append(NLV_9x20_cdr3_df), models, 'fit GIL 9x20 + CDR3, test GLC+YLQ+NLV 9x20 + CDR3', plot_true=False)
# GLC 
models = cv_logregmodel(GLC_9x20_cdr3_df[GLC_9x20_cdr3_df['partition'] != 4], 'GLC 9x20 + CDR3 log odds vector', 4, plot_true=False)
GLC_9x20_cdr3_test_auc, _, _ = loo_test(GLC_9x20_cdr3_df[GLC_9x20_cdr3_df['partition'] == 4], models, 'GLC 9x20 + CDR3 log odds vector', plot_true=False)
# test on new peptide
test_auc_9x20_cdr3_GLC, _, _ = loo_test(GIL_9x20_cdr3_df.append(YLQ_9x20_cdr3_df).append(NLV_9x20_cdr3_df), models, 'fit GLC 9x20 + CDR3, test GIL+YLQ+NLV 9x20 + CDR3', plot_true=False)
# YLQ
models = cv_logregmodel(YLQ_9x20_cdr3_df[YLQ_9x20_cdr3_df['partition'] != 4], 'YLQ 9x20 + CDR3 log odds vector', 4, plot_true=False)
YLQ_9x20_cdr3_test_auc, _, _ = loo_test(YLQ_9x20_cdr3_df[YLQ_9x20_cdr3_df['partition'] == 4], models, 'YLQ 9x20 + CDR3 log odds vector', plot_true=False)
# test on new peptide
test_auc_9x20_cdr3_YLQ, _, _ = loo_test(GIL_9x20_cdr3_df.append(GLC_9x20_cdr3_df).append(NLV_9x20_cdr3_df), models, 'fit YLQ 9x20 + CDR3, test GIL+GLC+NLV 9x20 + CDR3', plot_true=False)
# NLV
models = cv_logregmodel(NLV_9x20_cdr3_df[NLV_9x20_cdr3_df['partition'] != 4], 'NLV 9x20 + CDR3 log odds vector', 4, plot_true=False)
NLV_9x20_cdr3_test_auc, _, _ = loo_test(NLV_9x20_cdr3_df[NLV_9x20_cdr3_df['partition'] == 4], models, 'NLV 9x20 + CDR3 log odds vector', plot_true=False)
# test on new peptide
test_auc_9x20_cdr3_NLV, _, _ = loo_test(GIL_9x20_cdr3_df.append(GLC_9x20_cdr3_df).append(YLQ_9x20_cdr3_df), models, 'fit NLV 9x20 + CDR3, test GIL+GLC+YLQ 9x20 + CDR3', plot_true=False)


## 6x1 models ##
# GIL 
models = cv_logregmodel(GIL_6x1_df[GIL_6x1_df['partition'] != 4], 'GIL 6x1 log odds vector', 4, plot_true=False)
GIL_6x1_test_auc, _, _ = loo_test(GIL_6x1_df[GIL_6x1_df['partition'] == 4], models, 'GIL 6x1 log odds vector', plot_true=False)
# GLC
models = cv_logregmodel(GLC_6x1_df[GLC_6x1_df['partition'] != 4], 'GLC 6x1 log odds vector', 4, plot_true=False)
GLC_6x1_test_auc, _, _ = loo_test(GIL_6x1_df[GIL_6x1_df['partition'] == 4], models, 'GLC 6x1 log odds vector', plot_true=False)
# YLQ
models = cv_logregmodel(YLQ_6x1_df[YLQ_6x1_df['partition'] != 4], 'YLQ 6x1 log odds vector', 4, plot_true=False)
YLQ_6x1_test_auc, _, _ = loo_test(YLQ_6x1_df[YLQ_6x1_df['partition'] == 4], models, 'YLQ 6x1 log odds vector', plot_true=False)
# NLV
models = cv_logregmodel(NLV_6x1_df[NLV_6x1_df['partition'] != 4], 'NLV 6x1 log odds vector', 4, plot_true=False)
NLV_6x1_test_auc, _, _ = loo_test(NLV_6x1_df[NLV_6x1_df['partition'] == 4], models, 'NLV 6x1 log odds vector', plot_true=False)

## 6x20 models ## - also test on three other peptides
# GIL 
models = cv_logregmodel(GIL_6x20_df[GIL_6x20_df['partition'] != 4], 'GIL 6x20 log odds vector', 4, plot_true=False)
GIL_6x20_test_auc, _, _ = loo_test(GIL_6x20_df[GIL_6x20_df['partition'] == 4], models, 'GIL 6x20 log odds vector', plot_true=False)
# test on new peptide
test_auc_6x20_GIL, _, _ = loo_test(GLC_6x20_df.append(YLQ_6x20_df).append(NLV_6x20_df), models, 'fit GIL 9x20, test GLC+YLQ+NLV 9x20', plot_true=False)
# GLC
models = cv_logregmodel(GLC_6x20_df[GLC_6x20_df['partition'] != 4], 'GLC 6x20 log odds vector', 4, plot_true=False)
GLC_6x20_test_auc, _, _ = loo_test(GIL_6x20_df[GIL_6x20_df['partition'] == 4], models, 'GLC 6x20 log odds vector', plot_true=False)
# test on new peptide
test_auc_6x20_GLC, _, _ = loo_test(GIL_6x20_df.append(YLQ_6x20_df).append(NLV_6x20_df), models, 'fit GLC 9x20, test GIL+YLQ+NLV 9x20', plot_true=False)
# YLQ
models = cv_logregmodel(YLQ_6x20_df[YLQ_6x20_df['partition'] != 4], 'YLQ 6x20 log odds vector', 4, plot_true=False)
YLQ_6x20_test_auc, _, _ = loo_test(YLQ_6x20_df[YLQ_6x20_df['partition'] == 4], models, 'YLQ 6x20 log odds vector', plot_true=False)
# test on new peptide
test_auc_6x20_YLQ, _, _ = loo_test(GIL_6x20_df.append(GLC_6x20_df).append(NLV_6x20_df), models, 'fit YLQ 9x20, test GIL+GLC+NLV 9x20', plot_true=False)
# NLV
models = cv_logregmodel(NLV_6x20_df[NLV_6x20_df['partition'] != 4], 'NLV 6x20 log odds vector', 4, plot_true=False)
NLV_6x20_test_auc, _, _ = loo_test(NLV_6x20_df[NLV_6x20_df['partition'] == 4], models, 'NLV 6x20 log odds vector', plot_true=False)
# test on new peptide
test_auc_6x20_NLV, _, _ = loo_test(GIL_6x20_df.append(GLC_6x20_df).append(YLQ_6x20_df), models, 'fit NLV 9x20, test GIL+GLC+YLQ 9x20', plot_true=False)


###### plot logreg models ######

## plot performance for combinations (fit and test on same peptide-type)##
all_peptide_features = {'FA)\n9x1': {'GIL': GIL_9x1_test_auc, 'GLC': GLC_9x1_test_auc, 'YLQ': YLQ_9x1_test_auc, 'NLV': NLV_9x1_test_auc}, 
    'FB)\n9x20': {'GIL': GIL_9x20_test_auc, 'GLC': GLC_9x20_test_auc, 'YLQ': YLQ_9x20_test_auc, 'NLV': NLV_9x20_test_auc}, 
    'FC)\n9x1 + CDR3': {'GIL': GIL_9x1_cdr3_test_auc, 'GLC': GLC_9x1_cdr3_test_auc, 'YLQ': YLQ_9x1_cdr3_test_auc, 'NLV': NLV_9x1_cdr3_test_auc}, 
    'FD)\n9x20 + CDR3': {'GIL': GIL_9x20_cdr3_test_auc, 'GLC': GLC_9x20_cdr3_test_auc, 'YLQ': YLQ_9x20_cdr3_test_auc, 'NLV': NLV_9x20_cdr3_test_auc}, 
    'FE)\n6x1': {'GIL': GIL_6x1_test_auc, 'GLC': GLC_6x1_test_auc, 'YLQ': YLQ_6x1_test_auc, 'NLV': NLV_6x1_test_auc}, 
    'FF)\n6x20': {'GIL': GIL_6x20_test_auc, 'GLC': GLC_6x20_test_auc, 'YLQ': YLQ_6x20_test_auc, 'NLV': NLV_6x20_test_auc}}
all_peptide_features = pd.DataFrame.from_dict(all_peptide_features)

#plot
sns.set_style("whitegrid")
all_peptide_features.T.plot(kind="bar", color=sns.set_palette('colorblind'), figsize=(12,7))
plt.xticks(rotation=0)
plt.ylim((0.40, 0.80))
plt.ylabel('AUC', fontsize=14)
plt.legend(bbox_to_anchor=(0.15, 1), prop={'size': 14})
plt.savefig(Path(RESULTSPATH, 'logreg_variations.pdf'), format='pdf')

## plot performance of models tested on three other peptides ##
new_peps_test = {
    'FB)\n9x20': {'GIL': test_auc_9x20_GIL, 'GLC': test_auc_9x20_GLC, 'YLQ': test_auc_9x20_YLQ, 'NLV': test_auc_9x20_NLV},
    'FD)\n9x20 + CDR3': {'GIL': test_auc_9x20_cdr3_GIL, 'GLC': test_auc_9x20_cdr3_GLC, 'YLQ': test_auc_9x20_cdr3_YLQ, 'NLV': test_auc_9x20_cdr3_NLV},
    'FF)\n6x20': {'GIL': test_auc_6x20_GIL, 'GLC': test_auc_6x20_GLC, 'YLQ': test_auc_6x20_YLQ, 'NLV': test_auc_6x20_NLV}}
new_peps_test = pd.DataFrame.from_dict(new_peps_test)

#plot
sns.set_style("whitegrid")
new_peps_test.T.plot(kind="bar", color=sns.set_palette('colorblind'), width=0.7, figsize=(4,7))
plt.xticks(rotation=0)
plt.ylim((0.40, 0.80))
plt.ylabel('AUC', fontsize=14)
plt.legend(bbox_to_anchor=(1, 1.00), prop={'size': 14})
plt.savefig(Path(RESULTSPATH, 'logreg_variations_unseen_peps.pdf'), format='pdf')

############################## combined ############################## 

###feature tables 
# 9x20
GIL_9x20_train_df = GIL_9x20_df[GIL_9x20_df['partition'] != 4]
GIL_9x20_test_df = GIL_9x20_df[GIL_9x20_df['partition'] == 4]
GLC_9x20_train_df = GLC_9x20_df[GLC_9x20_df['partition'] != 4]
GLC_9x20_test_df = GLC_9x20_df[GLC_9x20_df['partition'] == 4]
#add peptide with seq column
GIL_9x20_train_df['peptide'] = ['GIL']*len(GIL_9x20_train_df)
GLC_9x20_train_df['peptide'] = ['GLC']*len(GLC_9x20_train_df)
#concat GIL and GLC train set 
GIL_GLC_9x20_combined_df = pd.concat([GIL_9x20_train_df,GLC_9x20_train_df])

# 9x20 + cdr3
GIL_9x20_cdr3_train_df = GIL_9x20_cdr3_df[GIL_9x20_cdr3_df['partition'] != 4]
GIL_9x20_cdr3_test_df = GIL_9x20_cdr3_df[GIL_9x20_cdr3_df['partition'] == 4]
GLC_9x20_cdr3_train_df = GLC_9x20_cdr3_df[GLC_9x20_cdr3_df['partition'] != 4]
GLC_9x20_cdr3_test_df = GLC_9x20_cdr3_df[GLC_9x20_cdr3_df['partition'] == 4]
#add peptide with seq column
GIL_9x20_cdr3_train_df['peptide'] = ['GIL']*len(GIL_9x20_cdr3_train_df)
GLC_9x20_cdr3_train_df['peptide'] = ['GLC']*len(GLC_9x20_cdr3_train_df)
#concat GIL and GLC 
GIL_GLC_9x20_cdr3_combined_df = pd.concat([GIL_9x20_cdr3_train_df,GLC_9x20_cdr3_train_df])

# 6x20
GIL_6x20_train_df = GIL_6x20_df[GIL_6x20_df['partition'] != 4]
GIL_6x20_test_df = GIL_6x20_df[GIL_6x20_df['partition'] == 4]
GLC_6x20_train_df = GLC_6x20_df[GLC_6x20_df['partition'] != 4]
GLC_6x20_test_df = GLC_6x20_df[GLC_6x20_df['partition'] == 4]
#add peptide with seq column
GIL_6x20_train_df['peptide'] = ['GIL']*len(GIL_6x20_train_df)
GLC_6x20_train_df['peptide'] = ['GLC']*len(GLC_6x20_train_df)
#concat GIL and GLC train set 
GIL_GLC_6x20_combined_df = pd.concat([GIL_6x20_train_df,GLC_6x20_train_df])


### run models ###
print("Running combined logreg models ...")

#9x20 no undersampling 
models, _ = cv_ds_logregmodel(GIL_GLC_9x20_combined_df, 'Fit logreg model on GIL and GLC partition 0-3', downsampling=False, us_n=None)
GIL_auc,_,_ = loo_test(GIL_9x20_test_df, models, 'Test on GIL partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
GLC_auc,_,_ = loo_test(GLC_9x20_test_df, models, 'Test on GLC partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
YLQ_auc,_,_ = loo_test(YLQ_9x20_df, models, 'Test on YLQ\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)

# 9x20 + cdr3
models_cdr3, _ = cv_ds_logregmodel(GIL_GLC_9x20_cdr3_combined_df, 'Fit logreg model on GIL and GLC with CDR3 partition 0-3', downsampling=False, us_n=None)
GIL_cdr3_auc, _,_ = loo_test(GIL_9x20_cdr3_test_df, models_cdr3, 'Test on GIL partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
GLC_cdr3_auc, _,_ = loo_test(GLC_9x20_cdr3_test_df, models_cdr3, 'Test on GLC partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
YLQ_cdr3_auc, _,_ = loo_test(YLQ_9x20_cdr3_df, models_cdr3, 'Test on YLQ\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)

#6x20
models_6x20, _ = cv_ds_logregmodel(GIL_GLC_6x20_combined_df, 'Fit logreg model on GIL and GLC partition 0-3 (pos3-8 with dim 6x20)', downsampling=False, us_n=None)
GIL_6x20_auc, _,_ = loo_test(GIL_6x20_test_df, models_6x20, 'Test on GIL partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=True)
GLC_6x20_auc,  _,_ = loo_test(GLC_6x20_test_df, models_6x20, 'Test on GLC partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=True)
YLQ_6x20_auc,  _,_ = loo_test(YLQ_6x20_df, models_6x20, 'Test on YLQ\n LogReg fit on GIL and GLC partition 0-3', plot_true=True)

# 9x20 undersampling to 1500
models_us1500, _ = cv_ds_logregmodel(GIL_GLC_9x20_combined_df, 'Fit logreg model on GIL and GLC partition 0-3', downsampling=True, us_n=1500)
GIL_us1500_auc, _,_ = loo_test(GIL_9x20_test_df, models_us1500, 'Test on GIL partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
GLC_us1500_auc, _,_ = loo_test(GLC_9x20_test_df, models_us1500, 'Test on GLC partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
YLQ_us1500_auc, _,_ = loo_test(YLQ_9x20_df, models_us1500, 'Test on YLQ\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)

# 9x20 undersampling to 15001000
models_us1000, _ = cv_ds_logregmodel(GIL_GLC_9x20_combined_df, 'Fit logreg model on GIL and GLC partition 0-3', downsampling=True, us_n=1000)
GIL_us1000_auc, _,_ = loo_test(GIL_9x20_test_df, models_us1000, 'Test on GIL partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
GLC_us1000_auc, _,_ = loo_test(GLC_9x20_test_df, models_us1000, 'Test on GLC partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
YLQ_us1000_auc, _,_ = loo_test(YLQ_9x20_df, models_us1000, 'Test on YLQ\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)

# 9x20 undersampling to 1500 600
models_us600, _ = cv_ds_logregmodel(GIL_GLC_9x20_combined_df, 'Fit logreg model on GIL and GLC partition 0-3', downsampling=True, us_n=600)
GIL_us600_auc, _,_ = loo_test(GIL_9x20_test_df, models_us600, 'Test on GIL partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
GLC_us600_auc, _,_ = loo_test(GLC_9x20_test_df, models_us600, 'Test on GLC partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
YLQ_us600_auc, _,_ = loo_test(YLQ_9x20_df, models_us600, 'Test on YLQ\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)

# 9x20 undersampling to 1500 400
models_us400, _ = cv_ds_logregmodel(GIL_GLC_9x20_combined_df, 'Fit logreg model on GIL and GLC partition 0-3', downsampling=True, us_n=400)
GIL_us400_auc, _,_ = loo_test(GIL_9x20_test_df, models_us400, 'Test on GIL partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
GLC_us400_auc, _,_ = loo_test(GLC_9x20_test_df, models_us400, 'Test on GLC partition 4,\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)
YLQ_us400_auc, _,_ = loo_test(YLQ_9x20_df, models_us400, 'Test on YLQ\n LogReg fit on GIL and GLC partition 0-3', plot_true=False)


### Plot combined logreg models
downsample_features = {'9x20': {'GIL': GIL_auc, 'GLC': GLC_auc, 'YLQ': YLQ_auc}, 
    '9x20\n+ CDR3': {'GIL': GIL_cdr3_auc, 'GLC': GLC_cdr3_auc, 'YLQ': YLQ_cdr3_auc},
    '6x20': {'GIL': GIL_6x20_auc, 'GLC': GLC_6x20_auc, 'YLQ': YLQ_6x20_auc}, 
    '9x20\nDS=1500': {'GIL': GIL_us1500_auc, 'GLC': GLC_us1500_auc, 'YLQ': YLQ_us1500_auc}, 
    '9x20\nDS=1000': {'GIL': GIL_us1000_auc, 'GLC': GLC_us1000_auc, 'YLQ': YLQ_us1000_auc}, 
    '9x20\nDS=600': {'GIL': GIL_us600_auc, 'GLC': GLC_us600_auc, 'YLQ': YLQ_us600_auc}, 
    '9x20\nDS=400': {'GIL': GIL_us400_auc, 'GLC': GLC_us400_auc, 'YLQ': YLQ_us400_auc}}
downsample_features = pd.DataFrame.from_dict(downsample_features)

#plot
plt.style.use('seaborn-v0_8-muted')
ax = downsample_features.T.plot(kind="bar", color=sns.set_palette('colorblind'))

for bars in ax.containers: 
    ax.bar_label(bars, fontsize=8, rotation=90, fmt="%.2f")

plt.xticks(rotation=0)
plt.ylim((0.40, 0.80))
plt.ylabel('AUC')
plt.legend(bbox_to_anchor=(1.0, 1.00))
plt.savefig(Path(RESULTSPATH, 'logreg_combined_DS.pdf'), format='pdf')