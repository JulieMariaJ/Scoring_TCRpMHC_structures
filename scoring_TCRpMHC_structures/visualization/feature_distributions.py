## PACKAGES ##

import glob
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import find_dotenv
from statannot import add_stat_annotation


## PATHS ##

dotenv_path = find_dotenv()
ROOTPATH = Path(find_dotenv()).parent
RAWPATH = Path(ROOTPATH, 'data/raw')
ESMPATH = Path(ROOTPATH, 'data/ESM-IF1_predictions')
RESULTSPATH = Path(ROOTPATH, 'result_figures')

## FUNCTIONS ##

def collect_top_temp_info(template_files):
    """ Collects identities for top templates """
    top_templates = {}
    for file in template_files:
        id_name = file.split("/")[-1].split("-")[0]   #.strip("-complex-template.csv")
        dataframe = pd.read_csv(file, header=0)
        # add only top template
        top_templates[id_name] = [dataframe["pdb_id"][0], dataframe["pep_identity"][0], dataframe["mhc_identity"][0], dataframe["tcrA_identity"][0], dataframe["tcrB_identity"][0], dataframe["total_identity"][0]]

    return top_templates

def convert_top_temp_to_df(template_dict):
    """ Converts dict of top temp info ti dataframe """
    templates_df = pd.DataFrame(template_dict, index=['top_temp_id', 'pep_identity', 
                                                    'MHC_identity', 'tcrA_identity', 
                                                    'tcrB_identity', 'total_identity']).T

    return templates_df


def plot_total_iden(identity_df, xname='', yname='', top_temp=True, filename=''):
    fig = plt.subplots(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.set_palette("rocket_r")

    box = sns.boxplot(x=identity_df[xname], 
                        y=identity_df[yname], 
                        hue=identity_df['binder'], 
                        hue_order=[1,0],
                        showmeans=True)

    add_stat_annotation(box, data=identity_df, x=xname, y=yname, hue='binder',
                        box_pairs=[(("GIL", 0), ("GIL", 1)),
                                    (("GLC", 0), ("GLC", 1)),
                                    (("YLQ", 0), ("YLQ", 1)),
                                    (("NLV", 0), ("NLV", 1))],
                        test='t-test_ind', text_format='star', loc='inside', verbose=1, comparisons_correction=None)

    plt.title('Total identity', fontsize=14)
    plt.ylabel("Total identity (%)")
    if top_temp == False:
        plt.ylabel("Avg. total identity (%)")
    plt.xlabel("")

    plt.savefig(filename, format="pdf", dpi=1200)


def plot_pep_MHC_tcrA_tcrB_iden(identity_df, identity_groups=[], titles=[], top_temp=True, filename=''):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_style("whitegrid")

    count = 0
    for i in range(2):
        for j in range(2):
            box = sns.boxplot(x=identity_df["peptide"], 
                        y=identity_df[identity_groups[count]], 
                        hue=identity_df['binder'],
                        hue_order=[1,0],
                        showmeans=True, 
                        ax=axes[i,j])

            add_stat_annotation(box, data=identity_df, 
                                        x='peptide', 
                                        y=identity_df[identity_groups[count]], 
                                        hue='binder',
                                        box_pairs=[(("GIL", 0), ("GIL", 1)),
                                                    (("GLC", 0), ("GLC", 1)),
                                                    (("YLQ", 0), ("YLQ", 1)),
                                                    (("NLV", 0), ("NLV", 1))],
                        test='t-test_ind', text_format='star', loc='inside', verbose=1, comparisons_correction=None)

            axes[i,j].set(ylabel = "Sequence identity (%)", xlabel = "")
            if top_temp == False:
                axes[i,j].set(ylabel = "Avg. sequence identity (%)", xlabel = "")
            axes[i,j].set_title(titles[count])
            axes[i,j].legend(loc='lower right')
            count += 1

    plt.savefig(filename, format="pdf", dpi=1200)


################################ Collect features ################################

print("Collecting features ...")

## get features and template csvs      
GIL_templates = glob.glob(str(Path(RAWPATH, 'template_identities/GIL')) + '/*complex-templates.csv')
GIL_df = pd.read_csv(Path(ESMPATH, 'peptide_features/GIL/features_padding.tsv'), sep='\t', index_col=0)

GLC_templates = glob.glob(str(Path(RAWPATH, 'template_identities/GLC')) + '/*complex-templates.csv')
GLC_df = pd.read_csv(Path(ESMPATH, 'peptide_features/GLC/features_padding.tsv'), sep='\t', index_col=0)

YLQ_templates = glob.glob(str(Path(RAWPATH, 'template_identities/YLQ')) + '/*complex-templates.csv')
YLQ_df = pd.read_csv(Path(ESMPATH, 'peptide_features/YLQ/features_padding.tsv'), sep='\t', index_col=0)

NLV_templates = glob.glob(str(Path(RAWPATH, 'template_identities/NLV')) + '/*complex-templates.csv')
NLV_df = pd.read_csv(Path(ESMPATH, 'peptide_features/NLV/features_padding.tsv'), sep='\t', index_col=0)

# collect top temp info
GIL_top_templates = collect_top_temp_info(GIL_templates)
GLC_top_templates = collect_top_temp_info(GLC_templates)
YLQ_top_templates = collect_top_temp_info(YLQ_templates)
NLV_top_templates = collect_top_temp_info(NLV_templates)

# convert to dataframe
GIL_top_temp_df = convert_top_temp_to_df(GIL_top_templates)
GLC_top_temp_df = convert_top_temp_to_df(GLC_top_templates)
YLQ_top_temp_df = convert_top_temp_to_df(YLQ_top_templates)
NLV_top_temp_df = convert_top_temp_to_df(NLV_top_templates)

## add binder information
for top_temp_df, feat_df in zip([GIL_top_temp_df, GLC_top_temp_df, YLQ_top_temp_df, NLV_top_temp_df],[GIL_df, GLC_df, YLQ_df, NLV_df]):
    top_temp_df['binder'] = feat_df['binder']

## collect to one dataframe
GIL_top_temp_df['peptide'], GLC_top_temp_df['peptide'], YLQ_top_temp_df['peptide'], NLV_top_temp_df['peptide'] = 'GIL', 'GLC', 'YLQ', 'NLV'
all_peptides_top_iden = pd.concat([GIL_top_temp_df, GLC_top_temp_df, YLQ_top_temp_df, NLV_top_temp_df])
#print(all_peptides_top_iden)

### collect features for avg identities
GIL_df['peptide'], GLC_df['peptide'], YLQ_df['peptide'], NLV_df['peptide'] = 'GIL', 'GLC', 'YLQ', 'NLV'
all_peptides_avg_iden = pd.concat([GIL_df, GLC_df, YLQ_df, NLV_df])


################################ Visualizations ################################

print("Doing visualizations ...")

### Plot 1: Distribution of total identity scores (Top template)
sns.set_style("whitegrid")
plot_total_iden(all_peptides_top_iden, 
                xname='peptide', 
                yname='total_identity', 
                top_temp=True, 
                filename=Path(RESULTSPATH, 'top_total_iden.pdf'))

### Plot 2: Distribution of chain identities (Top template)
plot_pep_MHC_tcrA_tcrB_iden(all_peptides_top_iden,
                            ['pep_identity', 'MHC_identity', 'tcrA_identity','tcrB_identity'], 
                            ["Peptide", "MHC", "TCRα", "TCRβ"],
                            top_temp=True,
                            filename=Path(RESULTSPATH, 'top_chain_iden.pdf'))

### Plot 3: Distribution of total identity scores (Avg. template)
plot_total_iden(all_peptides_avg_iden, 
                xname='peptide', 
                yname='total_iden_avg', 
                top_temp=False,
                filename=Path(RESULTSPATH, 'avg_total_iden.pdf'))

### Plot 4: Distribution of chain identities (Avg template)
plot_pep_MHC_tcrA_tcrB_iden(all_peptides_avg_iden,
                            ['pep_iden_avg', 'mhc_iden_avg', 'tcrA_iden_avg', 'tcrB_iden_avg'], 
                            ["Peptide", "MHC", "TCRα", "TCRβ"],
                            top_temp=False,
                            filename=Path(RESULTSPATH, 'avg_chain_iden.pdf'))

