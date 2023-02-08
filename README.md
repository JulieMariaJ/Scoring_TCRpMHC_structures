# Master thesis: Structure-based Prediction of TCRpMHC Interaction Using Graph Neural Networks
by: Charlotte Würtzen and Julie Maria Johansen

------------
Prediction of TCRpMHC interaction from TCRpMHCmodels modeled structures utilizing ESM-IF1 predictions for scoring complexes.

The overall aim of this master thesis is to predict interaction between T-cell receptors (TCR) and peptides bound major histocompatibility complexes (pMHC) based on the three dimensional coordinates of the protein structures. To do so, we will investigate the use of the pretrained graph-based Evolutionary Scale Modeling Inverse Folding 1 (ESM-IF1) language model from Facebook Research/Meta Research by [(Hsu et al. 2022)](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v2). 
Due to limited experimentally known TCRpMHC crystal structures, the structures used in this project is modeled using TCRpMHCmodels [(Kjærgaard et al. 2019)](https://doi.org/10.1038/s41598-019-50932-4). Sequence data for a larger upscaled dataset is generated by Mathias Fynbo Jensen [(Jensen 2023)](https://github.com/mnielLab/Master_MFJ/README.md).

------------



Project Organization
------------

    ├── TCRpMHCmodels
    │   ├── README.md
    │   ├── clean_pdb_files.py
    │   └── complex_modeling.py
    │
    ├── scoring_TCRpMHC_structures
    │   ├── README.md
    │   ├── data
    │   │   ├── raw
    │   │   │   ├── fastafiles                      <- fastafiles over TCRpMHC sequences (solved/WT, SW, GGYN data)
    │   │   │   ├── TCRpMHC_structures              <- PDB-files over TCRpMHC structures (solved, WT, SW, GGYN data)
    │   │   │   ├── template_identities             <- CSV-files over template selection identities (WT, SW, GGYN data)
    │   │   │   └── pymol_rmsd_solved_model_wt.csv
    │   │   │
    │   │   └── ESM-IF1_predictions
    │   │       ├── peptide_features                <- tsv-files over ESM-IF1 prediction for peptide (solved, WT, SW, GGYN data)
    │   │       ├── CDR3_features                   <- tsv-files over ESM-IF1 prediction for CDR3 residues (GGYN data)
    │   │       └── background_models               <- tsv-files over average predictions for each group (solved, WT, SW, GGYN data)
    │   │
    │   ├── datapreparation
    │   │   ├── make_dataset.py                     <- script generating ESM-IF1 predictions for peptides 
    │   │   ├── all_cdr3_neighbours.py              <- script generating ESM-IF1 predictions for CDR3 residues in TCRalpha and beta within 4 Å
    │   │   └── make_bg_frequencies.py              <- script generating average ESM-IF1 predictions for peptide across peptide group 
    │   │
    │   ├── prediction_models
    │   │   ├── benchmark_test.py                   <- script for initial test of score function on WT-SW dataset
    │   │   ├── score_function.py                   <- script for score function on GGYN dataset
    │   │   └── log_reg.py                          <- script for logistic regression models on GGYN dataset
    │   │
    │   ├── visualization  
    │   │   ├── feature_distribution.py             <- script for plotting template identities
    │   │   ├── ESM-IF1_bias.py                     <- script for plotting solved/WT scores and heatmap over average solved, WT, SW, GGYN data  
    │   │   └── CDR3_features.py                    <- script for plotting CDR3 bias and scores
    │   │
    │   └── result_figures                          <- plots for report
    │
    └── README.md
    
