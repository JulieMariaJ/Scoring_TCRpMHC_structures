# GitHub repository under construction

## Master thesis: Structure-based Prediction of TCRpMHC Interaction Using Graph Neural Networks
by: Charlotte Würtzen and Julie Maria Johansen
------------
Prediction of TCRpMHC interaction from TCRpMHCmodels modeled structures. ESM-IF is utilized for binding prediction.




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
    
