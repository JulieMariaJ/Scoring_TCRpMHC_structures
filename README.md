# GitHub repository under construction

## Master thesis: Structure-based Prediction of TCRpMHC Interaction Using Graph Neural Networks
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
    │   │   ├── raw_data
    │   │   │   ├── fastafiles              <- fastafiles over TCRpMHC sequences (solved/WT, SW, GGYN data)
    │   │   │   ├── TCRpMHC_structures      <- PDB-files over TCRpMHC structures (solved, WT, SW, GGYN data)
    │   │   │   └── template_identities     <- CSV-files over template selection identities (WT, SW, GGYN data)
    │   │   │
    │   │   └── ESM-IF1_predictions
    │   │       ├── peptide_features        <- tsv-files over ESM-IF1 prediction for peptide (solved, WT, SW, GGYN data)
    │   │       ├── CDR3_features           <- tsv-files over ESM-IF1 prediction for CDR3 residues (GGYN data)
    │   │       └── background_models       <- tsv-files over average predictions for each group (solved, WT, SW, GGYN data)
    │   │
    │   ├── datapreparation
    │   │   ├── make_dataset.py             <- scripts generating ESM-IF1 predictions for peptide 
    │   │   ├── all_cdr3_neighbours.py      <- scripts generating ESM-IF1 predictions for CDR3 residues in TCRalpha and beta within 4 Å
    │   │   └── make_bg_frequencies.py      <- script 
    │   │
    │   ├── prediction_models
    │   └── visualization
    │
    └── README.md
    
