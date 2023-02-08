# Master thesis: Structure-based Prediction of TCR-pMHC Interaction Using Graph Neural Networks
Prediction of TCRpMHC interaction from TCRpMHCmodels modeled structures. ESM-IF is utilized for binding prediction.

How to run make data set (example)
------------
    python3 make_dataset.py --input_dir Scoring_TCRpMHC_structures/scoring_TCRpMHC_structures/data/raw/TCRpMHCstructures/solved/ --output_dir Scoring_TCRpMHC_structures/scoring_TCRpMHC_structures/data/ESM-IF1_predictions/peptide_features/solved/ --raw_data None --fasta_dir Scoring_TCRpMHC_structures/scoring_TCRpMHC_structures/data/raw/fastafiles/solved_WT/ --annotation padding --relaxed False
