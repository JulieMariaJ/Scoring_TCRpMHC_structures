# Master thesis: Structure-based Prediction of TCR-pMHC Interaction Using Graph Neural Networks
Prediction of TCRpMHC interaction from TCRpMHCmodels modeled structures. ESM-IF is utilized for binding prediction.
------------

Guide for running pipeline 
------------
1. Generate structures with TCRpMHCmodels as explained in the TCRpMHCmodels directory README.md file 
2. Run make_dataset.py for ESM-IF1 prediction on peptide residues in complexes 
3. Run all_cdr3_neighbours.py for ESM-IF1 predictions on CDR3 residues (within 4Å) of TCR of complexes
4. Run make_bg_freqs.py for averaging ESM-IF1 predictions for each group (solved, WT, SW, GGYN data)

How to run make data set (example)
------------
    python3 make_dataset.py --input_dir Scoring_TCRpMHC_structures/scoring_TCRpMHC_structures/data/raw/TCRpMHCstructures/solved/ --output_dir Scoring_TCRpMHC_structures/scoring_TCRpMHC_structures/data/ESM-IF1_predictions/peptide_features/solved/ --raw_data None --fasta_dir Scoring_TCRpMHC_structures/scoring_TCRpMHC_structures/data/raw/fastafiles/solved_WT/ --annotation padding --relaxed False
