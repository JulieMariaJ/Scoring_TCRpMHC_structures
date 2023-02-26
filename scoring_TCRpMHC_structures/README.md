# Master thesis: Structure-based Prediction of TCR-pMHC Interaction Using Graph Neural Networks
Prediction of TCRpMHC interaction from TCRpMHCmodels modeled structures. ESM-IF is utilized for binding prediction.

------------

Guide for running pipeline 
------------
- Model structures 
    + Generate structures with TCRpMHCmodels as explained in the TCRpMHCmodels directory README.md file 
- Datapreparation
    + Run make_dataset.py for ESM-IF1 prediction on peptide residues in complexes 
    + Run all_cdr3_neighbours.py for ESM-IF1 predictions on CDR3 residues (within 4Å) of TCR of complexes
    + Run make_bg_freqs.py for averaging ESM-IF1 predictions for each group (solved, WT, SW, GGYN data) 
- Prediction models
    + benchmark 
    + score function 
    + logreg 
- Visualization
    + scripts for illustrating data 

Data found in data folder \
Result figures can be seen in result_figures folder 
