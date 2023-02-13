Guide to setup ESM-IF1 environment
------------
    conda create -n inverse_env
    conda activate inverse_env
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    conda install pyg -c pyg -c conda-forge
    pip install biotite
    pip install git+https://github.com/facebookresearch/esm.git

Additional packages:

    pip install matplotlib seaborn biopython python-dotenv
    pip uninstall scikit-learn
    pip install scikit-learn==1.1.0
    pip install umap-learn
    pip install py3Dmol
    pip install ProDy
    pip install pathlib
    pip install python-dotenv
    pip install statannot
    pip install bio


How to run make_dataset.py (example)
------------
    python3 make_dataset.py --input_dir Scoring_TCRpMHC_structures/scoring_TCRpMHC_structures/data/raw/TCRpMHCstructures/solved/ --output_dir Scoring_TCRpMHC_structures/scoring_TCRpMHC_structures/data/ESM-IF1_predictions/peptide_features/solved/ --raw_data None --fasta_dir Scoring_TCRpMHC_structures/scoring_TCRpMHC_structures/data/raw/fastafiles/solved_WT/ --annotation padding --relaxed False

How to run all_cdr3_neighbours.py (example)
------------
    python3 all_cdr3_neighbours.py --pdb_dir Scoring_TCRpMHC_structures/scoring_TCRpMHC_structures/data/raw/TCRpMHCstructures/solved/ --output_dir Scoring_TCRpMHC_structures/scoring_TCRpMHC_structures/data/ESM-IF1_predictions/CDR3_features/solved/ 
