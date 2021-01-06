#!/bin/bash

for DIR in $(ls $1)
do
    if [ -d "$DIR" ]
    then
        echo "Processing $DIR..."
        
        echo "=> evaluating colmap..."
        ETH3DMultiViewEvaluation --reconstruction_ply_path "$DIR/src_fused.ply" \
        --ground_truth_mlp_path "$DIR/dslr_scan_eval/scan_alignment.mlp" \
        --tolerances 0.01,0.02,0.05,0.1,0.2,0.5
        
        echo "=> evaluating SelJBPF..."
        ETH3DMultiViewEvaluation --reconstruction_ply_path "$DIR/fusedSelJBPF.ply" \
        --ground_truth_mlp_path "$DIR/dslr_scan_eval/scan_alignment.mlp" \
        --tolerances 0.01,0.02,0.05,0.1,0.2,0.5
        
        echo "=> evaluating NAF..."
        ETH3DMultiViewEvaluation --reconstruction_ply_path "$DIR/fusedNAF.ply" \
        --ground_truth_mlp_path "$DIR/dslr_scan_eval/scan_alignment.mlp" \
        --tolerances 0.01,0.02,0.05,0.1,0.2,0.5

        echo "$DIR processed."
        echo
    fi
done
