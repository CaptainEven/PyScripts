#!/bin/bash

LD_LIBRARY_PATH=/opt/Qt5.12.0/5.12.0/gcc_64/lib/
GPU_IDX=$1
MAX_IMG_SIZE=2000  # $2
ROOT=/mnt/data2/lyw/ETH3D/multi_view_training_dslr_undistorted  # $3

# indoor: ncc_sigma0.6, outdoor: nccsigma0.7, indoor: filter false, outdoor: filter true
Outdoor=("courtyard" "electro" "facade" "meadow" "playground" "terrace")
Indoor=("delivery_area" "kicker" "office" "pipes" "relief" "relief_2" "terrains")

for DATASET in $ROOT/*
do
    if [ -d "$DATASET" ]
    then
        echo
        echo "=> Processing directory $DATASET..."
        # ---------------------------------------------
        
        if [ ! -d "$DATASET/dense" ]
        then
            mkdir $DATASET/dense
        else
            rm -rf $DATASET/dense
            mkdir $DATASET/dense
        fi
        
        colmap image_undistorter \
        --image_path $DATASET/images \
        --input_path $DATASET/dslr_calibration_undistorted \
        --output_path $DATASET/dense \
        --output_type COLMAP \
        --max_image_size $MAX_IMG_SIZE
        
        if [[ "${Indoor[@]}" =~ "$(basename $DATASET)" ]]
        then
            echo "=> $(basename $DATASET) is indoor scene."
            
            colmap patch_match_stereo \
            --workspace_path $DATASET/dense \
            --workspace_format COLMAP \
            --PatchMatchStereo.geom_consistency true \
            --PatchMatchStereo.gpu_index $GPU_IDX \
            --PatchMatchStereo.ncc_sigma 0.6 \
            --PatchMatchStereo.window_radius 11 \
            --PatchMatchStereo.max_image_size $MAX_IMG_SIZE \
            --PatchMatchStereo.pyramid_stereo_match 0 \
            --PatchMatchStereo.pyramid_stereo_match_l 0 \
            --PatchMatchStereo.smoothness 1 \
            --PatchMatchStereo.filter 0
        elif [[ "${Outdoor[@]}" =~ "$(basename $DATASET)" ]]
        then
            echo "=> $(basename $DATASET) is outdoor scene."
            
            colmap patch_match_stereo \
            --workspace_path $DATASET/dense \
            --workspace_format COLMAP \
            --PatchMatchStereo.geom_consistency true \
            --PatchMatchStereo.gpu_index $GPU_IDX \
            --PatchMatchStereo.ncc_sigma 0.7 \
            --PatchMatchStereo.window_radius 11 \
            --PatchMatchStereo.max_image_size $MAX_IMG_SIZE \
            --PatchMatchStereo.pyramid_stereo_match 0 \
            --PatchMatchStereo.pyramid_stereo_match_l 0 \
            --PatchMatchStereo.smoothness 1 \
            --PatchMatchStereo.filter 1
        else
            echo "Err: $DATASET not included."
        fi
        
        colmap stereo_fusion \
        --workspace_path $DATASET/dense \
        --workspace_format COLMAP \
        --input_type geometric \
        --StereoFusion.max_image_size $MAX_IMG_SIZE \
        --output_path $DATASET/dense/colmap_fused.ply
        
        echo "$DATASET processed."
    fi
done
