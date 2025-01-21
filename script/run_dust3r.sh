#!/bin/bash

GPU_ID=0
DATA_ROOT_DIR="Add the path do the dust3r folder here, probably project/dust3r"  # Path to the dust3r folder
MODEL_PATH="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" #path to the mast3r weight
SPLAT_PATH="path to the instantsplat folder"
# Loop through each scene in the dust3r folder
for SCENE_DIR in ${DATA_ROOT_DIR}/*; do
    if [ -d "$SCENE_DIR" ] && [ "$(basename "$SCENE_DIR")" != "ref" ]; then
        SCENE_NAME=$(basename "$SCENE_DIR")

        echo "Processing scene: $SCENE_NAME"

        # Loop through all subdirectories inside the scene folder
        for SUBDIR in ${SCENE_DIR}/*; do
            if [ -d "$SUBDIR" ]; then
                # Check if the subdirectory is not 'ref'
                if [ "$(basename "$SUBDIR")" != "ref" ]; then

                    # Now loop through the subdirectory to find nested folders
                    for NESTED_DIR in ${SUBDIR}/*; do
                        if [ -d "$NESTED_DIR" ]; then
                            NESTED_DIR_NAME=$(basename "$NESTED_DIR")
                            
                            # Extract the number from folder names like '30_views'
                            if [[ "$NESTED_DIR_NAME" =~ ([0-9]+)_views ]]; then
                                NUMBER_OF_VIEWS="${BASH_REMATCH[1]}"
                                
                                # Print the full path and the number of views
                                echo "Full path: $NESTED_DIR"
                                echo "Number of views: $NUMBER_OF_VIEWS"

                                # Check if the 'sparse' folder already exists
                                if [ ! -d "${NESTED_DIR}/sparse" ]; then
                                    # Run the command if 'sparse' does not exist
                                    CMD_D1="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ${SPLAT_PATH} \
                                    -s ${NESTED_DIR} \
                                    -m ${NESTED_DIR} \
                                    --n_views ${NUMBER_OF_VIEWS} \
                                    --focal_avg \
                                    --co_vis_dsp \
                                    --conf_aware_ranking \
                                    --infer_video \
                                    --ckpt_path ${MODEL_PATH}"
                                    eval $CMD_D1
                                    # Find a folder with "sparse" in its name
                                    for SPARSE_DIR in ${NESTED_DIR}/*; do
                                        if [[ "$(basename "$SPARSE_DIR")" == *sparse* ]]; then
                                            echo "Renaming $SPARSE_DIR to ${NESTED_DIR}/sparse"
                                            mv "$SPARSE_DIR" "${NESTED_DIR}/sparse"

                                            # Delete the folder named "1" inside the renamed "sparse" directory
                                            if [ -d "${NESTED_DIR}/sparse/1" ]; then
                                                echo "Deleting folder: ${NESTED_DIR}/sparse/1"
                                                rm -rf "${NESTED_DIR}/sparse/1"
                                            fi
                                        fi
                                    done
                                else
                                    echo "Sparse folder already exists, skipping command."
                                fi
                                
                                
                            fi
                        fi
                    done
                fi
            fi
        done
    fi
done
