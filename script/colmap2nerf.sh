#!/bin/bash

GPU_ID=0
DATA_ROOT_DIR="path to your dust3r project, probably project/dust3r"  # Path to the dust3r folder, pro

for SCENE_DIR in ${DATA_ROOT_DIR}/*; do
    if [ -d "$SCENE_DIR" ] && [ "$(basename "$SCENE_DIR")" != "ref" ]; then
        SCENE_NAME=$(basename "$SCENE_DIR")

        echo "Processing scene: $SCENE_NAME"

        # Loop through all subdirectories inside the scene folder
        for SUBDIR in ${SCENE_DIR}/*; do
            if [ -d "$SUBDIR" ]; then
                
                # Loop through the subdirectory to find nested folders
                for NESTED_DIR in ${SUBDIR}/*; do
                    if [ -d "$NESTED_DIR" ]; then
                        NESTED_DIR_NAME=$(basename "$NESTED_DIR")
                        
                        # Extract the number from folder names like '30_views'
                        if [[ "$NESTED_DIR_NAME" =~ ([0-9]+)_views ]]; then
                            NUMBER_OF_VIEWS="${BASH_REMATCH[1]}"
                            
                            # Print the full path and the number of views
                            echo "Full path: $NESTED_DIR"
                            echo "Number of views: $NUMBER_OF_VIEWS"

                            IMG_PATH="${NESTED_DIR}/images"
                            SPARSE_PATH="${NESTED_DIR}/sparse/0/"
                            TXT_PATH="${SPARSE_PATH}points3D.txt"
                            TRANSFORM_JSON="${NESTED_DIR}/transforms.json"

                            # Check if transform.json exists
                            if [ ! -f "$TRANSFORM_JSON" ]; then
                                echo "transform.json not found. Processing..."
                                
                                # Construct and execute the command
                                CMD_D1="ns-process-data images --data ${IMG_PATH} --output-dir ${NESTED_DIR} --skip-colmap --colmap-model-path ${SPARSE_PATH} --skip-image-processing"
                                echo "Running command: $CMD_D1"
                                eval $CMD_D1
                            else
                                echo "transform.json already exists in $NESTED_DIR. Skipping..."
                            fi
                        fi
                    fi
                done
            fi
        done
    fi
done
