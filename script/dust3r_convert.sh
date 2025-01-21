#!/bin/bash

GPU_ID=0
DATA_ROOT_DIR="path to dust3r, probably project/dust3r"  # Path to the dust3r folder

# Loop through each scene in the dust3r folder
for SCENE_DIR in ${DATA_ROOT_DIR}/*; do
    if [ -d "$SCENE_DIR" ] && [ "$(basename "$SCENE_DIR")" != "ref" ]; then
        SCENE_NAME=$(basename "$SCENE_DIR")

        echo "Processing scene: $SCENE_NAME"

        # Loop through all subdirectories inside the scene folder
        for SUBDIR in ${SCENE_DIR}/*; do
            if [ -d "$SUBDIR" ]; then
                
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
                            PLY_PATH="${NESTED_DIR}/sparse/0/points3D.ply"
                            TXT_PATH="${NESTED_DIR}/sparse/0/points3D.txt"
                            # Construct and execute the command
                            CMD_D1="python script/ply_to_txt.py --ply_file ${PLY_PATH} --txt_file ${TXT_PATH}"
                            echo "Running command: $CMD_D1"
                            eval $CMD_D1
                        fi
                    fi
                done
            fi
        done
    fi
done
