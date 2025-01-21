import os
import shutil
import json

# Paths
dataset_path = "ThermoScenes"
project_path = "project"

folders_to_create = ["colmap", "dust3r"]
project_path2 = "project/dust3r"
# Define valid image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.png']
# Get scene names from ThermoScenes
scene_names = [name for name in os.listdir(project_path2) if os.path.isdir(os.path.join(dataset_path, name))]

# Loop through each folder (colmap, dust3r) and each scene name
for folder in folders_to_create:
    base_path = os.path.join(project_path, folder)

    for scene_name in scene_names:
        try:
            scene_folder_path = os.path.join(base_path, scene_name)
            print(f"Looping through {scene_folder_path}")

            # scene_name_1 
            for subfolder in os.listdir(scene_folder_path):
                if subfolder != 'ref':
                    subfolder_path = os.path.join(scene_folder_path, subfolder)
                    
                    # n_views
                    for subsubfolder in os.listdir(subfolder_path):
                        subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                        transform_path = os.path.join(subsubfolder_path,'transforms.json')
                        print(transform_path)
                        with open(transform_path, 'r') as file:
                            data = json.load(file)
                        frames = data["frames"]

                        for frame in frames:
                            original_path = frame["file_path"]
                            filename = os.path.basename(original_path)
                            frame["file_path"] = os.path.join("images", filename)
                            
                        
                            # Adjust thermal_file_path based on scene
                            if "buildingA_spring" in scene_name:
                                frame["thermal_file_path"] = os.path.join("thermal", filename.replace(".jpeg", ".jpeg"))
                            elif "exhibition_building" in scene_name:
                                frame["thermal_file_path"] = os.path.join("thermal", filename.replace(".jpg", ".PNG"))
                            elif "freezing_ice_cup" in scene_name:
                                frame["thermal_file_path"] = os.path.join("thermal", filename.replace(".JPG", ".JPG"))
                            elif "heater_water_cup" in scene_name:
                                frame["thermal_file_path"] = os.path.join("thermal", filename.replace(".JPG", ".JPG"))
                            elif "trees" in scene_name:
                                frame["thermal_file_path"] = os.path.join("thermal", filename.replace(".jpg", ".PNG"))
                            elif any(scene in scene_name for scene in ["buildingA_winter", "double_robot", "heater_water_kettle", "melting_ice_cup", "raspberrypi"]):
                                # For other scenes in thermoscenes, images are .JPG and thermal are .PNG
                                frame["thermal_file_path"] = os.path.join("thermal", filename.replace(".JPG", ".PNG"))
                            elif "trees" in scene_name:
                                frame["thermal_file_path"] = os.path.join("thermal", filename.replace(".jpeg", ".jpg"))
                            else:
                                #for scenes not in themoscenes, images are .jpeg and thermal are .PNG
                                frame["thermal_file_path"] = os.path.join("thermal", filename.replace(".jpeg", ".PNG"))
                        # Save the new JSON
                        data["frames"] = frames
                        with open(transform_path, 'w') as file:
                            json.dump(data, file, indent=4)
        except:
            pass
                    
