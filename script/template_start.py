import os
import random
import shutil

# Paths
dataset_path = "ThermoScenes"  # Path to the dataset containing scenes
project_path = "project"  # Path where the project structure will be created
folders_to_create = ["colmap","dust3r"]  # List of folder names to create under the project path

# Create the project folder if it doesn't already exist
os.makedirs(project_path, exist_ok=True)

# Get the names of all directories (scenes) in the dataset path
scene_names = [
    name for name in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, name))
]

# Loop through the folders to create, e.g., "dust3r"
for folder in folders_to_create:
    # Create the base folder (e.g., project/dust3r)
    base_path = os.path.join(project_path, folder)
    os.makedirs(base_path, exist_ok=True)

    # Loop through each scene in the dataset
    for scene_name in scene_names:
        # Path to the images folder for the current scene
        scene_path = os.path.join(dataset_path, scene_name, "images")

        # Get the list of train and evaluation images (case-insensitive file name check)
        train_images = [
            img for img in os.listdir(scene_path)
            if img.lower().startswith("frame_train")
        ]
        test_images = [
            img for img in os.listdir(scene_path)
            if img.lower().startswith("frame_eval")
        ]

        # Print the number of evaluation images found
        print(len(test_images))

        # Randomly select up to 10 train images for the reference folder
        selected_images = random.sample(train_images, min(10, len(train_images)))

        # Randomly select up to 5 evaluation images
        selected_test = random.sample(test_images, min(len(test_images), 5))

        # Create a ref folder for the scene inside the current folder (e.g., dust3r)
        ref_folder_path = os.path.join(base_path, scene_name, "ref")
        os.makedirs(ref_folder_path, exist_ok=True)

        # Copy the selected reference images to the ref folder
        for img in selected_images:
            img_path = os.path.join(scene_path, img)
            dest_img_path = os.path.join(ref_folder_path, img)
            shutil.copy(img_path, dest_img_path)

        # Remove the reference images from the train images list
        remaining_images = [img for img in train_images if img not in selected_images]

        # Group the remaining images into batches of up to 10 per group
        group_number = 1
        while remaining_images:
            # Shuffle the remaining images for random grouping
            random.shuffle(remaining_images)

            # Create a group folder (e.g., sceneA_1)
            group_folder_name = f"{scene_name}_{group_number}"
            group_folder_path = os.path.join(base_path, scene_name, group_folder_name)
            os.makedirs(group_folder_path, exist_ok=True)

            # Define the number of views for this group (up to 15 images)
            num_views = min(15, len(remaining_images))
            views_folder_name = f"{num_views + len(selected_images) + len(selected_test)}_views"
            views_folder_path = os.path.join(group_folder_path, views_folder_name)
            os.makedirs(views_folder_path, exist_ok=True)

            # Create an images folder within the views folder
            images_folder_path = os.path.join(views_folder_path, "images")
            os.makedirs(images_folder_path, exist_ok=True)

            # Copy the remaining images into the group folder
            for img in remaining_images[:num_views]:
                img_path = os.path.join(scene_path, img)
                dest_img_path = os.path.join(images_folder_path, img)
                shutil.copy(img_path, dest_img_path)

            # Copy the selected reference images to the group folder
            for img in selected_images:
                img_path = os.path.join(scene_path, img)
                dest_img_path = os.path.join(images_folder_path, img)
                shutil.copy(img_path, dest_img_path)

            # Copy the selected evaluation images to the group folder
            for img in selected_test:
                img_path = os.path.join(scene_path, img)
                dest_img_path = os.path.join(images_folder_path, img)
                shutil.copy(img_path, dest_img_path)

            # Remove the copied images from the remaining images list
            remaining_images = remaining_images[num_views:]

            # Increment the group number for the next batch
            group_number += 1

print("Images have been copied and grouped, and ref folders are populated.")


project_path2 = "project/colmap"
# Define valid image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.png']

# Get scene names from ThermoScenes
scene_names = [name for name in os.listdir(project_path2) if os.path.isdir(os.path.join(dataset_path, name))]

# Loop through each folder (colmap, dust3r) and each scene name
for folder in folders_to_create:
    base_path = os.path.join(project_path, folder)

    for scene_name in scene_names:
        scene_folder_path = os.path.join(base_path, scene_name)
        print(f"Looping through {scene_folder_path}")

        # scene_name_1 
        for subfolder in os.listdir(scene_folder_path):
            if subfolder != 'ref':
                subfolder_path = os.path.join(scene_folder_path, subfolder)
                
                # n_views
                for subsubfolder in os.listdir(subfolder_path):
                    subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                    images_path = os.path.join(subsubfolder_path, 'images')
                    # Check if the 'thermal' directory exists, if not create it
                    thermal_path = os.path.join(subsubfolder_path, 'thermal')
                    if not os.path.exists(thermal_path):
                        os.makedirs(thermal_path)
                        print(f"Created 'thermal' directory at: {thermal_path}")

                    # Loop through image files in 'images' directory
                    for file_name in os.listdir(images_path):
                        image_path = os.path.join(images_path, file_name)

                        # Check if the file is an image
                        if any(file_name.lower().endswith(ext) for ext in image_extensions):
                            # Image name without extension
                            image_name_no_ext = os.path.splitext(file_name)[0]

                            # Find the equivalent image in ThermoScenes/scene_name/thermal
                            thermal_folder_path = os.path.join(dataset_path, scene_name, "thermal")

                            # Search for a matching image in the thermal folder
                            for thermal_image in os.listdir(thermal_folder_path):
                                if image_name_no_ext in thermal_image:
                                    thermal_image_path = os.path.join(thermal_folder_path, thermal_image)
                                    # Copy the matching thermal image to the 'thermal' directory
                                    destination_path = os.path.join(thermal_path, thermal_image)
                                    shutil.copy2(thermal_image_path, destination_path)
            print(f"Copied matching thermal image for {subfolder}")
