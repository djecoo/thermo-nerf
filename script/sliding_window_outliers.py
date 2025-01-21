import os
import json
import numpy as np
import quaternionic
import shutil
import argparse

def rotate_quaternion(q, R):
    """
    Rotate a quaternion q by a rotation matrix R.
    """
    R = quaternionic.array.from_rotation_matrix(R).normalized
    q = quaternionic.array(q).normalized
    return (R * q).normalized

def quat_to_homo(q, t):
    """
    Transform a quaternion (q) and translation vector (t) into a homogeneous matrix.
    """
    rotation = q.to_rotation_matrix
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = t
    return T

def ralign(X, Y):
    """
    Perform rigid alignment of two sets of points X and Y.
    Returns rotation matrix (R), scale factor (c), and translation vector (t).
    This function was taken from https://gist.github.com/CarloNicolini/7118015.
    """
    m, n = X.shape
    mx, my = X.mean(1), Y.mean(1)
    Xc, Yc = X - np.tile(mx, (n, 1)).T, Y - np.tile(my, (n, 1)).T
    sx, sy = np.mean(np.sum(Xc * Xc, 0)), np.mean(np.sum(Yc * Yc, 0))
    Sxy = np.dot(Yc, Xc.T) / n

    U, D, V = np.linalg.svd(Sxy, full_matrices=True)
    V = V.T
    S = np.eye(m)

    if np.linalg.matrix_rank(Sxy) > m - 1 and np.linalg.det(Sxy) < 0:
        S[m - 1, m - 1] = -1

    R = np.dot(np.dot(U, S), V.T)
    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)
    return R, c, t

def main(directory):
    # Directories and initialization
    source_directory = directory
    destination_file = os.path.join(directory, "transforms.json")
    transforms_list = []
    ref_list = []

    # Collect paths to transform files and reference frames
    for item in os.listdir(source_directory):
        item_path = os.path.join(source_directory, item)
        if item == "ref":
            ref_list.extend(os.listdir(item_path))
        else:
            transforms_list.extend(
                os.path.join(item_path, subitem, "transforms.json")
                for subitem in os.listdir(item_path)
            )

    # Initialize for merging transformations
    data_final = None
    ref_full = []

    # Process transforms
    for transform_path in transforms_list:
        with open(transform_path, "r") as f:
            data = json.load(f)

        # Initialize the final data structure and process reference frames
        if data_final is None:
            data_final = data
            for frame in data.get("frames", []):
                frame_name = os.path.basename(frame["file_path"])
                if frame_name in ref_list:
                    homo_matrix = np.array(frame["transform_matrix"])
                    pos = homo_matrix[:3, -1]
                    rotation_matrix = homo_matrix[:3, :3]
                    quat_R = quaternionic.array.from_rotation_matrix(rotation_matrix)
                    ref_full.append({"name": frame_name, "quaternion": quat_R, "position": pos})

            # Extract positions and quaternions of references
            pos_ref = np.array([item["position"] for item in ref_full]).T
            quat_ref = np.array([item["quaternion"] for item in ref_full])

        else:
            # Temporary storage for current file's frames
            ref_temp = []
            train_temp = []

            for frame in data.get("frames", []):
                frame_name = os.path.basename(frame["file_path"])
                homo_matrix = np.array(frame["transform_matrix"])
                pos = homo_matrix[:3, -1]
                rotation_matrix = homo_matrix[:3, :3]
                quat_R = quaternionic.array.from_rotation_matrix(rotation_matrix)

                if frame_name in ref_list:
                    ref_temp.append({"name": frame_name, "quaternion": quat_R, "position": pos})
                elif "train" in frame["file_path"]:
                    train_temp.append({"name": frame_name, "quaternion": quat_R, "position": pos})

            # Match references to the global order
            ref_temp_dict = {item["name"]: item for item in ref_temp}
            ref_temp_ordered = [ref_temp_dict[item["name"]] for item in ref_full]

            # Align positions
            pos_ref_temp = np.array([item["position"] for item in ref_temp_ordered]).T
            pos_train_temp = np.array([item["position"] for item in train_temp]).T

            R, c, t = ralign(pos_ref_temp, pos_ref)
            residuals = np.linalg.norm(c * np.dot(R, pos_ref_temp) + t[:, np.newaxis] - pos_ref, axis=0)
            worst_indices = np.argsort(residuals)[-2:]

            # Filter out outliers and re-align
            pos_ref_temp_filtered = np.delete(pos_ref_temp, worst_indices, axis=1)
            pos_ref_filtered = np.delete(pos_ref, worst_indices, axis=1)
            R, c, t = ralign(pos_ref_temp_filtered, pos_ref_filtered)

            # Rotate training positions and quaternions
            pos_train_rotated = (c * np.dot(R, pos_train_temp) + t[:, np.newaxis]).T
            quat_train_rotated = [rotate_quaternion(q, R) for q in np.array([item["quaternion"] for item in train_temp])]

            # Update data frames
            for quat, pos, frame in zip(quat_train_rotated, pos_train_rotated, train_temp):
                frame["transform_matrix"] = quat_to_homo(quat, pos).tolist()
                data_final["frames"].append(frame)

    # Save the final transforms file
    with open("transforms.json", "w") as f:
        json.dump(data_final, f, indent=4)

    # Copy the result to the destination
    shutil.copy("transforms.json", destination_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process transform files.")
    parser.add_argument("--directory", type=str, required=True, help="Source directory containing transform files") 
    args = parser.parse_args()
    main(args.directory)
