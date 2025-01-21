import os
import json
import numpy as np
import quaternionic
import argparse

def rotate_quaternion(q, R):
    q1 = q
    R = quaternionic.array.from_rotation_matrix(R)
    q = quaternionic.array(q1)

    R = R.normalized
    q = q.normalized

    rotated = R * q
    rotated = rotated.normalized

    return rotated

def quat_to_homo(q, t):
    """Transform q (a quaternionic array [q0, q1, q2, q3]) and t ([x, y, z]) to
    a homogeneous matrix.
    Returns the homogeneous matrix as a list of lists.
    """
    rotation = q.to_rotation_matrix
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = t
    return T

def ralign(X, Y):
    """Align two point sets X and Y using the Umeyama algorithm.
    This function was taken from https://gist.github.com/CarloNicolini/7118015.
    """
    m, n = X.shape
    mx = X.mean(1)
    my = Y.mean(1)
    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T
    sx = np.mean(np.sum(Xc * Xc, 0))
    sy = np.mean(np.sum(Yc * Yc, 0))
    Sxy = np.dot(Yc, Xc.T) / n
    U, D, V = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = V.T.copy()

    r = np.ndim(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if np.linalg.det(Sxy) < 0:
            S[m - 1, m - 1] = -1
        elif r == m - 1 and np.linalg.det(U) * np.linalg.det(V) < 0:
            S[m - 1, m - 1] = -1
    R = np.dot(np.dot(U, S), V.T)
    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)
    return R, c, t

def main(directory):
    transforms_list = []
    ref_list = []

    # Iterate through scenes in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if filename != "ref":
            # Iterate through subfolders for n_views
            for subfilename in os.listdir(filepath):
                subfilepath = os.path.join(filepath, subfilename)
                transform_path = os.path.join(subfilepath, "transforms.json")
                transforms_list.append(transform_path)
        else:
            for subfilename in os.listdir(filepath):
                ref_list.append(subfilename)

    data_final = None
    ref_full = []  # Frames that will be the references

    # Process each transform file
    for transform in transforms_list:
        with open(transform, "r") as f:
            data = json.load(f)

        if data_final is None:
            data_final = data
            for frame in data.get("frames", []):
                frame_name = frame["file_path"].split("/")[-1]
                if frame_name in ref_list:
                    homo_matrix = np.array(frame["transform_matrix"])
                    pos = np.array(homo_matrix[:3, -1])

                    rotation_matrix = np.array(homo_matrix[:3, :3])
                    quat_R = quaternionic.array.from_rotation_matrix(rotation_matrix)
                    ref_full.append({"name": frame_name, "quaternion": quat_R, 'position': pos})

            pos_ref = np.array([item["position"] for item in ref_full]).T
            quat_ref = np.array([item["quaternion"] for item in ref_full])
        else:
            ref_full_temporary = []
            train_full_temporary = []

            for frame in data.get("frames", []):
                frame_name = frame["file_path"].split("/")[-1]
                if frame_name in ref_list:
                    homo_matrix = np.array(frame["transform_matrix"])
                    pos = np.array(homo_matrix[:3, -1])

                    rotation_matrix = np.array(homo_matrix[:3, :3])
                    quat_R = quaternionic.array.from_rotation_matrix(rotation_matrix)
                    ref_full_temporary.append({"name": frame_name, "quaternion": quat_R, 'position': pos})
                elif "train" in frame["file_path"]:
                    homo_matrix = np.array(frame["transform_matrix"])
                    pos = np.array(homo_matrix[:3, -1])

                    rotation_matrix = np.array(homo_matrix[:3, :3])
                    quat_R = quaternionic.array.from_rotation_matrix(rotation_matrix)
                    train_full_temporary.append({"name": frame_name, "quaternion": quat_R, 'position': pos})

            name_to_ref = {item["name"]: item for item in ref_full_temporary}
            ref_full_temporary = [name_to_ref[item["name"]] for item in ref_full]

            pos_ref_temporary = np.array([item["position"] for item in ref_full_temporary]).T
            quat_ref_temporary = np.array([item["quaternion"] for item in ref_full_temporary])
            pos_train_temporary = np.array([item["position"] for item in train_full_temporary]).T
            quat_train_temporary = np.array([item["quaternion"] for item in train_full_temporary])

            R, c, t = ralign(pos_ref_temporary, pos_ref)
            pos_train_rotated = c * np.dot(R, pos_train_temporary) + t[:, np.newaxis]
            pos_train_rotated = pos_train_rotated.T

            quat_train_rotated = []
            for quaternion in quat_train_temporary:
                quat_train_rotated.append(rotate_quaternion(quaternion, R))

            for quat, pos in zip(quat_train_rotated, pos_train_rotated):
                homogenous = quat_to_homo(quat, pos)
                new_frame = frame
                new_frame["transform_matrix"] = homogenous.tolist()
                data_final["frames"].append(new_frame)

    with open("transforms.json", "w") as f:
        json.dump(data_final, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align quaternions and save transformed frames.")
    parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing the scene data.")
    args = parser.parse_args()

    main(args.directory)
