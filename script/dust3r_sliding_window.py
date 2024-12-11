
import numpy as np
import os
import quaternionic
import shutil
import json

def create_dir(json_path, backup_dir):
    # Check if the source transforms.json file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The file {json_path} does not exist.")
    
    # Create the backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    # Copy the transforms.json to the backup directory as transforms_copy.json
    shutil.copy(json_path, os.path.join(backup_dir, 'transforms_copy.json'))
    
    # Create an empty transforms.json file in the backup directory
    empty_json = {}
    with open(os.path.join(backup_dir, 'transforms.json'), 'w') as f:
        json.dump(empty_json, f, indent=4)


def ralign(X,Y):
    
    """
    Copyright: Carlo Nicolini, 2013
    Code adapted from the Mark Paskin Matlab version
    from http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m 
    """
    m, n = X.shape
    mx = X.mean(1)
    my = Y.mean(1)
    Xc =  X - np.tile(mx, (n, 1)).T
    Yc =  Y - np.tile(my, (n, 1)).T
    sx = np.mean(np.sum(Xc*Xc, 0))
    sy = np.mean(np.sum(Yc*Yc, 0))
    Sxy = np.dot(Yc, Xc.T) / n
    U,D,V = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    V=V.T.copy()
    #print U,"\n\n",D,"\n\n",V
    r = np.ndim(Sxy)
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.det(Sxy) < 0 ):
            S[m, m] = -1
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1  
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R,c,t
    R = np.dot( np.dot(U, S ), V.T)
    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)
    return R,c,t

def get_pos_ref(file_path, folder_path):

    file_path = os.path.join(file_path, 'images.txt')
    # Get the list of image names in the folder
    image_names = set(os.listdir(folder_path))

    rows = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components
            values = line.split()
            if values:
                image_name = values[-1]  # Assuming the image name is the last column

                # Check if the image name is in the folder
                if image_name in image_names:
                    # Extract the desired columns and convert to floats
                    extracted_values = list(map(float, values[5:8]))
                    rows.append(extracted_values)

    # Convert the list of rows to a numpy array
    array = np.array(rows)

    #print(array)  # Optional: can be removed or replaced with a logging statement
    return array.T

def get_quaternion_ref(file_path,ref_path):
    
    # Get the list of image names in the folder
    image_names = set(os.listdir(ref_path))

    rows = []
    file_path = os.path.join(file_path, 'cameras.txt')

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components
            values = line.split()
            if values:
                image_name = values[-1]  # Assuming the image name is the last column

                # Check if the image name is in the folder
                if image_name in image_names:
                    # Extract the desired columns and convert to floats
                    extracted_values = list(map(float, values[1:5]))
                    extracted_values = np.array(extracted_values)
                    rows.append(extracted_values)

    return rows

def img_ref(file_path):

    rows = []
    names = []
    camera_param = []
    file_path1 = os.path.join(file_path, 'images.txt')
    with open(file_path1, 'r') as file:
        for line in file:
            # Split the line into components
            values = line.split()
            if values:
                image_name = values[-1]  # Assuming the image name is the last column

                # Check if the image name is in the folder
                    # Extract the desired columns and convert to floats
                extracted_values = list(map(float, values[1:8]))
                extracted_values = np.array(extracted_values)
                rows.append(extracted_values)
                names.append(image_name)
                
    file_path1 = os.path.join(file_path, 'cameras.txt')
    with open(file_path1, 'r') as file:
        for line in file:
            # Split the line into components
            values = line.split()
            if values:
                extracted_values = list(map(float, values[2:]))
                extracted_values = np.array(extracted_values)
                camera_param.append(extracted_values)
                

    return rows,names,camera_param


def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a quaternion."""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])

def rotate_quaternion(q,R):

    q1 = q
    R1 = R
    R= quaternionic.array.from_rotation_matrix(R)
    q = quaternionic.array(q1)
    
    R = R.normalized
    q = q.normalized
    
    R_conj = R.conjugate()
    rotated = R*q
    rotated = rotated.normalized
    #rotated = rotated*R_con
    return rotated
   


def get_pos_full(file_path):

    rows = []
    names = []
    camera_param = []
    file_path1 = os.path.join(file_path, 'images.txt')
    
    with open(file_path1, 'r') as file:
        for line in file:
            # Split the line into components
            values = line.split()
            if values:
                image_name = values[-1]  # Assuming the image name is the last column
                # Extract the desired columns and convert to floats
                extracted_values = list(map(float, values[5:8]))
                extracted_values = np.array(extracted_values)
                rows.append(extracted_values)
                names.append(image_name)

    file_path1 = os.path.join(file_path, 'cameras.txt')
    with open(file_path1, 'r') as file:
        for line in file:
            # Split the line into components
            values = line.split()
            if values:
                extracted_values = list(map(float, values[2:]))
                extracted_values = np.array(extracted_values)
                camera_param.append(extracted_values)
    rows = np.array(rows)
    return rows.T,names,camera_param

def get_quaternion_full(folder_path):

    rows = []
    folder_path = os.path.join(folder_path, 'images.txt')
    with open(folder_path, 'r') as file:
        for line in file:
            # Split the line into components
            values = line.split()
            if values:
                extracted_values = list(map(float, values[1:5]))
                extracted_values = np.array(extracted_values)
                rows.append(extracted_values)
    return rows

def quat_to_homo(q,t,scale):
    """tranform q (an array [q0,q1,q2,q3]) and T ([x,y,z]) to
        an homgenous matrix
        return the homogenous in the form of a list of list-"""
    q = quaternionic.array(-q)
    
    rotation = q.to_rotation_matrix
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = t*scale
    c2w = np.linalg.inv(T)
    c2w[0:3,2] *= -1 # flip the y and z axis
    c2w[0:3,1] *= -1
    c2w = c2w[[1,0,2,3],:]
    c2w[2,:] *= -1 # flip whole world upside down
    c2w = c2w.tolist()
    return c2w


def rewrite_json(homogenous, name, json_path, backup_dir,camera_param):
    # Define the path for transforms.json and its copy
    transforms_json_path = os.path.join(backup_dir, 'transforms.json')
    transforms_copy_json_path = os.path.join(backup_dir, 'transforms_copy.json')
    


    # Load the existing transforms_copy.json to modify it
    with open(transforms_copy_json_path, 'r') as f:
        transforms_data = json.load(f)

    # Find the corresponding frame by its name and update it
    frame_found = False
    new_frame = None
    for frame in transforms_data["frames"]:
        if frame["file_path"].endswith(name):
            # Update the transform matrix with the homogeneous matrix
            frame["transform_matrix"] = homogenous
            new_frame = frame
            # Flag that the frame was found and modified
            frame_found = True
            break

    # If the frame was found, append it to transforms.json
    if frame_found:
        # Check if transforms.json exists and is not empty
        if os.path.exists(transforms_json_path) and os.path.getsize(transforms_json_path) > 10:
            # Load the existing transforms.json to append the modified frame
            with open(transforms_json_path, 'r') as f:
                current_transforms_data = json.load(f)
        else:
            # If the file is empty or doesn't exist, initialize with an empty frames list
            current_transforms_data = {"frames": []}
            """ current_transforms_data = {"w": 480,
                                        "h": 640,
                                        "fl_x": 651.3070264293906,
                                        "fl_y": 662.6473858525916,
                                        "cx": 235.47617353194124,
                                        "cy": 300.19070215471106,
                                        "k1": -0.047002285856008276,
                                        "k2": 0.04458670803824037,
                                        "p1": -0.0017245655384768163,
                                        "p2": -0.0006383928538902444,
                                        "camera_model": "OPENCV",
                                        "frames": []
                                    } """
        
         # Append the modified frame to the list of frames
        new_frame.update({
            "w": camera_param[0],
            "h": camera_param[1],
            "fl_x": camera_param[2],
            "fl_y": camera_param[3],
            "cx": camera_param[4],
            "cy": camera_param[5],
            "k1": 0,
            "k2": 0,
            "k3": 0,
            "k4": 0,
            "p1": 0,
            "p2": 0,
            "is_fisheye": False,
            "camera_model": "PINHOLE"
        }) 
        current_transforms_data["frames"].append(new_frame)

        # Save the updated transforms_data back to transforms.json
        with open(transforms_json_path, 'w') as f:
            json.dump(current_transforms_data, f, indent=4)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Define paths relative to the project root
    ref_path = os.path.join(project_root, 'data', 'dust3r', 'kettle_GS', 'ref')
    A_path = os.path.join(project_root, 'data', 'dust3r', 'kettle_GS', 'kettle_1', '30_views', 'sparse', '0')
    B_path = os.path.join(project_root, 'data', 'dust3r', 'kettle_GS', 'kettle_2', '30_views', 'sparse', '0')
    C_path = os.path.join(project_root, 'data', 'dust3r', 'kettle_GS', 'kettle_3', '30_views', 'sparse', '0')
    D_path = os.path.join(project_root, 'data', 'dust3r', 'kettle_GS', 'kettle_4', '17_views', 'sparse', '0')

    json_path = os.path.join(project_root, 'data', 'transforms.json')
    backup_dir = os.path.join(project_root, 'data', 'results')

    list_path = [A_path,B_path,C_path,D_path]


    A=get_pos_ref(A_path,ref_path)
    q_A = get_quaternion_ref(A_path,ref_path)
    scale = 1.
    all_frames = []
    create_dir(json_path,backup_dir)
    for fold_path in list_path:
        if fold_path == A_path:
            # A_path = 10quat, ref_path = matching names

            #rows has quat and pos
            rows,image_name,camera_params = img_ref(A_path)
            for element,name,cam_param in zip(rows,image_name,camera_params):
                q_ref = np.array(element[:4])
                t_ref = np.array(element[4:])
                homogenous = quat_to_homo(q_ref,t_ref,scale)
                rewrite_json(homogenous,name,json_path,backup_dir,cam_param)
            continue

        B=get_pos_ref(fold_path,ref_path)
        
        R, c, t = ralign(B,A)

        B,names,camera_params = get_pos_full(fold_path)
        B_projected = c*np.dot(R,B) +t[:, np.newaxis]
        B_projected = B_projected.T

        q_B = get_quaternion_full(fold_path)
        q_B_rotated = []

        for quaternion in q_B:
            q_B_rotated.append(rotate_quaternion(quaternion,R))
        q_B_rotated = np.array(q_B_rotated)


        for quat, pos,name,cam_param in zip(q_B_rotated, B_projected,names,camera_params):
            homogenous = quat_to_homo(quat,pos,scale)
            rewrite_json(homogenous,name,json_path,backup_dir,cam_param)
        
    print('transforms.json saved in data/results')
    #here I want to edit the transforms.json
    #I want the image name with it's equivalent name
