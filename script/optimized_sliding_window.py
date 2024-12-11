"""
RALIGN - Rigid alignment of two sets of points in k-dimensional
        Euclidean space.  Given two sets of points in
        correspondence, this function computes the scaling,
        rotation, and translation that define the transform TR
        that minimizes the sum of squared errors between TR(X)
        and its corresponding points in Y.  This routine takes
        O(n k^3)-time.
Inputs:
 X - a k x n matrix whose columns are points 
 Y - a k x n matrix whose columns are points that correspond to
     the points in X
Outputs: 
 c, R, t - the scaling, rotation matrix, and translation vector
           defining the linear map TR as 

                     TR(x) = c * R * x + t
           such that the average norm of TR(X(:, i) - Y(:, i))
           is minimized.
"""
"""
Copyright: Carlo Nicolini, 2013
Code adapted from the Mark Paskin Matlab version
from http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m 
"""
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

def get_pos_ref(file_path, folder_path,file_npy):

    # Get the list of image names in the folder
    image_names = set(os.listdir(folder_path))
    data_pose = np.load(file_npy)
    with open(file_path, 'r') as f:
        data = json.load(f)
    positions = []
    i = 0
    for entry in data:
        # Split the line into components
        image_name = entry['img_name'] + '.JPG'  # Assuming the image name is the last column

        # Check if the image name is in the folder
        if image_name in image_names:
            # Extract the desired columns and convert to floats
            #positions.append(np.array(entry['position']))
            positions.append(np.array(data_pose[i,:3,-1]))
        i +=1
    #print(array)  # Optional: can be removed or replaced with a logging statement
    positions = np.array(positions)
    return positions.T

def get_quaternion_ref(file_path,ref_path,file_npy):
    
    # Get the list of image names in the folder
    image_names = set(os.listdir(ref_path))
    data_pose = np.load(file_npy)
    with open(file_path, 'r') as f:
        data = json.load(f)
    rows = []
    i =0
    for entry in data:
        image_name = entry['img_name'] + '.JPG'  # Assuming the image name is the last column
        # Check if the image name is in the folder
        if image_name in image_names:
            #rotation_matrix = np.array(entry['rotation'])
            rotation_matrix = np.array(data_pose[i,:3,:3])
            quat_R = quaternionic.array.from_rotation_matrix(rotation_matrix)
            rows.append(quat_R)
        i +=1
    return rows

def img_ref(file_path,file_npy):

    rows = []
    names = []
    camera_param = []
    i = 0
    with open(file_path, 'r') as f:
        data = json.load(f)
    data_pose = np.load(file_npy)
    for entry in data:
        single_param=[]
        image_name = entry['img_name'] + '.JPG'
        names.append(image_name)
        #pos = np.array(entry['position'])
        pos = np.array(data_pose[i,:3,-1])
        #rotation_matrix = np.array(entry['rotation'])
        rotation_matrix = np.array(data_pose[i,:3,:3])
        quat_R = quaternionic.array.from_rotation_matrix(rotation_matrix)
        rows.append(np.concatenate((quat_R, pos)))

        single_param.append(entry["width"])
        single_param.append(entry["height"])
        single_param.append(entry["fx"])
        single_param.append(entry["fy"])

        camera_param.append(np.array(single_param))
        i +=1
    return rows,names,camera_param



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
   


def get_pos_full(file_path,file_npy):

    rows = []
    names = []
    camera_param = []

    with open(file_path, 'r') as f:
        data = json.load(f)
    data_pose = np.load(file_npy)
    i =0
    for entry in data:
        single_param = []
        rows.append(np.array(data_pose[i,:3,-1]))
        
        image_name = entry['img_name'] + '.JPG'
        names.append(image_name)

        single_param.append(entry["width"])
        single_param.append(entry["height"])
        single_param.append(entry["fx"])
        single_param.append(entry["fy"])

        camera_param.append(np.array(single_param))
        i +=1
    rows = np.array(rows)
    return rows.T,names,camera_param

def get_quaternion_full(folder_path,file_npy):

    rows = []
    with open(folder_path, 'r') as f:
        data = json.load(f)
    data_pose = np.load(file_npy)
    i = 0
    for entry in data:
        #rotation_matrix = np.array(entry['rotation'])
        rotation_matrix = np.array(data_pose[i,:3,:3])
        quat_R = quaternionic.array.from_rotation_matrix(rotation_matrix)
        rows.append(quat_R)
        i +=1

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
            "k1": 0,
            "k2": 0,
            "k3": 0,
            "k4": 0,
            "p1": 0,
            "p2": 0,
            "cx": 240.0,
            "cy": 320.0,
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
    A_path = os.path.join(project_root, 'data', 'optimized', 'kettle_GS', 'kettle_1', 
                        '30_views_1000Iter_1xPoseLR', 'cameras.json')
    A_pose_path = os.path.join(project_root, 'data', 'optimized', 'kettle_GS', 'kettle_1', 
                            '30_views_1000Iter_1xPoseLR', 'pose', 'pose_1000.npy')
    B_path = os.path.join(project_root, 'data', 'optimized', 'kettle_GS', 'kettle_2', 
                        '30_views_1000Iter_1xPoseLR', 'cameras.json')
    B_pose_path = os.path.join(project_root, 'data', 'optimized', 'kettle_GS', 'kettle_2', 
                            '30_views_1000Iter_1xPoseLR', 'pose', 'pose_1000.npy')
    C_path = os.path.join(project_root, 'data', 'optimized', 'kettle_GS', 'kettle_3', 
                        '30_views_1000Iter_1xPoseLR', 'cameras.json')
    C_pose_path = os.path.join(project_root, 'data', 'optimized', 'kettle_GS', 'kettle_3', 
                            '30_views_1000Iter_1xPoseLR', 'pose', 'pose_1000.npy')
    D_path = os.path.join(project_root, 'data', 'optimized', 'kettle_GS', 'kettle_4', 
                        '17_views_1000Iter_1xPoseLR', 'cameras.json')
    D_pose_path = os.path.join(project_root, 'data', 'optimized', 'kettle_GS', 'kettle_4', 
                            '17_views_1000Iter_1xPoseLR', 'pose', 'pose_1000.npy')

    json_path = os.path.join(project_root, 'data', 'transforms.json')
    backup_dir = os.path.join(project_root, 'data', 'results')
    list_path = [A_path,B_path,C_path,D_path]
    list_path_pose = [A_pose_path,B_pose_path,C_pose_path,D_pose_path]

    A=get_pos_ref(A_path,ref_path,A_pose_path)
    q_A = get_quaternion_ref(A_path,ref_path,A_pose_path)
    scale = 1.
    all_frames = []
    create_dir(json_path,backup_dir)
    j = 0
    for fold_path in list_path:
        if fold_path == A_path:
            # A_path = 10quat, ref_path = matching names

            #rows has quat and pos
            rows,image_name,camera_params = img_ref(A_path,A_pose_path)
            for element,name,cam_param in zip(rows,image_name,camera_params):
                q_ref = np.array(element[:4])
                t_ref = np.array(element[4:])
                homogenous = quat_to_homo(q_ref,t_ref,scale)
                rewrite_json(homogenous,name,json_path,backup_dir,cam_param)
            j +=1
            continue

        B=get_pos_ref(fold_path,ref_path,list_path_pose[j])
    
        R, c, t = ralign(B,A)

        B,names,camera_params = get_pos_full(fold_path,list_path_pose[j])
        B_projected = c*np.dot(R,B) +t[:, np.newaxis]
        B_projected = B_projected.T
        
        q_B = get_quaternion_full(fold_path,list_path_pose[j])
        q_B_rotated = []

        for quaternion in q_B:
            q_B_rotated.append(rotate_quaternion(quaternion,R))
        q_B_rotated = np.array(q_B_rotated)


        for quat, pos,name,cam_param in zip(q_B_rotated, B_projected,names,camera_params):
            homogenous = quat_to_homo(quat,pos,scale)
            rewrite_json(homogenous,name,json_path,backup_dir,cam_param)
        
        j +=1
    print('transforms.json saved in data/results')