import argparse
import numpy as np
import open3d as o3d

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert a PLY point cloud to a TXT file with ID, XYZ, and RGB.")
    parser.add_argument("--ply_file", required=True, help="Path to the input PLY file")
    parser.add_argument("--txt_file", required=True, help="Path to the output TXT file")
    args = parser.parse_args()

    # Load the PLY file
    ply_file_path = args.ply_file
    txt_file_path = args.txt_file

    print(f"Loading PLY file: {ply_file_path}")
    pcd = o3d.io.read_point_cloud(ply_file_path)

    # Convert the point cloud to numpy arrays
    points = np.asarray(pcd.points)  # XYZ coordinates
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)  # RGB colors, scaled to 0-255

    # Combine ID, XYZ, and RGB
    num_points = points.shape[0]
    ids = np.arange(1, num_points + 1).reshape(-1, 1)  # Create IDs starting from 1
    data = np.hstack((ids, points, colors))  # Combine ID, XYZ, and RGB

    # Save to a text file
    header = "ID X Y Z R G B"
    np.savetxt(txt_file_path, data, fmt="%d %.6f %.6f %.6f %d %d %d", comments='')

    print(f"Converted {ply_file_path} to {txt_file_path}")

if __name__ == "__main__":
    main()
