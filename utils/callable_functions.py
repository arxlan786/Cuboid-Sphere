
import h5py
import numpy as np


def process_hdf5_file(filename):
    with h5py.File(filename, "r") as f:
        depth_map     = f['depth_map'] 
        intensity_map = f['intensity_map']
        fx = depth_map.attrs['horizontal_fov_deg']
        fy = depth_map.attrs['vertical_fov_deg']
        depth_map     = np.array(depth_map, np.uint8)
        intensity_map = np.array(intensity_map, np.uint8)
        return {'depth_map'    : depth_map,
                'intensity_map': intensity_map,
                'fx':fx,'fy':fy
               }

def save_point_cloud(points, save_path="./point_cloud.hdf5"):

    # Create HDF5 file
    hdf5 = h5py.File(save_path, 'w')

    # Create datasets
    hdf5.create_dataset('points', data=points)
    hdf5.close()
    print("Point cloud saved to: ", save_path)
    
    

def depth2pointcloud(metadata, Cx=0.5, Cy=0.5):
    
    h,w = metadata['intensity_map'].shape
    
    point_clouds = []
    
    for v in range(h):
        for u in range(w):
            z = metadata['intensity_map'][v,u]
            
            # convert to point cloud as followd by above link
            # AS u/f = x/z
            # x = u*z/f
                
            x_over_z = (u - Cx) * z / metadata['fx']
            y_over_z = (v - Cy) * z / metadata['fy']
            
            
            # to skip the negative or invalid pixels
            if z == 0:
                continue            
            
            point_clouds.append([x_over_z, y_over_z, z])
            
    return point_clouds



    