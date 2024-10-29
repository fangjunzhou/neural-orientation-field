import pycolmap
import numpy as np
import scipy.spatial.transform as transform

def get_point_cloud(model: pycolmap.Reconstruction):
    """Extract point cloud data from the COLMAP reconstruction.

    Args:
        model: COLMAP reconstruction model.

    Returns: (points: np.ndarray, colors: np.ndarray)
        points: (num_points, 3) np.ndarray containing all the positions of the 
        point.
        colors: (num_points, 3) np.ndarray containing all the colors of the 
        point.
    """
    num_points = len(model.points3D)
    points = np.zeros((num_points, 3))
    colors = np.zeros((num_points, 3))
    # Extract point cloud.
    for i, point in enumerate(model.points3D.values()):
        points[i] = point.xyz
        colors[i] = point.color
    # Scale color.
    colors = colors / 256
    return points, colors

def get_camera_poses(model: pycolmap.Reconstruction):
    """Extract camera data for each image from the COLMAP reconstruction.

    Args:
        model: COLMAP reconstruction model.

    Returns: (cam_transes: np.ndarray, cam_rots, np.ndarray, 
    cam_params: np.ndarray, image_file_names: list[str])
        cam_transes: (num_images, 3) np.ndarray containing all the positions 
        of the cameras.
        cam_rots: (num_cams, 4) np.ndarray containing all the camera 
        rotation as (x, y, z, w) quaternions.
        cam_params: (num_cams, 3) np.ndarray containing all the camera 
        parameters as (f, cx, cy) pairs.
        image_file_names: a list of str of size num_cams. Containing all the 
        file names of the image to the corresponding camera.
    """
    images = model.images.values()
    num_images = len(images)
    # Extract camera view.
    cam_transes = np.zeros((num_images, 3))
    cam_rots = np.zeros((num_images, 4))
    cam_params = np.zeros((num_images, 3))
    image_file_names = []

    for i, image in enumerate(images):
        # Get camera pose.
        cam_trans = image.cam_from_world.inverse().translation
        cam_rot = image.cam_from_world.inverse().rotation.quat
        cam_transes[i, :] = cam_trans
        cam_rots[i, :] = cam_rot
        cam_id = image.camera_id
        cam_params[i, :] = model.cameras[cam_id].params
        # Add image file name.
        image_file_names.append(image.name)

    return cam_transes, cam_rots, cam_params, image_file_names

def get_bases_from_quat(quat: np.ndarray):
    """Get the camera bases from the rotation quaternion.

    Args:
        quat: camera rotation.

    Returns: (cam_x: np.ndarray, cam_y: np.ndarray, cam_z: np.ndarray)
        cam_x: points to the left of the camera.
        cam_y: points to the up of the camera.
        cam_z: points to the front of the camera.
    """
    rot_mat = transform.Rotation.from_quat(quat).as_matrix()
    cam_x: np.ndarray = np.matmul(rot_mat, np.array([-1, 0, 0]))
    cam_y: np.ndarray = np.matmul(rot_mat, np.array([0, -1, 0]))
    cam_z: np.ndarray = np.matmul(rot_mat, np.array([0, 0, 1]))
    return cam_x, cam_y, cam_z
