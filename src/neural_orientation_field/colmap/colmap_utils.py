import logging
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

    Returns: (cam_transforms: np.ndarray, cam_params: np.ndarray,
              image_file_names: list[str])
        cam_transforms: (num_images, 4, 4) np.ndarray containing the 
        transformation matrix for each camera in homogeneous coordinate.
        cam_params: (num_cams, 3) np.ndarray containing all the camera 
        parameters as (f, cx, cy) pairs.
        image_file_names: a list of str of size num_cams. Containing all the 
        file names of the image to the corresponding camera.
    """
    images = model.images.values()
    num_images = len(images)
    # Extract camera view.
    cam_transforms = np.zeros((num_images, 4, 4))
    cam_params = np.zeros((num_images, 3))
    image_file_names = []

    for i, image in enumerate(images):
        # Get camera pose.
        cam_trans = image.cam_from_world.translation
        cam_rot = image.cam_from_world.rotation.quat
        trans_mat = np.identity(4)
        trans_mat[0:3, 3] = cam_trans
        rot_mat = np.identity(4)
        rot_mat[:3, :3] = transform.Rotation.from_quat(cam_rot).as_matrix()
        cam_transforms[i] = np.matmul(trans_mat, rot_mat)
        cam_id = image.camera_id
        cam_params[i, :] = model.cameras[cam_id].params
        # Add image file name.
        image_file_names.append(image.name)

    return cam_transforms, cam_params, image_file_names

def get_projection_mat(f: float, cx: float, cy: float, near:float, far: float):
    """Get the projection matrix given the camera parameter and clipping 
    distance.

    Args:
        f: focal length.
        cx: half width of the focal plane.
        cy: half hight of the focal plane.
        near: near clipping distance.
        far: far clipping distance.

    Returns:
        A 4x4 projection matrix from the eye space to the clipping space. To
        further convert to NDC space, divide all coordinate by z.
    """
    return np.array([
        [f/cx, 0, 0, 0],
        [0, f/cy, 0, 0],
        [0, 0, -(far + near)/(far - near), 2*far*near/(far - near)],
        [0, 0, -1, 0]
    ])
