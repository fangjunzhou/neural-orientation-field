import bpy
import random
import math
import os
import numpy as np
from mathutils import Vector

random.seed(42)

# Define the center of the scene
scene_center = Vector((0, 0, 15))

# Define the radius of the sphere on which the camera will be placed
radius = 10

cam_transforms = []
frame_paths = []

# Function to position and orient the camera
def position_camera_randomly(frame_number):
    # Generate random spherical coordinates
    theta = random.uniform(0, 2 * math.pi)  # Random angle in radians
    phi = random.uniform(0.1 * math.pi, 0.5 * math.pi)        # Random angle in radians

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)

    # Set camera location
    bpy.context.scene.camera.location = Vector((x, y, z)) + scene_center

    # Make the camera point to the scene center
    direction = bpy.context.scene.camera.location - scene_center
    rot_quat = direction.to_track_quat('Z', 'Y')
    bpy.context.scene.camera.rotation_euler = rot_quat.to_euler()

    # Insert keyframes for location and rotation
    bpy.context.scene.camera.keyframe_insert(data_path='location', frame=frame_number)
    bpy.context.scene.camera.keyframe_insert(data_path='rotation_euler', frame=frame_number)
    
    cam_transforms.append(np.matrix(bpy.context.scene.camera.matrix_world))
    frame_paths.append(bpy.context.scene.render.frame_path(frame=frame_number))

# Set the total number of frames
total_frames = 128
bpy.context.scene.frame_end = total_frames

directory = os.path.dirname(bpy.data.filepath)
# Camera intrinsics.
h = bpy.data.scenes["hairstyles main"].render.resolution_y
w = bpy.data.scenes["hairstyles main"].render.resolution_x
cx = w/2
cy = h/2
f = bpy.data.cameras["Camera"].lens / bpy.data.cameras["Camera"].sensor_width * w
with open(directory + "/camera-params.npy", "wb") as cam_param_file:
    np.save(cam_param_file, (h, w, f, cx, cy))

# Position the camera randomly for each frame
for frame in range(1, total_frames + 1):
    position_camera_randomly(frame)
    
cam_transforms = np.array(cam_transforms)
with open(directory + "/camera-transforms.npy", "wb") as cam_trans_file:
    np.save(cam_trans_file, cam_transforms)

with open(directory + "/frame-paths.txt", "w") as frame_file:
    frame_file.writelines(frame_paths)