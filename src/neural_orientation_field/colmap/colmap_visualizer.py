import pathlib
import argparse
import logging
import sys
import json
from dataclasses import dataclass

from PIL import Image
import numpy as np
import scipy.spatial.transform as transform
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt

import pycolmap
import pyvista as pv
from pyvista_imgui import ImguiPlotter
from imgui_bundle import imgui, immapp, imgui_ctx, immvision, imgui_fig

import neural_orientation_field.utils as utils
import neural_orientation_field.imgui_utils as imgui_utils
import neural_orientation_field.colmap.colmap_utils as colutils


def main():
    logging.basicConfig(level=logging.DEBUG)
    # ---------------------- Argument Setup ---------------------- #
    parser = argparse.ArgumentParser(
        prog="COLMAP Visualizer",
        description="""
        Visualize the point cloud and camera pose data extracted by COLMAP.
        """
    )
    parser.add_argument(
        "-i",
        "--images",
        required=True,
        help="""
        Input images directory.
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="""
        The COLMAP model path. Should be a directory containing cameras.bin, 
        images.bin, and points3D.bin. The default path is ./data/output/colmap/model/0/
        """,
        type=pathlib.Path
    )
    # --------------------- Parse Arguments  --------------------- #
    args = parser.parse_args()
    image_path: pathlib.Path = args.images
    if not image_path.exists():
        logging.error("The project config doesn't exist.")
        sys.exit(1)
    model_path: pathlib.Path = args.model
    if not model_path.exists():
        logging.error("The COLMAP model directory doesn't exist.")
        sys.exit(1)
    # -------------------- COLMAP Extraction  -------------------- #
    try:
        model = pycolmap.Reconstruction(model_path)
    except Exception as e:
        logging.error(
            f"Failed to read COLMAP model at {str(model_path.resolve())}",
            stack_info=True
        )
        sys.exit(1)
    logging.debug(model.summary())
    # Extract the point cloud data.
    points, colors = colutils.get_point_cloud(model)
    logging.debug(f"points.shape: {points.shape}")
    logging.debug(f"colors.shape: {colors.shape}")
    # Point distance.
    dists = np.linalg.norm(points, axis=1)
    max_dist = np.amax(dists)
    min_dist = np.amin(dists)
    # Extract camera poses.
    cam_transforms, cam_params, image_file_names = \
        colutils.get_camera_poses(model)
    # ------------------- COLMAP Visualization ------------------- #
    pl = ImguiPlotter()

    def draw_point_cloud(points: np.ndarray, colors: np.ndarray):
        """Draw point clouds.

        Args:
            points: (num_points, 3) np.ndarray containing all the position of 
            the points.
            colors: (num_points, 3) np.ndarray containing all the color of the 
            points.
        """
        pl.add_points(
            points,
            scalars=colors,
            rgba=True
        )

    def draw_cameras(
        cam_transforms: np.ndarray,
        cam_params: np.ndarray,
        nc: float,
        fc: float
    ):
        """Draw camera gizmos.

        Args:
            cam_transforms: (num_images, 4, 4) np.ndarray containing the 
            transformation matrix for each camera in homogeneous coordinate.
            cam_params: (num_cams, 3) np.ndarray containing all the camera 
            parameters as (f, cx, cy) pairs.
        """
        num_cam = cam_transforms.shape[0]
        cam_transes = np.zeros((num_cam, 3))
        cam_ups = np.zeros((num_cam, 3))
        cam_gizmos_v = np.zeros((num_cam * 8, 3))
        cam_gizmos_e = np.zeros((num_cam * 12, 3), dtype=int)
        for i in range(num_cam):
            # Inverse of camera transformation.
            # The xyz bases of the inverse transformation points to the right,
            # down, and backward to the camera.
            inv_cam_transform = np.linalg.inv(cam_transforms[i])
            cam_transes[i] = np.matmul(
                inv_cam_transform,
                np.array([0, 0, 0, 1])
            )[:3]
            cam_ups[i] = np.matmul(
                inv_cam_transform, np.array([0, -1, 0, 0]))[:3]
            # Camera parameters.
            f, cx, cy = cam_params[i]
            inv_ndc_proj = np.linalg.inv(
                colutils.get_projection_mat(f, cx, cy, nc, fc)
            )
            for ix, x in enumerate([-1, 1]):
                for iy, y in enumerate([-1, 1]):
                    for iz, z in enumerate([-1, 1]):
                        j = ix * 4 + iy * 2 + iz
                        w = nc if z == -1 else fc
                        cam_gizmos_v[i*8 + j] = np.matmul(
                            np.matmul(inv_cam_transform, inv_ndc_proj),
                            np.array([x*w, y*w, z*w, w])
                        )[:3]
            cam_gizmos_e[i*12 + 0] = [2, i*8 + 0, i*8 + 1]
            cam_gizmos_e[i*12 + 1] = [2, i*8 + 2, i*8 + 3]
            cam_gizmos_e[i*12 + 2] = [2, i*8 + 0, i*8 + 2]
            cam_gizmos_e[i*12 + 3] = [2, i*8 + 1, i*8 + 3]
            cam_gizmos_e[i*12 + 4] = [2, i*8 + 4, i*8 + 5]
            cam_gizmos_e[i*12 + 5] = [2, i*8 + 6, i*8 + 7]
            cam_gizmos_e[i*12 + 6] = [2, i*8 + 4, i*8 + 6]
            cam_gizmos_e[i*12 + 7] = [2, i*8 + 5, i*8 + 7]
            cam_gizmos_e[i*12 + 8] = [2, i*8 + 0, i*8 + 4]
            cam_gizmos_e[i*12 + 9] = [2, i*8 + 1, i*8 + 5]
            cam_gizmos_e[i*12 + 10] = [2, i*8 + 2, i*8 + 6]
            cam_gizmos_e[i*12 + 11] = [2, i*8 + 3, i*8 + 7]
        # Draw camera mesh.
        cam_points = pv.PolyData(cam_transes)
        cam_points["up"] = cam_ups
        cam_up_arrows = cam_points.glyph(
            orient="up",  # pyright: ignore
            scale=False,
            factor=0.5
        )
        pl.add_mesh(cam_points, color="red")
        pl.add_mesh(cam_up_arrows, color="blue")
        cam_mesh = pv.PolyData(cam_gizmos_v, cam_gizmos_e)
        pl.add_mesh(cam_mesh, color="red", style="wireframe")

    @dataclass
    class AppState:
        # ----------------------- Point Cloud  ----------------------- #
        # If trimming the point cloud.
        trim_point_cloud: bool
        # Trimming distance.
        trim_distance: float
        # -------------------------- Camera -------------------------- #
        # Showing single camera view.
        show_one_cam: bool
        show_camera_point: bool
        cam_id: int
        # Near and far clipping distance.
        nc_distance: float
        fc_distance: float
        # ---------------------- Internal State ---------------------- #
        # Point cloud.
        points: np.ndarray
        colors: np.ndarray
        # Display current camera image.
        point_ax: matplotlib.axes.Axes
        point_fig: matplotlib.figure.Figure
        cam_image: np.ndarray | None = None
        # If the app is initialized for the first time (for dockspace).
        init: bool = True

        def draw(self):
            """Redraw the scene."""
            pl.clear()
            pl.show_axes()  # pyright: ignore
            pl.show_grid()  # pyright: ignore
            draw_point_cloud(self.points, self.colors)
            if self.show_one_cam:
                draw_cameras(
                    cam_transforms[self.cam_id].reshape(1, 4, 4),
                    cam_params[self.cam_id].reshape(1, 3),
                    self.nc_distance,
                    self.fc_distance
                )
            else:
                draw_cameras(
                    cam_transforms,
                    cam_params,
                    self.nc_distance,
                    self.fc_distance
                )

        def load_cam_image(self):
            """Load image as np.ndarray"""
            image_file_name: str = image_file_names[self.cam_id]
            image_file_path = image_path / image_file_name
            self.cam_image = np.array(Image.open(image_file_path))

        def plot_point_cloud(self):
            cam_transform = cam_transforms[self.cam_id]
            f, cx, cy = cam_params[self.cam_id]
            cam_projection = colutils.get_projection_mat(
                f,
                cx,
                cy,
                self.nc_distance,
                self.fc_distance
            )
            # Project points to the camera perspective.
            projected_points = np.matmul(
                np.matmul(cam_projection, cam_transform),
                np.concatenate(
                    (self.points, np.ones((self.points.shape[0], 1))),
                    axis=1
                ).T
            ).T
            # Convert to NDC coordinate.
            projected_points = projected_points / np.tile(
                projected_points[:, 3].reshape(-1, 1), (1, 4)
            )
            projected_points = projected_points[:, 0:3]
            # Filter clipping space.
            projected_norms = np.max(np.abs(projected_points), axis=1)
            projected_points = projected_points[projected_norms < 1, :]

            self.point_ax.clear()
            self.point_ax.scatter(
                projected_points[:, 0],
                projected_points[:, 1],
                c=self.colors[projected_norms < 1, :]
            )
            self.point_ax.set_xlim(-1, 1)
            self.point_ax.set_ylim(1, -1)
            self.point_ax.set_aspect(cy/cx)

    point_fig, point_ax = plt.subplots()
    app_state = AppState(
        trim_point_cloud=False,
        trim_distance=max_dist,
        points=points,
        colors=colors,
        show_one_cam=False,
        show_camera_point=False,
        cam_id=0,
        nc_distance=0.1,
        fc_distance=10,
        point_fig=point_fig,
        point_ax=point_ax
    )
    app_state.draw()

    def gui():
        dockspace_id = imgui_utils.setup_dockspace()
        if (app_state.init):
            app_state.init = False
            imgui.internal.dock_builder_remove_node(dockspace_id)
            imgui.internal.dock_builder_add_node(dockspace_id)
            dock1 = imgui.internal.dock_builder_split_node(
                dockspace_id,
                imgui.Dir.left,
                0.75
            )
            imgui.internal.dock_builder_dock_window(
                "PyVista Plotter",
                dock1.id_at_dir
            )
            imgui.internal.dock_builder_dock_window(
                "Plotter Control",
                dock1.id_at_opposite_dir
            )
            imgui.internal.dock_builder_finish(dockspace_id)
        # ---------------------- Immediate GUI  ---------------------- #
        # PyVista plotter window.
        imgui.set_next_window_size_constraints(
            size_min=imgui.ImVec2(300, 200),
            size_max=imgui.ImVec2(imgui.FLT_MAX, imgui.FLT_MAX)
        )
        with imgui_ctx.begin("PyVista Plotter"):
            pl.render_imgui()

        with imgui_ctx.begin("Plotter Control"):
            # --------------------- Trim Point Cloud --------------------- #
            imgui.separator_text("Point Cloud Range")
            # Trim point cloud checkbox.
            changed, new_trim = imgui.checkbox(
                "Trim Point Cloud",
                app_state.trim_point_cloud
            )
            if changed:
                app_state.trim_point_cloud = new_trim
                if new_trim:
                    app_state.points = points[dists <=
                                              app_state.trim_distance, :]
                    app_state.colors = colors[dists <=
                                              app_state.trim_distance, :]
                else:
                    app_state.points = points
                    app_state.colors = colors
                app_state.draw()
            # Trim distance slider.
            if app_state.trim_point_cloud:
                changed, new_trim_dist = imgui.slider_float(
                    "Trim Distance",
                    app_state.trim_distance,
                    v_min=min_dist,
                    v_max=max_dist
                )
                if changed:
                    app_state.trim_distance = new_trim_dist
                    app_state.points = points[dists <=
                                              app_state.trim_distance, :]
                    app_state.colors = colors[dists <=
                                              app_state.trim_distance, :]
                    app_state.draw()
            # ----------------------- Camera View  ----------------------- #
            imgui.separator_text("Camera View")
            changed, new_nc_distance, new_fc_distance = imgui.drag_float_range2(
                "Clipping Range",
                v_current_min=app_state.nc_distance,
                v_current_max=app_state.fc_distance,
                v_min=0.01,
                v_max=20,
                v_speed=0.1
            )
            if changed:
                if new_nc_distance < new_fc_distance:
                    app_state.nc_distance = new_nc_distance
                    app_state.fc_distance = new_fc_distance
                    app_state.draw()
                    app_state.plot_point_cloud()
            changed, new_one_cam = imgui.checkbox(
                "Select One Camera",
                app_state.show_one_cam
            )
            if changed:
                app_state.show_one_cam = new_one_cam
                app_state.draw()
                app_state.load_cam_image()
                app_state.plot_point_cloud()
            if app_state.show_one_cam:
                changed, new_cam_id = imgui.slider_int(
                    "Camera ID",
                    app_state.cam_id,
                    v_min=0,
                    v_max=cam_transforms.shape[0]-1
                )
                if changed:
                    app_state.cam_id = new_cam_id
                    app_state.draw()
                    app_state.load_cam_image()
                    app_state.plot_point_cloud()

            if app_state.show_one_cam:
                if type(app_state.cam_image) is np.ndarray:
                    immvision.image_display_resizable(
                        image_file_names[app_state.cam_id],
                        app_state.cam_image,
                        size=imgui.ImVec2(
                            imgui.get_content_region_avail().x, 0),
                        is_bgr_or_bgra=False
                    )
                changed, new_camera_point = imgui.checkbox(
                    "Show Camera Points",
                    app_state.show_camera_point
                )
                if changed:
                    app_state.show_camera_point = new_camera_point
                if type(app_state.point_fig) is matplotlib.figure.Figure:
                    if app_state.show_camera_point:
                        imgui_fig.fig(
                            "Point Cloud Plot",
                            app_state.point_fig,
                            refresh_image=True,
                            size=imgui.ImVec2(
                                imgui.get_content_region_avail().x, 0)
                        )

    immapp.run(
        gui_function=gui,
        window_title="COLMAP Visualizer",
        window_size=(960, 540)
    )


if __name__ == "__main__":
    main()
