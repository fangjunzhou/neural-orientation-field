import pathlib
import argparse
import logging
import sys
import json
from dataclasses import dataclass

from PIL import Image
import numpy as np
import scipy.spatial.transform as transform

import pycolmap
import pyvista as pv
from pyvista_imgui import ImguiPlotter
from imgui_bundle import imgui, immapp, imgui_ctx, immvision

import neural_orientation_field.utils as utils
import neural_orientation_field.colmap.colmap_utils as colutils

logging.basicConfig(level=logging.DEBUG)

def main():
    # ---------------------- Argument Setup ---------------------- #
    parser = argparse.ArgumentParser(
        prog="COLMAP Visualizer",
        description="""
        Visualize the point cloud and camera pose data extracted by COLMAP.
        """
    )
    parser.add_argument(
        "-p",
        "--path",
        default=pathlib.Path("./nof-config.json"),
        help="""
        The project config json file path. The default path is 
        ./nof-config.json
        """,
        type=pathlib.Path
    )
    parser.add_argument(
        "-m",
        "--model",
        default=pathlib.Path("./data/cache/colmap/0/"),
        help="""
        The COLMAP model path. Should be a directory containing cameras.bin, 
        images.bin, and points3D.bin. The default path is ./data/cache/colmap/0/
        """,
        type=pathlib.Path
    )
    # --------------------- Parse Arguments  --------------------- #
    args = parser.parse_args()
    config_path: pathlib.Path = args.path
    model_path: pathlib.Path = args.model
    if not model_path.exists():
        logging.error("The COLMAP model directory doesn't exist.")
        sys.exit(1)
    if not config_path.exists():
        logging.error("The project config doesn't exist.")
        sys.exit(1)
    with open(config_path, "r") as config_file:
        try:
            config_dict = json.load(config_file)
            project_config = utils.ProjectConfig.from_dict(config_dict)
        except Exception as e:
            logging.error("Failed to load project config.", stack_info=True)
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
    cam_transes, cam_rots, cam_params, image_file_names = \
        colutils.get_camera_poses(model)
    # ------------------- COLMAP Visualization ------------------- #
    pl = ImguiPlotter()

    def draw_point_cloud(points: np.ndarray, colors: np.ndarray):
        pl.add_points(
            points,
            scalars=colors,
            rgba=True
        )

    def draw_cameras(
        cam_transes: np.ndarray,
        cam_rots: np.ndarray,
        cam_params: np.ndarray
    ):
        num_cam = cam_transes.shape[0]
        cam_xs = np.zeros((num_cam, 3))
        cam_ys = np.zeros((num_cam, 3))
        cam_zs = np.zeros((num_cam, 3))
        cam_gizmos_v = np.zeros((num_cam * 5, 3))
        cam_gizmos_e = np.zeros((num_cam * 8, 3), dtype=int)
        for i in range(num_cam):
            # Camera transposition and orientation.
            cam_trans = cam_transes[i]
            cam_rot = cam_rots[i]
            cam_x, cam_y, cam_z = colutils.get_bases_from_quat(cam_rot)
            cam_xs[i] = cam_x
            cam_ys[i] = cam_y
            cam_zs[i] = cam_z
            # Camera parameters.
            f, cx, cy = cam_params[i]
            dx = cx / f
            dy = cy / f
            cam_gizmos_v[i*5 + 0] = cam_trans
            cam_gizmos_v[i*5 + 1] = cam_trans + cam_z + cam_x * dx + cam_y * dy
            cam_gizmos_v[i*5 + 2] = cam_trans + cam_z + cam_x * dx - cam_y * dy
            cam_gizmos_v[i*5 + 3] = cam_trans + cam_z - cam_x * dx + cam_y * dy
            cam_gizmos_v[i*5 + 4] = cam_trans + cam_z - cam_x * dx - cam_y * dy
            cam_gizmos_e[i*8 + 0] = [2, i*5 + 0, i*5 + 1]
            cam_gizmos_e[i*8 + 1] = [2, i*5 + 0, i*5 + 2]
            cam_gizmos_e[i*8 + 2] = [2, i*5 + 0, i*5 + 3]
            cam_gizmos_e[i*8 + 3] = [2, i*5 + 0, i*5 + 4]
            cam_gizmos_e[i*8 + 4] = [2, i*5 + 1, i*5 + 2]
            cam_gizmos_e[i*8 + 5] = [2, i*5 + 3, i*5 + 4]
            cam_gizmos_e[i*8 + 6] = [2, i*5 + 1, i*5 + 3]
            cam_gizmos_e[i*8 + 7] = [2, i*5 + 2, i*5 + 4]
        # Draw camera mesh.
        cam_points = pv.PolyData(cam_transes)
        cam_points["up"] = cam_ys
        cam_up_arrows = cam_points.glyph(
            orient="up", # pyright: ignore
            scale=False,
            factor=0.5
        )
        pl.add_mesh(cam_points, color="red")
        pl.add_mesh(cam_up_arrows, color="blue")
        cam_mesh = pv.PolyData(cam_gizmos_v, cam_gizmos_e)
        pl.add_mesh(cam_mesh, color="red", style="wireframe")


    @dataclass
    class AppState:
        trim_point_cloud: bool
        trim_distance: float
        points: np.ndarray
        colors: np.ndarray
        show_one_cam: bool
        cam_id: int
        cam_image: np.ndarray | None = None

        def draw(self):
            """Redraw the scene."""
            pl.clear()
            pl.show_axes() # pyright: ignore
            draw_point_cloud(self.points, self.colors)
            if self.show_one_cam:
                draw_cameras(
                    cam_transes[self.cam_id].reshape(1, -1),
                    cam_rots[self.cam_id].reshape(1, -1),
                    cam_params[self.cam_id].reshape(1, -1)
                )
            else:
                draw_cameras(cam_transes, cam_rots, cam_params)

        def load_cam_image(self):
            image_file_name: str = image_file_names[self.cam_id]
            image_file_path = project_config.input_path / image_file_name
            self.cam_image = np.array(Image.open(image_file_path))

    app_state = AppState(
        trim_point_cloud=True,
        trim_distance=10,
        points=points,
        colors=colors,
        show_one_cam=False,
        cam_id=0,
    )
    app_state.draw()

    def gui():
        # ---------------------- Immediate GUI  ---------------------- #
        control_panel_width = max(300, imgui.get_content_region_max().x * 0.2)
        # PyVista plotter window.
        with imgui_ctx.begin_child(
            "pyvista_plotter",
            size=imgui.ImVec2(
                imgui.get_content_region_avail().x - control_panel_width,
                imgui.get_content_region_max().y
            ),
            child_flags=imgui.ChildFlags_.border.value
        ):
            pl.render_imgui()

        imgui.same_line()

        with imgui_ctx.begin_child(
            "plotter_control",
            size=imgui.ImVec2(
                control_panel_width,
                imgui.get_content_region_avail().y
            )
        ):
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
                    app_state.points = points[dists <= app_state.trim_distance, :]
                    app_state.colors = colors[dists <= app_state.trim_distance, :]
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
                    app_state.points = points[dists <= app_state.trim_distance, :]
                    app_state.colors = colors[dists <= app_state.trim_distance, :]
                    app_state.draw()
            # ----------------------- Camera View  ----------------------- #
            imgui.separator_text("Camera View")
            changed, new_one_cam = imgui.checkbox(
                "Select One Camera",
                app_state.show_one_cam
            )
            if changed:
                app_state.show_one_cam = new_one_cam
                app_state.draw()
                app_state.load_cam_image()
            if app_state.show_one_cam:
                changed, new_cam_id = imgui.slider_int(
                    "Camera ID",
                    app_state.cam_id,
                    v_min=0,
                    v_max=cam_transes.shape[0]-1
                )
                if changed:
                    app_state.cam_id = new_cam_id
                    app_state.draw()
                    app_state.load_cam_image()

            if app_state.show_one_cam and \
                (type(app_state.cam_image) is np.ndarray):
                immvision.image_display_resizable(
                    image_file_names[app_state.cam_id],
                    app_state.cam_image,
                    size=imgui.ImVec2(imgui.get_content_region_avail().x, 0),
                    is_bgr_or_bgra=False
                )


    immapp.run(
        gui_function=gui,
        window_title="COLMAP Visualizer",
        window_size_auto=False
    )

if __name__ == "__main__":
    main()
