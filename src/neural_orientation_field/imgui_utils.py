from imgui_bundle import imgui, imgui_ctx

def setup_dockspace(name="dock_space"):
    """Setup the dockspace for main viewport.

    Returns: dockspace_id
        dockspace_id: main dockspace id.
    """
    main_viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(main_viewport.work_pos)
    imgui.set_next_window_size(main_viewport.work_size)
    imgui.set_next_window_viewport(main_viewport.id_)
    imgui.push_style_var(imgui.StyleVar_.window_rounding.value, 0)
    imgui.push_style_var(imgui.StyleVar_.window_border_size.value, 0)
    dockspace_flag = imgui.DockNodeFlags_.passthru_central_node.value
    window_flags = imgui.WindowFlags_.no_nav_focus.value |\
        imgui.WindowFlags_.no_docking.value |\
        imgui.WindowFlags_.no_title_bar.value |\
        imgui.WindowFlags_.no_resize.value |\
        imgui.WindowFlags_.no_move.value |\
        imgui.WindowFlags_.no_collapse.value |\
        imgui.WindowFlags_.no_background.value
    with imgui_ctx.begin("root", flags=window_flags):
        imgui.pop_style_var(2)
        dockspace_id = imgui.dock_space(
            imgui.get_id(name),
            flags=dockspace_flag
        )
    return dockspace_id
