"""Creating a link collision model for the Franka with a series of spheres of various radii"""

# All positions are in link frame, not link COM frame

# Di dalam file mycobot_collision_model.py

# Posisi (x, y, z) dalam frame link lokal
# Radius dalam meter

# Link 1 (Base/Pundak)
link_1_pos = ((0, 0, 0.05),)
link_1_radii = (0.04,)

# Link 2 (Lengan Atas)
link_2_pos = ((0, 0, 0.0), (0, 0, -0.08),)
link_2_radii = (0.035, 0.035,)

# Link 3 (Siku)
link_3_pos = ((0, 0, 0.0), (0, 0, -0.03),)
link_3_radii = (0.03,) * 2

# Link 4 (Lengan Bawah)
link_4_pos = ((0, 0, 0.0),)
link_4_radii = (0.03,)

# Link 5 (Pergelangan Tangan)
link_5_pos = ((0, 0, -0.02),)
link_5_radii = (0.03,)

# Link 6 (Flange/Tangan)
link_6_pos = ((0, 0, 0.0),)
link_6_radii = (0.025,)

# TODO: Get rid of these dictionaries and just use lists
positions = {
    "link_1": link_1_pos,
    "link_2": link_2_pos,
    "link_3": link_3_pos,
    "link_4": link_4_pos,
    "link_5": link_5_pos,
    "link_6": link_6_pos,
}
radii = {
    "link_1": link_1_radii,
    "link_2": link_2_radii,
    "link_3": link_3_radii,
    "link_4": link_4_radii,
    "link_5": link_5_radii,
    "link_6": link_6_radii,
}

# Kumpulkan semua data
positions_list = (
    link_1_pos,
    link_2_pos,
    link_3_pos,
    link_4_pos,
    link_5_pos,
    link_6_pos,
)
radii_list = (
    link_1_radii,
    link_2_radii,
    link_3_radii,
    link_4_radii,
    link_5_radii,
    link_6_radii,
)

mycobot_collision_data = {"positions": positions_list, "radii": radii_list}