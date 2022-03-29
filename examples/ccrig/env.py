from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in

def default_multiobj_env():
    x_var = 0.2 # determines size of the workspace
    x_low = -x_var
    x_high = x_var
    y_low = 0.5
    y_high = 0.7
    t = 0.05 # border of the workspace for the hand
    params = dict(
        num_objects=1,
        object_meshes=None,
        fixed_start=True,
        num_scene_objects=[1],
        maxlen=0.1,
        action_repeat=1,
        puck_goal_low=(x_low + 0.01, y_low + 0.01),
        puck_goal_high=(x_high - 0.01, y_high - 0.01),
        hand_goal_low=(x_low + 3*t, y_low + t),
        hand_goal_high=(x_high - 3*t, y_high -t),
        mocap_low=(x_low + 2*t, y_low , 0.0),
        mocap_high=(x_high - 2*t, y_high, 0.5),
        object_low=(x_low + 0.01, y_low + 0.01, 0.02),
        object_high=(x_high - 0.01, y_high - 0.01, 0.02),
        use_textures=True,
        init_camera=sawyer_init_camera_zoomed_in,
        cylinder_radius=0.05,
    )
    return SawyerMultiobjectEnv, params
