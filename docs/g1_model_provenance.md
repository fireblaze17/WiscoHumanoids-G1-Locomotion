# Unitree G1 Model Provenance

This project vendors the official Unitree `g1_29dof_rev_1_0` MuJoCo model and the exact 35 meshes referenced by that XML into `assets/g1/`. The local scene `assets/g1/g1_scene.xml` is a minimal include wrapper for flat-ground loading and viewing.

The official asset remains the immutable provenance/viewing source. Training uses separately named derived XML files documented below; runtime environment and PPO code live under `src/g1_locomotion/` and do not modify the official XML on disk.

## Sources Inspected

| Source | Repository | Inspected SHA | Files |
| --- | --- | --- | --- |
| Official Unitree RL Gym | `https://github.com/unitreerobotics/unitree_rl_gym` | `276801e46c5d433564f24658bac64f254b7d2d4b` | `resources/robots/g1_description/g1_29dof_rev_1_0.xml`, `meshes/`, `LICENSE` |
| Wisconsin humanoid reference | `https://github.com/Lavansurendra/berkeley_humanoid` | `ee0e37e78d751d3376dc71c27166d18460966a05` | `environment/g1_robot.xml`, `environment/g1_scene.xml`, `environment/g1_env.py`, `environment/robot_mesh_files/unitree_g1/` |
| Official Unitree G1 asset README | `https://github.com/unitreerobotics/unitree_lerobot` | `41c2805742de879ddab2d8d6beaeaf215f876395` | `unitree_lerobot/eval_robot/assets/g1/README.md` |

The Unitree asset README identifies `g1_29dof_rev_1_0` as the up-to-date 29-DOF revision 1.0 model, with `mode_machine=5`, `6*2` leg DOF, `3` waist DOF, `7*2` arm DOF, and no hand DOF.

## Vendored Files

- `assets/g1/g1_29dof_rev_1_0.xml`: official Unitree MuJoCo XML.
- `assets/g1/meshes/`: only the 35 mesh files referenced by that XML.
- `assets/g1/g1_scene.xml`: local minimal scene/include wrapper for the official 29-DOF model.
- `assets/g1/g1_29dof_rev_1_0_mjx_feetonly.xml`: MuJoCo Playground-derived MJX/Warp training model for the same 29-DOF G1, adapted to local meshes and simplified collision geoms.
- `assets/g1/g1_mjx_contact_sensors.xml`: local contact-sensor definitions included by the derived training model.
- `assets/g1/g1_scene_mjx_feetonly.xml`: local flat-ground training scene for the MJX/Warp feet-only model.
- `assets/g1/LICENSE.unitree_rl_gym`: Unitree source license notice.
- `assets/g1/LICENSE.mujoco_playground`: MuJoCo Playground-derived XML license/provenance notice.

The official Unitree `scene.xml` was not copied because it includes `g1_12dof.xml`, which is not the target robot.

The MJX/Warp training XML is derived from MuJoCo Playground's Apache-2.0 `g1_mjx_feetonly.xml` in the pinned Playground source. It is used for training stability because MuJoCo Playground's Warp locomotion tasks use simplified collision geoms rather than full visual mesh collision. The local copy restores `right_hip_roll_joint` to the official/Wisconsin range `[-2.9671, 0.5236]`; the upstream Playground feet-only XML had that one range mirrored. The official Unitree XML remains the provenance source for robot identity and viewing.

## File Hashes

| File | SHA256 |
| --- | --- |
| Official Unitree `g1_29dof_rev_1_0.xml` | `165fa7a5275745e45fa25e1bc78010c1055d81bb81c013b3c7bc8b9347a3b9c3` |
| Derived training `g1_29dof_rev_1_0_mjx_feetonly.xml` | `ad5f94d716155892d2bc15e7af9100642bfe6c2daef66874f52e7c6fd9c75472` |
| Derived sensors `g1_mjx_contact_sensors.xml` | `f8b5c9e5ddf5c18dfc31a721edef67856c8a16c82a046e7a8714c353b4b80221` |
| Official local scene `g1_scene.xml` | `8ad4ab7313c38b54c86995770961469e0b065b0af1e734fd77a7bda7d75a5e01` |
| Derived training scene `g1_scene_mjx_feetonly.xml` | `cf694e16fe64f2a125c55193195d0863fabe0ed962039d35c2456e1ec77eb128` |
| MuJoCo Playground license | `928823a8c388d908417e3f287a2f87ee0c19ffbb79f2e50e6afdbb2b174b4d4e` |
| Unitree RL Gym license | `aef6394ba1597725a68308167324e675f562e6606027404deb1b9da254c2b9c1` |
| Wisconsin `environment/g1_robot.xml` | `ab74063fb4ebe3eaf5a7a26d57c7e2095da4e7552b37ecd15b8f5dfd8784420b` |
| Wisconsin `environment/g1_scene.xml` | `5c01beb485bfb7f72d2d597267ee92debbbd4a7331af264a768163839945ba20` |

## Loaded Model Comparison

The official Unitree `g1_29dof_rev_1_0.xml` and the Wisconsin `g1_scene.xml` compile to the same robot structure for the checks relevant to this project:

| Field | Value |
| --- | --- |
| Root model name | `g1_29dof_rev_1_0` |
| `nq` | `36` |
| `nv` | `35` |
| `nu` | `29` |
| Actuated joint order | identical |
| Joint ranges | identical |
| Joint-level actuator force ranges | identical |
| Body masses/inertias | identical |
| Total body mass | `33.341142 kg` |
| Initial base position | `[0, 0, 0.793]` |
| Timestep | `0.002` |
| Common mesh files | 35 files, byte-identical |

Wisconsin's XML changes are simulation conveniences rather than robot-parameter changes:

- changes `meshdir`;
- adds `implicitfast`;
- changes motor actuators to position actuators;
- adds defaults, materials, and classes;
- adds foot sites plus foot touch/distance sensors;
- adds `stand` and `crouch` keyframes;
- splits the scene into `g1_scene.xml`;
- changes visual, camera, and skybox scene settings.

## Actuated Joint Order

Ranges are radians. Force ranges are the compiled MuJoCo joint actuator force ranges.

| # | Joint | Actuator | Joint range | Force range |
| --- | --- | --- | --- | --- |
| 1 | `left_hip_pitch_joint` | `left_hip_pitch_joint` | [-2.5307, 2.8798] | [-88, 88] |
| 2 | `left_hip_roll_joint` | `left_hip_roll_joint` | [-0.5236, 2.9671] | [-139, 139] |
| 3 | `left_hip_yaw_joint` | `left_hip_yaw_joint` | [-2.7576, 2.7576] | [-88, 88] |
| 4 | `left_knee_joint` | `left_knee_joint` | [-0.087267, 2.8798] | [-139, 139] |
| 5 | `left_ankle_pitch_joint` | `left_ankle_pitch_joint` | [-0.87267, 0.5236] | [-50, 50] |
| 6 | `left_ankle_roll_joint` | `left_ankle_roll_joint` | [-0.2618, 0.2618] | [-50, 50] |
| 7 | `right_hip_pitch_joint` | `right_hip_pitch_joint` | [-2.5307, 2.8798] | [-88, 88] |
| 8 | `right_hip_roll_joint` | `right_hip_roll_joint` | [-2.9671, 0.5236] | [-139, 139] |
| 9 | `right_hip_yaw_joint` | `right_hip_yaw_joint` | [-2.7576, 2.7576] | [-88, 88] |
| 10 | `right_knee_joint` | `right_knee_joint` | [-0.087267, 2.8798] | [-139, 139] |
| 11 | `right_ankle_pitch_joint` | `right_ankle_pitch_joint` | [-0.87267, 0.5236] | [-50, 50] |
| 12 | `right_ankle_roll_joint` | `right_ankle_roll_joint` | [-0.2618, 0.2618] | [-50, 50] |
| 13 | `waist_yaw_joint` | `waist_yaw_joint` | [-2.618, 2.618] | [-88, 88] |
| 14 | `waist_roll_joint` | `waist_roll_joint` | [-0.52, 0.52] | [-50, 50] |
| 15 | `waist_pitch_joint` | `waist_pitch_joint` | [-0.52, 0.52] | [-50, 50] |
| 16 | `left_shoulder_pitch_joint` | `left_shoulder_pitch_joint` | [-3.0892, 2.6704] | [-25, 25] |
| 17 | `left_shoulder_roll_joint` | `left_shoulder_roll_joint` | [-1.5882, 2.2515] | [-25, 25] |
| 18 | `left_shoulder_yaw_joint` | `left_shoulder_yaw_joint` | [-2.618, 2.618] | [-25, 25] |
| 19 | `left_elbow_joint` | `left_elbow_joint` | [-1.0472, 2.0944] | [-25, 25] |
| 20 | `left_wrist_roll_joint` | `left_wrist_roll_joint` | [-1.9722, 1.9722] | [-25, 25] |
| 21 | `left_wrist_pitch_joint` | `left_wrist_pitch_joint` | [-1.6144, 1.6144] | [-5, 5] |
| 22 | `left_wrist_yaw_joint` | `left_wrist_yaw_joint` | [-1.6144, 1.6144] | [-5, 5] |
| 23 | `right_shoulder_pitch_joint` | `right_shoulder_pitch_joint` | [-3.0892, 2.6704] | [-25, 25] |
| 24 | `right_shoulder_roll_joint` | `right_shoulder_roll_joint` | [-2.2515, 1.5882] | [-25, 25] |
| 25 | `right_shoulder_yaw_joint` | `right_shoulder_yaw_joint` | [-2.618, 2.618] | [-25, 25] |
| 26 | `right_elbow_joint` | `right_elbow_joint` | [-1.0472, 2.0944] | [-25, 25] |
| 27 | `right_wrist_roll_joint` | `right_wrist_roll_joint` | [-1.9722, 1.9722] | [-25, 25] |
| 28 | `right_wrist_pitch_joint` | `right_wrist_pitch_joint` | [-1.6144, 1.6144] | [-5, 5] |
| 29 | `right_wrist_yaw_joint` | `right_wrist_yaw_joint` | [-1.6144, 1.6144] | [-5, 5] |

## Meshes Vendored

- `head_link.STL`
- `left_ankle_pitch_link.STL`
- `left_ankle_roll_link.STL`
- `left_elbow_link.STL`
- `left_hip_pitch_link.STL`
- `left_hip_roll_link.STL`
- `left_hip_yaw_link.STL`
- `left_knee_link.STL`
- `left_rubber_hand.STL`
- `left_shoulder_pitch_link.STL`
- `left_shoulder_roll_link.STL`
- `left_shoulder_yaw_link.STL`
- `left_wrist_pitch_link.STL`
- `left_wrist_roll_link.STL`
- `left_wrist_yaw_link.STL`
- `logo_link.STL`
- `pelvis.STL`
- `pelvis_contour_link.STL`
- `right_ankle_pitch_link.STL`
- `right_ankle_roll_link.STL`
- `right_elbow_link.STL`
- `right_hip_pitch_link.STL`
- `right_hip_roll_link.STL`
- `right_hip_yaw_link.STL`
- `right_knee_link.STL`
- `right_rubber_hand.STL`
- `right_shoulder_pitch_link.STL`
- `right_shoulder_roll_link.STL`
- `right_shoulder_yaw_link.STL`
- `right_wrist_pitch_link.STL`
- `right_wrist_roll_link.STL`
- `right_wrist_yaw_link.STL`
- `torso_link_rev_1_0.STL`
- `waist_roll_link_rev_1_0.STL`
- `waist_yaw_link_rev_1_0.STL`
