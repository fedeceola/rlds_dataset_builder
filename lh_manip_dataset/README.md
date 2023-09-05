## Long Horizon Manipulation Dataset

This dataset consists of 200 successful manipulation trajectories that solve every day long horizon task in a tabletop cluttered setting. The dataset consists on 10 variations of 20 different tasks performed by a Franka Panda robot. All the episodes are collected via teleoperation relying on an OptiTrack system.

For each step the following information is provided:

* Robot State: 3x end-effector position (x, y, z) w.r.t. root_frame, 4x end-effector orientation quaternions (x, y, z, w) w.r.t. root_frame, 7x robot joint angles, 2x gripper position in (0, 0.0404), 7x robot joint velocities.

* Observation: 1x external camera images, 1x wrist camera image.

* Action: 3x end-effector position offset (x, y, z) w.r.t. root_frame, 4x end-effector orientation quaternions offset (x, y, z, w) w.r.t. root_frame, 1x desired gripper opening offset.

* Language instruction.
