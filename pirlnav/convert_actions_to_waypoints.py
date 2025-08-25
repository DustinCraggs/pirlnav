from enum import Enum
import numpy as np
import strictfire
import zarr


class HabitatSimActions(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5


MOVE_FORWARD_METERS = 0.25
WAYPOINT_TURN_MOVEMENT_METERS = 0.05
TURN_ANGLE_DEGREES = np.radians(30)


def add_waypoint_actions(zarr_path):
    data = zarr.open(zarr_path, mode="r")["data"]
    actions = data["next_actions"]
    dones = data["done"]

    ep_num = 0
    waypoints = []
    simple_waypoints = []

    for action, done in zip(actions, dones):
        action = HabitatSimActions(action)
        if done:
            # Plot waypoints:
            import matplotlib.pyplot as plt

            if ep_num > 0:
                waypoints = np.array(waypoints)
                simple_waypoints = np.array(simple_waypoints)

                plt.plot(
                    waypoints[:, 0], waypoints[:, 1], label="waypoints", marker="."
                )
                plt.plot(
                    simple_waypoints[:, 0],
                    simple_waypoints[:, 1],
                    label="simple_waypoints",
                    marker=".",
                )
                plt.legend()
                plt.show()

            waypoints = []
            simple_waypoints = []

            # This is the start step of a new episode.
            base_pose = (0, 0, 0)
            current_pose = (0, 0, 0)
            current_simple_pose = (0, 0, 0)

            ep_num += 1

        waypoints.append(current_pose)
        simple_waypoints.append(current_simple_pose)

        current_pose, base_pose = get_next_waypoint_with_yaw_compensation(
            action,
            current_pose,
            base_pose,
        )
        current_simple_pose = get_next_simple_waypoint(action, current_simple_pose)

        if ep_num == 3:
            break

    waypoints.append(current_pose)
    simple_waypoints.append(current_simple_pose)


def get_next_simple_waypoint(action, current_pose):
    if action == HabitatSimActions.STOP:
        return current_pose

    x, y, yaw = current_pose
    # Movement:
    if action == HabitatSimActions.MOVE_FORWARD:
        x += MOVE_FORWARD_METERS * np.cos(yaw)
        y += MOVE_FORWARD_METERS * np.sin(yaw)
        return (x, y, yaw)

    # Rotation:
    rotation_direction = 1 if action == HabitatSimActions.TURN_LEFT else -1
    yaw += rotation_direction * TURN_ANGLE_DEGREES
    return (x, y, yaw)


def get_next_waypoint_with_yaw_compensation(action, current_pose, base_pose):
    if action == HabitatSimActions.STOP:
        return current_pose, current_pose

    # Movement:
    if action == HabitatSimActions.MOVE_FORWARD:
        # Use the base_pose, as if the previous action was a rotation current_pose will
        # have deviated:
        _, _, current_yaw = current_pose
        base_x, base_y, _ = base_pose
        final_x = base_x + MOVE_FORWARD_METERS * np.cos(current_yaw)
        final_y = base_y + MOVE_FORWARD_METERS * np.sin(current_yaw)

        # Intermediate pose is half way between the base_pose and final pose:
        # x, y, yaw = base_pose
        # mid_x = (x + final_x) / 2
        # mid_y = (y + final_y) / 2
        # return (mid_x, mid_y, yaw), (final_x, final_y, yaw)
        # Return new pose and base pose:
        new_pose = (final_x, final_y, current_yaw)
        return new_pose, new_pose

    # Rotation:
    rotation_direction = 1 if action == HabitatSimActions.TURN_LEFT else -1
    x, y, yaw = current_pose
    yaw += rotation_direction * TURN_ANGLE_DEGREES
    x += WAYPOINT_TURN_MOVEMENT_METERS * np.cos(yaw)
    y += WAYPOINT_TURN_MOVEMENT_METERS * np.sin(yaw)

    # Return new pose and base pose. The base pose is the pose prior to the rotation:
    return (x, y, yaw), base_pose


if __name__ == "__main__":
    strictfire.StrictFire(add_waypoint_actions)
