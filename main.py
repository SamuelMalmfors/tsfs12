import argparse
import csv
import math
import time
from pathlib import Path

import carla

from controllers.pid_controller import PIDController
from controllers.mpc_controller import MPCController
from controllers.presets import PID_PRESETS, MPC_PRESETS
from scenarios.lead_vehicle import get_lead_target_speed


# ============================================================
# SETTINGS
# ============================================================

DT = 0.05
ROAD_CHOICE = 1       # 0 = bästa vägen, 1 = näst bästa, 2 = tredje
STRAIGHT_START = 70.0

# Default duration per scenario [s]
DURATIONS = {
    "gentle": 35.0,
    "emergency": 30.0,
    "stop_and_go": 45.0,
}

MIN_DISTANCE = 8.0
TIME_GAP = 1.5
CRUISE_SPEED = 80.0 / 3.6

LEAD_MAX_ACCEL = 2.5
LEAD_MAX_BRAKE = 6.0

ROAD_STEP = 5.0
MAX_ROUTE_LENGTH = 1500.0
MIN_ROUTE_LENGTH = 150.0

PID_COLOR = "0,200,80"
MPC_COLOR = "0,130,255"
LEAD_COLOR = "240,240,240"


# ============================================================
# HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def get_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def get_longitudinal_accel(vehicle):
    a = vehicle.get_acceleration()
    forward = vehicle.get_transform().get_forward_vector()
    return a.x * forward.x + a.y * forward.y + a.z * forward.z


def get_time_gap(gap, speed):
    if speed < 1.0:
        return float("nan")
    return gap / speed


def get_gap(ego, lead):
    transform = ego.get_transform()
    forward = transform.get_forward_vector()

    a = transform.location
    b = lead.get_location()

    center_gap = (
        (b.x - a.x) * forward.x
        + (b.y - a.y) * forward.y
    )

    return center_gap - (
        ego.bounding_box.extent.x
        + lead.bounding_box.extent.x
    )


def angle_difference(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def same_direction(a, b):
    return angle_difference(
        a.transform.rotation.yaw,
        b.transform.rotation.yaw
    ) < 10.0


# ============================================================
# FIND PARALLEL LANES
# ============================================================


def find_parallel_lanes(road_map):
    """
    Find alternative parallel roads in Town04.

    Requirements:
    - Two adjacent driving lanes
    - Same driving direction
    - Straight starting section
    - Long route for both cars

    ROAD_CHOICE selects which road to use.
    """

    candidates = []
    seen = set()

    def valid_pair(a, b):
        if a is None or b is None:
            return False

        if a.is_junction or b.is_junction:
            return False

        if a.lane_type != carla.LaneType.Driving:
            return False

        if b.lane_type != carla.LaneType.Driving:
            return False

        # Require two different lanes in same direction
        if a.lane_id * b.lane_id <= 0:
            return False

        if not same_direction(a, b):
            return False

        distance = a.transform.location.distance(
            b.transform.location
        )

        return 2.5 <= distance <= 6.0

    for spawn in road_map.get_spawn_points():

        start_a = road_map.get_waypoint(
            spawn.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if start_a is None or start_a.is_junction:
            continue

        for start_b in (
            start_a.get_left_lane(),
            start_a.get_right_lane()
        ):

            if not valid_pair(start_a, start_b):
                continue

            if start_a.road_id != start_b.road_id:
                continue

            # Avoid duplicate starting positions
            pair_ids = tuple(sorted((
                start_a.lane_id,
                start_b.lane_id
            )))

            key = (
                start_a.road_id,
                pair_ids,
                round(start_a.s / 100)
            )

            if key in seen:
                continue

            seen.add(key)

            route_a = [start_a]
            route_b = [start_b]

            a = start_a
            b = start_b

            start_yaw = a.transform.rotation.yaw
            valid_start = True

            for i in range(
                int(MAX_ROUTE_LENGTH / ROAD_STEP)
            ):

                pairs = []

                for na in a.next(ROAD_STEP):
                    for nb in b.next(ROAD_STEP):

                        if not valid_pair(na, nb):
                            continue

                        # Must follow the same road section
                        if na.road_id != nb.road_id:
                            continue

                        # Must remain in separate lanes
                        if na.lane_id == nb.lane_id:
                            continue

                        # Check straight starting section
                        distance_travelled = (
                            i * ROAD_STEP
                        )

                        if distance_travelled < STRAIGHT_START:

                            if angle_difference(
                                na.transform.rotation.yaw,
                                start_yaw
                            ) > 3.0:
                                continue

                            if angle_difference(
                                nb.transform.rotation.yaw,
                                start_yaw
                            ) > 3.0:
                                continue

                        # Prevent sudden turns
                        if angle_difference(
                            a.transform.rotation.yaw,
                            na.transform.rotation.yaw
                        ) > 10.0:
                            continue

                        if angle_difference(
                            b.transform.rotation.yaw,
                            nb.transform.rotation.yaw
                        ) > 10.0:
                            continue

                        pairs.append((na, nb))

                if not pairs:
                    if i * ROAD_STEP < STRAIGHT_START:
                        valid_start = False
                    break

                # Prefer the pair continuing straight
                na, nb = min(
                    pairs,
                    key=lambda p: (
                        angle_difference(
                            a.transform.rotation.yaw,
                            p[0].transform.rotation.yaw
                        )
                        + angle_difference(
                            b.transform.rotation.yaw,
                            p[1].transform.rotation.yaw
                        )
                    )
                )

                route_a.append(na)
                route_b.append(nb)

                a = na
                b = nb

            length = (
                len(route_a) - 1
            ) * ROAD_STEP

            if valid_start and length >= 150.0:
                candidates.append({
                    "length": length,
                    "road_id": start_a.road_id,
                    "routes": (route_a, route_b)
                })

    if not candidates:
        raise RuntimeError(
            "No suitable parallel road found in Town04"
        )

    # Longest roads first
    candidates.sort(
        key=lambda c: c["length"],
        reverse=True
    )

    print("\nAvailable roads:")

    for i, candidate in enumerate(candidates[:10]):
        print(
            f"{i}: Road {candidate['road_id']} "
            f"- {candidate['length']:.0f} m"
        )

    if ROAD_CHOICE >= len(candidates):
        raise RuntimeError(
            f"ROAD_CHOICE={ROAD_CHOICE} unavailable. "
            f"Only {len(candidates)} candidates found."
        )

    selected = candidates[ROAD_CHOICE]

    print(
        f"\nSelected NEW road: "
        f"{selected['road_id']} "
        f"({selected['length']:.0f} m)"
    )

    return selected["routes"]


# ============================================================
# VEHICLES
# ============================================================

def spawn_vehicle(world, blueprint_name, waypoint, color):
    bp = world.get_blueprint_library().find(
        blueprint_name
    )

    if bp.has_attribute("color"):
        bp.set_attribute("color", color)

    t = waypoint.transform

    transform = carla.Transform(
        t.location + carla.Location(z=0.5),
        t.rotation
    )

    vehicle = world.try_spawn_actor(bp, transform)

    if vehicle is None:
        raise RuntimeError(
            "Failed to spawn vehicle"
        )

    return vehicle


def waypoint_ahead(waypoint, distance):
    """
    Waypoint exactly `distance` metres ahead in the same lane.
    """

    candidates = waypoint.next(distance)

    same_lane = [
        w for w in candidates
        if w.road_id == waypoint.road_id
        and w.lane_id == waypoint.lane_id
    ]

    if not (same_lane or candidates):
        raise RuntimeError("Selected route is too short")

    return (same_lane or candidates)[0]


def set_vehicle_speed(vehicle, target_speed):
    forward = vehicle.get_transform().get_forward_vector()

    vehicle.set_target_velocity(
        carla.Vector3D(
            x=forward.x * target_speed,
            y=forward.y * target_speed,
            z=0.0
        )
    )


def attach_collision_sensor(world, vehicle, flag):
    """
    flag is a dict; flag["hit"] is set to True on first collision.
    """

    bp = world.get_blueprint_library().find(
        "sensor.other.collision"
    )

    sensor = world.spawn_actor(
        bp,
        carla.Transform(),
        attach_to=vehicle
    )

    def on_collision(event):
        if not flag["hit"]:
            print(
                f"COLLISION: {flag['name']} hit "
                f"{event.other_actor.type_id}"
            )
        flag["hit"] = True

    sensor.listen(on_collision)

    return sensor


# ============================================================
# PURE PURSUIT STEERING
# ============================================================

def steering(vehicle, route, lookahead=15.0):
    """
    Follow the center of the assigned lane.

    Uses a lookahead point along the route.
    """

    transform = vehicle.get_transform()
    location = transform.location

    closest_index = min(
        range(len(route)),
        key=lambda i: (
            (route[i].transform.location.x - location.x)**2
            + (route[i].transform.location.y - location.y)**2
        )
    )

    target_index = min(
        closest_index + max(
            1,
            round(lookahead / ROAD_STEP)
        ),
        len(route) - 1
    )

    target = route[target_index].transform.location

    forward = transform.get_forward_vector()
    right = transform.get_right_vector()

    dx = target.x - location.x
    dy = target.y - location.y

    local_x = (
        dx * forward.x
        + dy * forward.y
    )

    local_y = (
        dx * right.x
        + dy * right.y
    )

    if local_x <= 0:
        return 0.0

    angle = math.atan2(
        local_y,
        local_x
    )

    steer = 1.2 * angle

    return clamp(
        steer,
        -0.3,
        0.3
    )


# ============================================================
# VISUALIZATION
# ============================================================

def hud_lines(scenario, t, args, lead_speed, pid_state, mpc_state):
    """
    Text for the corner panel in the viewer window.
    *_state: dict with speed, gap, desired_gap, min_gap, collision
    """

    def controller_lines(name, preset, state, color):
        status = "  COLLISION" if state["collision"] else ""
        time_gap = get_time_gap(state["gap"], state["speed"])
        time_gap_text = "  -  " if math.isnan(time_gap) else f"{time_gap:4.1f}"

        return [
            (f"{name} ({preset}){status}", (255, 80, 80) if status else color),
            (
                f"  {state['speed']*3.6:5.1f} km/h   "
                f"gap {state['gap']:5.1f} / {state['desired_gap']:4.1f} m   "
                f"{time_gap_text} s",
                color
            ),
            (f"  min gap {state['min_gap']:5.1f} m", (170, 170, 180)),
        ]

    return [
        (f"{scenario.replace('_', ' ')}   t = {t:4.1f} s", (235, 235, 235)),
        (f"Lead  {lead_speed*3.6:5.1f} km/h", (200, 200, 200)),
        ("", None),
        *controller_lines("PID", args.pid, pid_state, (60, 220, 110)),
        ("", None),
        *controller_lines("MPC", args.mpc, mpc_state, (90, 170, 255)),
    ]


def update_camera(world, ego_pid, ego_mpc):
    a = ego_pid.get_location()
    b = ego_mpc.get_location()

    center = carla.Location(
        x=(a.x + b.x) / 2.0,
        y=(a.y + b.y) / 2.0,
        z=(a.z + b.z) / 2.0
    )

    transform = ego_pid.get_transform()
    forward = transform.get_forward_vector()

    camera_location = carla.Location(
        x=center.x - 35.0 * forward.x,
        y=center.y - 35.0 * forward.y,
        z=center.z + 35.0
    )

    camera_transform = carla.Transform(
        camera_location,
        carla.Rotation(
            pitch=-40.0,
            yaw=transform.rotation.yaw
        )
    )

    world.get_spectator().set_transform(camera_transform)

    return camera_transform


# ============================================================
# LEAD VEHICLE SPEED
# ============================================================

def update_lead_speed(current, target, dt):
    change = clamp(
        target - current,
        -LEAD_MAX_BRAKE * dt,
        LEAD_MAX_ACCEL * dt
    )

    return max(
        0.0,
        current + change
    )


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PID vs MPC adaptive cruise control in CARLA"
    )

    parser.add_argument(
        "--scenario",
        choices=list(DURATIONS),
        default="stop_and_go"
    )

    parser.add_argument(
        "--pid",
        choices=list(PID_PRESETS),
        default="good",
        help="PID tuning preset"
    )

    parser.add_argument(
        "--mpc",
        choices=list(MPC_PRESETS),
        default="good",
        help="MPC tuning preset"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation length in seconds (default depends on scenario)"
    )

    parser.add_argument(
        "--record",
        action="store_true",
        help="Also save the viewer window as an mp4 video"
    )

    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Do not open the viewer window"
    )

    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run as fast as possible instead of real time"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    scenario = args.scenario
    duration = args.duration or DURATIONS[scenario]

    run_name = f"{scenario}_pid-{args.pid}_mpc-{args.mpc}"
    output_dir = Path("results") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(
        "localhost",
        2000
    )

    client.set_timeout(60.0)

    world = client.get_world()

    # Reloading the map is slow and can crash the server
    # on Windows, so only load Town04 when needed
    if not world.get_map().name.endswith("Town04"):
        print("Loading Town04...")
        world = client.load_world("Town04")

    original_settings = world.get_settings()

    # A crashed earlier run may have left the server in
    # synchronous mode with its cars still in the world
    if original_settings.synchronous_mode:
        original_settings.synchronous_mode = False
        original_settings.fixed_delta_seconds = None
        world.apply_settings(original_settings)

    leftovers = [
        actor for actor in world.get_actors()
        if actor.type_id.startswith(("vehicle.", "sensor."))
    ]

    if leftovers:
        print(f"Removing {len(leftovers)} leftover actors")
        client.apply_batch_sync([
            carla.command.DestroyActor(actor)
            for actor in leftovers
        ])

    vehicles = []
    sensors = []
    view = None

    # The log is written row by row so it survives a crash
    log_path = output_dir / "log.csv"
    log_file = log_path.open("w", newline="", encoding="utf-8")
    log_writer = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT

        world.apply_settings(settings)

        road_map = world.get_map()

        # Find two parallel lanes
        route_pid, route_mpc = find_parallel_lanes(
            road_map
        )

        # Start in steady state: all cars at the lead's
        # initial speed, at the desired following distance
        initial_speed = get_lead_target_speed(
            0.0,
            scenario
        )

        initial_gap = (
            MIN_DISTANCE
            + TIME_GAP * initial_speed
        )

        # ==========================================
        # SPAWN PID PAIR
        # ==========================================

        ego_pid = spawn_vehicle(
            world,
            "vehicle.tesla.model3",
            route_pid[0],
            PID_COLOR
        )
        vehicles.append(ego_pid)

        # Center-to-center distance giving the exact
        # bumper-to-bumper gap (both cars are the same model)
        lead_distance = (
            initial_gap
            + 2.0 * ego_pid.bounding_box.extent.x
        )

        lead_pid = spawn_vehicle(
            world,
            "vehicle.tesla.model3",
            waypoint_ahead(route_pid[0], lead_distance),
            LEAD_COLOR
        )
        vehicles.append(lead_pid)

        # ==========================================
        # SPAWN MPC PAIR
        # ==========================================

        ego_mpc = spawn_vehicle(
            world,
            "vehicle.tesla.model3",
            route_mpc[0],
            MPC_COLOR
        )
        vehicles.append(ego_mpc)

        lead_mpc = spawn_vehicle(
            world,
            "vehicle.tesla.model3",
            waypoint_ahead(route_mpc[0], lead_distance),
            LEAD_COLOR
        )
        vehicles.append(lead_mpc)

        # ==========================================
        # COLLISION SENSORS
        # ==========================================

        pid_collision = {"name": "PID", "hit": False}
        mpc_collision = {"name": "MPC", "hit": False}

        sensors.append(
            attach_collision_sensor(world, ego_pid, pid_collision)
        )
        sensors.append(
            attach_collision_sensor(world, ego_mpc, mpc_collision)
        )

        for vehicle in vehicles:
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=0.0,
                    steer=0.0
                )
            )

        # Let the cars settle on the road before
        # giving them their initial speed
        world.tick()

        for vehicle in vehicles:
            set_vehicle_speed(
                vehicle,
                initial_speed
            )

        world.tick()

        # ==========================================
        # CONTROLLERS
        # ==========================================

        pid = PIDController(
            dt=DT,
            **PID_PRESETS[args.pid]
        )

        mpc = MPCController(
            dt=0.1,
            min_distance=MIN_DISTANCE,
            time_gap=TIME_GAP,
            cruise_speed=CRUISE_SPEED,
            **MPC_PRESETS[args.mpc]
        )

        mpc_acceleration = 0.0
        lead_reference_speed = initial_speed

        print(
            f"Scenario: {scenario} | "
            f"PID: {args.pid} | MPC: {args.mpc}"
        )

        print(
            f"PID initial gap: "
            f"{get_gap(ego_pid, lead_pid):.1f} m"
        )

        print(
            f"MPC initial gap: "
            f"{get_gap(ego_mpc, lead_mpc):.1f} m"
        )

        if not args.no_view:
            from viewer import CameraView

            view = CameraView(
                world,
                DT,
                video_path=(
                    output_dir / "video.mp4" if args.record else None
                )
            )

        pid_min_gap = float("inf")
        mpc_min_gap = float("inf")

        # ==========================================
        # SIMULATION LOOP
        # ==========================================

        for step in range(
            int(duration / DT)
        ):
            loop_start = time.perf_counter()
            t = step * DT

            # ======================================
            # LEAD VEHICLES
            # ======================================

            target_speed = get_lead_target_speed(
                t,
                scenario
            )

            lead_reference_speed = update_lead_speed(
                lead_reference_speed,
                target_speed,
                DT
            )

            for lead, route in (
                (lead_pid, route_pid),
                (lead_mpc, route_mpc)
            ):
                set_vehicle_speed(
                    lead,
                    lead_reference_speed
                )

                lead.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        brake=0.0,
                        steer=steering(
                            lead,
                            route
                        )
                    )
                )

            # ======================================
            # PID CONTROLLER
            # ======================================

            pid_speed = get_speed(ego_pid)
            pid_lead_speed = get_speed(lead_pid)

            pid_gap = get_gap(
                ego_pid,
                lead_pid
            )

            pid_desired_gap = (
                MIN_DISTANCE
                + TIME_GAP * pid_speed
            )

            pid_error = (
                pid_gap
                - pid_desired_gap
            )

            pid_u = pid.update(
                pid_error
            )

            if pid_speed >= CRUISE_SPEED:
                pid_u = min(
                    pid_u,
                    0.0
                )

            pid_throttle = clamp(
                pid_u,
                0.0,
                1.0
            )

            pid_brake = clamp(
                -pid_u,
                0.0,
                1.0
            )

            ego_pid.apply_control(
                carla.VehicleControl(
                    throttle=pid_throttle,
                    brake=pid_brake,
                    steer=steering(
                        ego_pid,
                        route_pid
                    )
                )
            )

            # ======================================
            # MPC CONTROLLER
            # ======================================

            mpc_speed = get_speed(ego_mpc)
            mpc_lead_speed = get_speed(lead_mpc)

            mpc_gap = get_gap(
                ego_mpc,
                lead_mpc
            )

            mpc_desired_gap = (
                MIN_DISTANCE
                + TIME_GAP * mpc_speed
            )

            if step % 2 == 0:
                mpc_acceleration = mpc.update(
                    gap=mpc_gap,
                    ego_speed=mpc_speed,
                    lead_speed=mpc_lead_speed
                )
            # Speed limit: 80 km/h
            if mpc_speed >= CRUISE_SPEED:
                mpc_acceleration = min(mpc_acceleration, 0.0)

            if mpc_acceleration >= 0:
                mpc_throttle = clamp(
                    mpc_acceleration / 3.0,
                    0.0,
                    1.0
                )

                mpc_brake = 0.0

            else:
                mpc_throttle = 0.0

                mpc_brake = clamp(
                    -mpc_acceleration / 8.0,
                    0.0,
                    1.0
                )

            ego_mpc.apply_control(
                carla.VehicleControl(
                    throttle=mpc_throttle,
                    brake=mpc_brake,
                    steer=steering(
                        ego_mpc,
                        route_mpc
                    )
                )
            )

            # ======================================
            # CAMERA
            # ======================================

            camera_transform = update_camera(
                world,
                ego_pid,
                ego_mpc
            )

            if view is not None:
                view.set_transform(camera_transform)

            pid_min_gap = min(pid_min_gap, pid_gap)
            mpc_min_gap = min(mpc_min_gap, mpc_gap)

            # ======================================
            # LOGGING
            # ======================================

            row = {
                "t_s": round(t, 3),

                "lead_reference_kmh":
                    lead_reference_speed * 3.6,

                "pid_ego_kmh":
                    pid_speed * 3.6,

                "pid_lead_kmh":
                    pid_lead_speed * 3.6,

                "pid_gap_m":
                    pid_gap,

                "pid_desired_gap_m":
                    pid_desired_gap,

                "pid_time_gap_s":
                    get_time_gap(pid_gap, pid_speed),

                "pid_accel":
                    get_longitudinal_accel(ego_pid),

                "pid_lead_accel":
                    get_longitudinal_accel(lead_pid),

                "pid_throttle":
                    pid_throttle,

                "pid_brake":
                    pid_brake,

                "pid_collision":
                    int(pid_collision["hit"]),

                "mpc_ego_kmh":
                    mpc_speed * 3.6,

                "mpc_lead_kmh":
                    mpc_lead_speed * 3.6,

                "mpc_gap_m":
                    mpc_gap,

                "mpc_desired_gap_m":
                    mpc_desired_gap,

                "mpc_time_gap_s":
                    get_time_gap(mpc_gap, mpc_speed),

                "mpc_accel":
                    get_longitudinal_accel(ego_mpc),

                "mpc_lead_accel":
                    get_longitudinal_accel(lead_mpc),

                "mpc_target_accel":
                    mpc_acceleration,

                "mpc_throttle":
                    mpc_throttle,

                "mpc_brake":
                    mpc_brake,

                "mpc_collision":
                    int(mpc_collision["hit"])
            }

            if log_writer is None:
                log_writer = csv.DictWriter(
                    log_file,
                    fieldnames=row.keys()
                )
                log_writer.writeheader()

            log_writer.writerow(row)
            log_file.flush()

            frame_id = world.tick()

            if view is not None:
                view.show(frame_id, hud_lines(
                    scenario,
                    t,
                    args,
                    lead_reference_speed,
                    {
                        "speed": pid_speed,
                        "gap": pid_gap,
                        "desired_gap": pid_desired_gap,
                        "min_gap": pid_min_gap,
                        "collision": pid_collision["hit"],
                    },
                    {
                        "speed": mpc_speed,
                        "gap": mpc_gap,
                        "desired_gap": mpc_desired_gap,
                        "min_gap": mpc_min_gap,
                        "collision": mpc_collision["hit"],
                    },
                ), labels=[
                    (
                        "PID",
                        ego_pid.get_location() + carla.Location(z=2.5),
                        (30, 170, 80)
                    ),
                    (
                        "MPC",
                        ego_mpc.get_location() + carla.Location(z=2.5),
                        (40, 120, 230)
                    ),
                ])

            if step % 20 == 0:
                print(
                    f"t={t:5.1f}s | "
                    f"PID gap={pid_gap:5.1f} m | "
                    f"MPC gap={mpc_gap:5.1f} m | "
                    f"Lead={lead_reference_speed*3.6:5.1f} km/h"
                )

            if not args.no_realtime:
                elapsed = (
                    time.perf_counter()
                    - loop_start
                )

                time.sleep(
                    max(
                        0.0,
                        DT - elapsed
                    )
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print("Cleaning up...")

        log_file.close()
        print(f"Saved results to {log_path}")

        if view is not None:
            try:
                view.close()
                if args.record:
                    print(f"Saved video to {output_dir / 'video.mp4'}")
            except Exception as exc:
                print(f"Viewer warning: {exc}")

        for sensor in sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()

        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()

        world.apply_settings(
            original_settings
        )


if __name__ == "__main__":
    main()
