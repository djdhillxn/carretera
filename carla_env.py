"""Gymnasium environment for state-based highway tactical control in CARLA.

The environment is deliberately responsible for every simulator tick. Tactical
policies choose a target speed or lane once per second; waypoint/PID control and
the shared car-following override execute that decision at 20 Hz.
"""

import collections
import json
import math
import queue
import random
import time
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np

try:
    import carla
except ImportError:
    carla = None


MAINTAIN = 0
ACCELERATE = 1
DECELERATE = 2
CHANGE_LEFT = 3
CHANGE_RIGHT = 4

ACTION_NAMES = {
    MAINTAIN: "MAINTAIN",
    ACCELERATE: "ACCELERATE",
    DECELERATE: "DECELERATE",
    CHANGE_LEFT: "CHANGE_LEFT",
    CHANGE_RIGHT: "CHANGE_RIGHT",
}


def calculate_reward_components(config, delta_distance_m, speed_kmh,
                                lane_change_rejected=False,
                                lane_change_accepted=False,
                                collision=False, offroad=False, stuck=False):
    """Pure reward calculation used by the environment and offline checks."""

    reward_config = config["reward"]
    environment_config = config["environment"]
    components = {
        "progress_reward": (
            reward_config["progress_weight"]
            * delta_distance_m
            / reward_config["progress_scale_m"]
        ),
        "speed_reward": (
            reward_config["speed_weight"]
            * np.clip(
                speed_kmh / environment_config["initial_target_speed_kmh"],
                0.0,
                1.2,
            )
        ),
        "lane_change_penalty": (
            reward_config["unsafe_lane_change_penalty"]
            if lane_change_rejected
            else 0.0
        ),
        "accepted_lane_change_cost": (
            reward_config["accepted_lane_change_cost"]
            if lane_change_accepted
            else 0.0
        ),
        "collision_penalty": (
            reward_config["collision_penalty"] if collision else 0.0
        ),
        "offroad_penalty": (
            reward_config["offroad_penalty"] if offroad else 0.0
        ),
        "stuck_penalty": (
            reward_config["stuck_penalty"] if stuck else 0.0
        ),
    }
    components["total_reward"] = float(sum(components.values()))
    return components


SAFE_PASSENGER_CAR_PATTERNS = [
    "vehicle.audi.a2",
    "vehicle.audi.etron",
    "vehicle.audi.tt",
    "vehicle.bmw.grandtourer",
    "vehicle.chevrolet.impala",
    "vehicle.citroen.c3",
    "vehicle.dodge.charger_2020",
    "vehicle.ford.mustang",
    "vehicle.linear.seater",
    "vehicle.mercedes.coupe",
    "vehicle.mini.cooper_s",
    "vehicle.nissan.micra",
    "vehicle.nissan.patrol",
    "vehicle.seat.leon",
    "vehicle.tesla.model3",
    "vehicle.toyota.prius",
    "vehicle.volkswagen.t2_2020",
]


def get_safe_vehicle_blueprints(blueprint_library):
    if blueprint_library is None:
        return []
    all_vehicles = sorted(blueprint_library.filter("vehicle.*"), key=lambda b: b.id)
    safe = []
    excluded_keywords = (
        "carlacola", "cybertruck", "ambulance", "firetruck", "police", "bus",
        "truck", "van", "isetta", "vespa", "harley", "yamaha", "kawasaki",
        "crossbike", "gazelle", "bh", "diamondback", "montreal"
    )
    for bp in all_vehicles:
        lowered = bp.id.lower()
        if bp.has_attribute("number_of_wheels"):
            try:
                if int(bp.get_attribute("number_of_wheels")) != 4:
                    continue
            except ValueError:
                pass
        if any(word in lowered for word in excluded_keywords):
            continue
        safe.append(bp)
    if not safe:
        for bp in all_vehicles:
            if any(pattern in bp.id.lower() for pattern in SAFE_PASSENGER_CAR_PATTERNS):
                safe.append(bp)
    return safe if safe else all_vehicles


class HighwayDecisionEnv(gym.Env):
    """Five-action, 20-observation Town04 highway environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config, mode="train", scenario=None, artifact_root=None,
                 render_mode=None):
        super().__init__()
        if mode not in ("train", "evaluate", "render"):
            raise ValueError("mode must be train, evaluate, or render")
        if mode in ("evaluate", "render") and scenario is None:
            raise ValueError("%s mode requires a frozen scenario row" % mode)
        if carla is None:
            raise RuntimeError(
                "The CARLA Python API is not installed. Install requirements.txt "
                "before creating HighwayDecisionEnv."
            )

        self.config = config
        self.mode = mode
        self.scenario = scenario
        self.render_mode = render_mode
        self.artifact_root = Path(
            artifact_root or config["paths"]["artifact_root"]
        ).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        self.action_space = gym.spaces.Discrete(5)
        lower = np.array(
            [0.0, 0.0, -2.0, -1.0, -1.0, 0.0, 0.0, 0.0]
            + [0.0, -1.0, 0.0, -1.0] * 3,
            dtype=np.float32,
        )
        upper = np.array(
            [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            + [1.0, 1.0, 1.0, 1.0] * 3,
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=lower, high=upper, dtype=np.float32
        )

        self.client = None
        self.world = None
        self.map = None
        self.traffic_manager = None
        self.original_world_settings = None
        self.owns_world_settings = False
        self.actors = []
        self.sensors = []
        self.ego = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.camera = None
        self.camera_queue = queue.Queue(maxsize=2)
        self.render_frames = collections.deque(
            maxlen=config["environment"]["action_repeat_ticks"] + 2
        )
        self.last_rgb_frame = None
        self.last_render_frame_id = None
        self.render_frames.clear()
        self.episode_token = 0
        self.closed = False
        self.highway_candidates = []
        self.selected_candidate = None
        self.current_state = None
        self.recorder_path = None
        try:
            self._connect_and_setup()
            self._discover_highway_candidates()
            preview = self.highway_candidates[0]
            print(
                "Discovered %s Town04 highway candidates; automatic default is "
                "candidate %s (spawn %s)."
                % (
                    len(self.highway_candidates),
                    preview["candidate_index"],
                    preview["carla_spawn_index"],
                )
            )
        except Exception:
            if self.world is not None:
                self.close()
            raise

    @staticmethod
    def synthetic_observation(state, neighbor_radius_m=80.0):
        """Construct the exact normalized vector from a structured state."""

        lane_features = []
        for lane_name in ("left", "current", "right"):
            lane = state["lanes"][lane_name]
            if not lane.get("available", False):
                lane_features.extend([1.0, 0.0, 1.0, 0.0])
                continue
            lane_features.extend(
                [
                    np.clip(
                        lane.get("front_gap_m", neighbor_radius_m)
                        / neighbor_radius_m,
                        0.0,
                        1.0,
                    ),
                    np.clip(
                        lane.get("front_relative_speed_mps", 0.0) / 30.0,
                        -1.0,
                        1.0,
                    ),
                    np.clip(
                        lane.get("rear_gap_m", neighbor_radius_m)
                        / neighbor_radius_m,
                        0.0,
                        1.0,
                    ),
                    np.clip(
                        lane.get("rear_relative_speed_mps", 0.0) / 30.0,
                        -1.0,
                        1.0,
                    ),
                ]
            )

        observation = np.array(
            [
                np.clip(state.get("speed_kmh", 0.0) / 100.0, 0.0, 1.0),
                np.clip(
                    state.get("target_speed_kmh", 0.0) / 100.0, 0.0, 1.0
                ),
                np.clip(state.get("lateral_offset_normalized", 0.0), -2.0, 2.0),
                np.clip(math.sin(state.get("heading_error_rad", 0.0)), -1.0, 1.0),
                np.clip(math.cos(state.get("heading_error_rad", 0.0)), -1.0, 1.0),
                float(state["lanes"]["left"].get("available", False)),
                float(state["lanes"]["right"].get("available", False)),
                float(state.get("lane_change_active", False)),
            ]
            + lane_features,
            dtype=np.float32,
        )
        if observation.shape != (20,) or not np.all(np.isfinite(observation)):
            raise RuntimeError("Observation construction did not produce 20 finite values.")
        return observation

    def _connect_and_setup(self):
        carla_config = self.config["carla"]
        self.client = carla.Client(carla_config["host"], carla_config["port"])
        self.client.set_timeout(carla_config["client_timeout_seconds"])
        try:
            client_version = self.client.get_client_version()
            server_version = self.client.get_server_version()
        except RuntimeError as exc:
            raise ConnectionError(
                "Could not reach CARLA at %s:%s. Start the packaged CARLA "
                "0.9.16 server or update CARLA_HOST/CARLA_PORT. Original error: %s"
                % (carla_config["host"], carla_config["port"], exc)
            ) from exc
        expected = str(carla_config["version"])
        if client_version != expected or server_version != expected:
            raise RuntimeError(
                "CARLA version mismatch: expected client/server %s, got client "
                "%s and server %s." % (expected, client_version, server_version)
            )

        self.world = self.client.get_world()
        expected_map = carla_config["map"]
        current_map = self.world.get_map().name.split("/")[-1]
        if current_map.lower() != expected_map.lower():
            try:
                self.world = self.client.load_world(expected_map, False)
            except TypeError:
                self.world = self.client.load_world(expected_map)
        self.map = self.world.get_map()
        self.original_world_settings = self.world.get_settings()
        self._apply_world_settings()
        self.traffic_manager = self.client.get_trafficmanager(
            carla_config["traffic_manager_port"]
        )
        try:
            self.original_tm_sync = self.traffic_manager.get_synchronous_mode()
        except Exception:
            self.original_tm_sync = False
        self.traffic_manager.set_synchronous_mode(True)
        self.tm_sync_enabled = True

    def _apply_world_settings(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = bool(self.config["carla"]["synchronous_mode"])
        settings.fixed_delta_seconds = self.config["carla"]["fixed_delta_seconds"]
        server_config = self.config["carla"]["server"]
        if self.mode == "render":
            settings.no_rendering_mode = server_config[
                "rendering_no_rendering_mode"
            ]
        else:
            settings.no_rendering_mode = server_config[
                "training_no_rendering_mode"
            ]
        self.world.apply_settings(settings)
        self.owns_world_settings = True

    def _clear_cached_references(self):
        self.highway_candidates = []
        self.selected_candidate = None
        self.current_state = None

    def reload_world(self):
        """Reload while preserving fixed-step settings and invalidating actor caches."""

        self._cleanup_actors()
        try:
            self.world = self.client.reload_world(False)
        except TypeError:
            self.world = self.client.reload_world()
        self.map = self.world.get_map()
        self._apply_world_settings()
        self.traffic_manager = self.client.get_trafficmanager(
            self.config["carla"]["traffic_manager_port"]
        )
        self.traffic_manager.set_synchronous_mode(True)
        self._clear_cached_references()
        self._discover_highway_candidates()

    def _same_direction(self, first, second):
        if first is None or second is None:
            return False
        first_forward = first.transform.get_forward_vector()
        second_forward = second.transform.get_forward_vector()
        return (
            first_forward.x * second_forward.x
            + first_forward.y * second_forward.y
            + first_forward.z * second_forward.z
        ) > 0.5

    def _driving_adjacent(self, waypoint, side):
        adjacent = (
            waypoint.get_left_lane() if side == "left" else waypoint.get_right_lane()
        )
        if adjacent is None:
            return None
        if adjacent.lane_type != carla.LaneType.Driving:
            return None
        if not self._same_direction(waypoint, adjacent):
            return None
        return adjacent

    def _choose_branch(self, current, branches, desired_lane_id=None):
        if not branches:
            return None
        current_forward = current.transform.get_forward_vector()

        def score(candidate):
            forward = candidate.transform.get_forward_vector()
            direction = current_forward.x * forward.x + current_forward.y * forward.y
            lane_match = 1.0 if desired_lane_id == candidate.lane_id else 0.0
            non_junction = 1.0 if not candidate.is_junction else 0.0
            return (non_junction, lane_match, direction, -abs(candidate.lane_id))

        return sorted(branches, key=score, reverse=True)[0]

    def _continuity(self, waypoint):
        spacing = max(5.0, self.config["controller"]["waypoint_spacing_m"] * 5.0)
        required = self.config["environment"]["highway_continuity_distance_m"]
        current = waypoint
        traveled = 0.0
        while traveled < required:
            next_waypoint = self._choose_branch(
                current, current.next(spacing), current.lane_id
            )
            if next_waypoint is None or next_waypoint.is_junction:
                break
            if next_waypoint.lane_type != carla.LaneType.Driving:
                break
            if not self._same_direction(current, next_waypoint):
                break
            traveled += current.transform.location.distance(
                next_waypoint.transform.location
            )
            current = next_waypoint
        return traveled

    def _spawn_availability_score(self, transform, spawn_points):
        count = 0
        forward = transform.get_forward_vector()
        for other in spawn_points:
            relative = other.location - transform.location
            longitudinal = relative.x * forward.x + relative.y * forward.y
            lateral = abs(relative.x * -forward.y + relative.y * forward.x)
            if abs(longitudinal) <= 150.0 and lateral <= 15.0:
                count += 1
        return count

    def _discover_highway_candidates(self):
        spawn_points = self.map.get_spawn_points()
        candidates = []
        for spawn_index, transform in enumerate(spawn_points):
            waypoint = self.map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None or waypoint.is_junction:
                continue
            left = self._driving_adjacent(waypoint, "left")
            right = self._driving_adjacent(waypoint, "right")
            if left is None and right is None:
                continue
            continuity = self._continuity(waypoint)
            required = self.config["environment"]["highway_continuity_distance_m"]
            if continuity < required:
                continue
            nearby_spawns = self._spawn_availability_score(transform, spawn_points)
            adjacent_count = int(left is not None) + int(right is not None)
            location = transform.location
            candidates.append(
                {
                    "carla_spawn_index": spawn_index,
                    "road_id": waypoint.road_id,
                    "lane_id": waypoint.lane_id,
                    "section_id": waypoint.section_id,
                    "location": {
                        "x": round(location.x, 4),
                        "y": round(location.y, 4),
                        "z": round(location.z, 4),
                    },
                    "yaw": round(transform.rotation.yaw, 4),
                    "left_lane_available": left is not None,
                    "right_lane_available": right is not None,
                    "continuity_score": round(continuity, 3),
                    "spawn_availability_score": nearby_spawns,
                    "_sort_score": (
                        adjacent_count,
                        round(continuity, 3),
                        nearby_spawns,
                        -spawn_index,
                    ),
                }
            )
        candidates.sort(key=lambda item: item["_sort_score"], reverse=True)
        for candidate_index, candidate in enumerate(candidates):
            candidate["candidate_index"] = candidate_index
            del candidate["_sort_score"]
        self.highway_candidates = candidates
        if not candidates:
            raise RuntimeError(
                "Town04 exposed no highway candidate with a legal adjacent lane and "
                "the configured continuity. Run doctor to inspect "
                "artifacts/logs/runtime/highway_candidates.json, then adjust only "
                "environment.highway_candidate_index after inspection."
            )

    def write_highway_candidates(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.highway_candidates, indent=2) + "\n", encoding="utf-8"
        )
        return path

    def _select_candidate(self):
        configured = self.config["environment"].get("highway_candidate_index")
        scenario_index = None
        if self.scenario:
            scenario_index = self.scenario.get("ego_highway_candidate_index")
        candidate_index = scenario_index if scenario_index is not None else configured
        if candidate_index is None:
            candidate_index = 0
        if candidate_index < 0 or candidate_index >= len(self.highway_candidates):
            raise IndexError(
                "Highway candidate index %s is invalid; doctor found %s candidates."
                % (candidate_index, len(self.highway_candidates))
            )
        self.selected_candidate = self.highway_candidates[candidate_index]
        print(
            "Selected Town04 highway candidate %s (spawn %s, road %s, lane %s)"
            % (
                candidate_index,
                self.selected_candidate["carla_spawn_index"],
                self.selected_candidate["road_id"],
                self.selected_candidate["lane_id"],
            )
        )

    def _set_weather(self, weather_name):
        if not hasattr(carla.WeatherParameters, weather_name):
            raise ValueError("Unknown CARLA weather preset %r" % weather_name)
        self.world.set_weather(getattr(carla.WeatherParameters, weather_name))

    def _episode_seed(self, seed):
        if seed is None:
            if self.scenario:
                seed = self.scenario["seed"]
            else:
                seed = self.config["project"]["default_seed"]
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.traffic_manager.set_random_device_seed(seed)
        return seed

    def _reset_episode_state(self):
        self.target_speed_kmh = self.config["environment"][
            "initial_target_speed_kmh"
        ]
        self.target_lane_key = None
        self.target_waypoint = None
        self.lane_change_active = False
        self.lane_change_completion_counter = 0
        self.collision = False
        self.collision_actor_type = ""
        self.lane_invasions = 0
        self.offroad_ticks = 0
        self.stuck_ticks = 0
        self.total_ticks = 0
        self.decision_steps = 0
        self.elapsed_seconds = 0.0
        self.traveled_distance_m = 0.0
        self.episode_return = 0.0
        self.lane_changes_accepted = 0
        self.lane_changes_completed = 0
        self.lane_change_aborted_count = 0
        self.lane_change_start_tick = 0
        self.lane_change_elapsed_ticks = 0
        self.lane_change_side = None
        self.unsafe_lane_changes = 0
        self.emergency_interventions = 0
        self.safety_override_active = False
        self.safety_override_activation_events = 0
        self.safety_override_active_ticks = 0
        self.safety_override_active_seconds = 0.0
        self.action_counts = {name: 0 for name in ACTION_NAMES.values()}
        self.speed_samples = []
        self.minimum_ttc_seconds = math.inf
        self.minimum_front_gap_m = math.inf
        self.offroad = False
        self.stuck = False
        self.timeout = False
        self.success = False
        self.last_location = None
        self.last_action_name = ACTION_NAMES[MAINTAIN]
        self.last_lane_change_rejected = False
        self.last_lane_change_accepted = False
        self.last_reward_components = {}
        self.reward_component_totals = collections.defaultdict(float)
        self.previous_steer = 0.0
        self.lateral_integral = 0.0
        self.lateral_previous_error = 0.0
        self.speed_integral = 0.0
        self.speed_previous_error = 0.0
        self.initial_lane_id = None
        self.actual_npc_count = 0
        self.requested_npc_count = 0
        self.last_rgb_frame = None
        while not self.camera_queue.empty():
            try:
                self.camera_queue.get_nowait()
            except queue.Empty:
                break

    def _blueprint(self, blueprint_filter, role_name):
        library = self.world.get_blueprint_library()
        matches = library.filter(blueprint_filter)
        if not matches:
            raise RuntimeError("No CARLA blueprint matched %r" % blueprint_filter)
        blueprint = sorted(matches, key=lambda item: item.id)[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role_name)
        return blueprint

    def _spawn_ego(self):
        spawn_points = self.map.get_spawn_points()
        transform = spawn_points[self.selected_candidate["carla_spawn_index"]]
        transform.location.z += 0.2
        blueprint = self._blueprint(
            self.config["environment"]["ego_vehicle_filter"], "hero"
        )
        self.ego = self.world.try_spawn_actor(blueprint, transform)
        if self.ego is None:
            raise RuntimeError(
                "Could not spawn ego at selected highway candidate. Ensure no stale "
                "actors are occupying the spawn and rerun."
            )
        self.actors.append(self.ego)
        waypoint = self.map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        self.initial_lane_id = waypoint.lane_id
        self.target_waypoint = waypoint

    def _planned_traffic(self):
        density = (
            self.scenario["traffic_density"]
            if self.scenario
            else str(self.rng.choice(self.config["traffic"]["training_density_choices"]))
        )
        count = int(self.config["traffic"]["density_counts"][density])
        plan = {
            "traffic_density": density,
            "requested_npc_count": count,
            "lead_distance_m": float(
                self.rng.uniform(*self.config["traffic"]["lead_distance_range_m"])
            ),
            "lead_speed_kmh": float(
                self.rng.uniform(*self.config["traffic"]["lead_speed_range_kmh"])
            ),
            "npc_spawn_indices": [],
            "npc_blueprints": [],
            "npc_desired_speeds_kmh": [],
            "npc_following_distances_m": [],
            "npc_auto_lane_change": self.config["traffic"]["npc_auto_lane_change"],
        }
        if self.scenario:
            for key in plan:
                if key in self.scenario:
                    plan[key] = self.scenario[key]
        return plan

    def _compatible_background_spawns(self):
        ego_location = self.ego.get_location()
        radius = self.config["traffic"]["background_spawn_radius_m"]
        options = []
        for spawn_index, transform in enumerate(self.map.get_spawn_points()):
            if spawn_index == self.selected_candidate["carla_spawn_index"]:
                continue
            distance = transform.location.distance(ego_location)
            if distance > radius or distance < 12.0:
                continue
            waypoint = self.map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None or waypoint.is_junction:
                continue
            if not self._same_direction(self.target_waypoint, waypoint):
                continue
            options.append((spawn_index, transform))
        options.sort(key=lambda item: item[0])
        return options

    def _set_npc_traffic_manager(self, actor, speed_kmh, following_distance,
                                 auto_lane_change):
        actor.set_autopilot(True, self.config["carla"]["traffic_manager_port"])
        if hasattr(self.traffic_manager, "set_desired_speed"):
            self.traffic_manager.set_desired_speed(actor, float(speed_kmh))
        else:
            speed_limit = max(actor.get_speed_limit(), 1.0)
            percentage = 100.0 * (1.0 - float(speed_kmh) / speed_limit)
            self.traffic_manager.vehicle_percentage_speed_difference(
                actor, percentage
            )
        self.traffic_manager.distance_to_leading_vehicle(
            actor, float(following_distance)
        )
        self.traffic_manager.auto_lane_change(actor, bool(auto_lane_change))

    def _spawn_structured_lead(self, plan):
        if not self.config["traffic"]["structured_lead_vehicle"]:
            return 0
        branches = self.target_waypoint.next(float(plan["lead_distance_m"]))
        lead_waypoint = self._choose_branch(
            self.target_waypoint, branches, self.target_waypoint.lane_id
        )
        if lead_waypoint is None:
            warnings.warn("Structured lead waypoint was unavailable.", RuntimeWarning)
            return 0
        transform = lead_waypoint.transform
        transform.location.z += 0.2
        blueprint_library = self.world.get_blueprint_library()
        vehicles = get_safe_vehicle_blueprints(blueprint_library)
        blueprint = vehicles[0]
        if lead_waypoint is None:
            warnings.warn("Structured lead waypoint was unavailable.", RuntimeWarning)
            return 0
        transform = lead_waypoint.transform
        transform.location.z += 0.2
        blueprint_library = self.world.get_blueprint_library()
        vehicles = get_safe_vehicle_blueprints(blueprint_library)
        blueprint = vehicles[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            warnings.warn(
                "CARLA rejected the planned structured lead spawn.", RuntimeWarning
            )
            return 0
        self.actors.append(actor)
        following = self.config["traffic"]["npc_following_distance_range_m"][0]
        self._set_npc_traffic_manager(
            actor,
            plan["lead_speed_kmh"],
            following,
            False,
        )
        return 1

    def _spawn_background_traffic(self, plan, remaining):
        if remaining <= 0:
            return 0
        options = self._compatible_background_spawns()
        option_lookup = {item[0]: item for item in options}
        planned_indices = list(plan.get("npc_spawn_indices", []))
        if planned_indices:
            selected = [
                option_lookup[index]
                for index in planned_indices
                if index in option_lookup
            ][:remaining]
        else:
            order = self.rng.permutation(len(options))
            selected = [options[index] for index in order[:remaining]]

        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = get_safe_vehicle_blueprints(blueprint_library)
        commands = []
        metadata = []
        for position, (spawn_index, transform) in enumerate(selected):
            configured_ids = plan.get("npc_blueprints", [])
            if position < len(configured_ids):
                bp_id = configured_ids[position]
                try:
                    blueprint = blueprint_library.find(bp_id)
                except (IndexError, RuntimeError):
                    raise RuntimeError("Blueprint %s in scenario plan is missing in CARLA." % bp_id)
            else:
                blueprint = vehicle_blueprints[
                    int(self.rng.integers(0, len(vehicle_blueprints)))
                ]
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", "autopilot")
            spawn_transform = carla.Transform(transform.location, transform.rotation)
            spawn_transform.location.z += 0.2
            command = carla.command.SpawnActor(blueprint, spawn_transform).then(
                carla.command.SetAutopilot(
                    carla.command.FutureActor,
                    True,
                    self.config["carla"]["traffic_manager_port"],
                )
            )
            commands.append(command)
            metadata.append((spawn_index, blueprint.id))
        if not commands:
            return 0
        responses = self.client.apply_batch_sync(commands, False)
        spawned = 0
        for position, response in enumerate(responses):
            if response.error:
                warnings.warn(
                    "CARLA rejected planned NPC spawn %s: %s"
                    % (metadata[position][0], response.error),
                    RuntimeWarning,
                )
                continue
            actor = self.world.get_actor(response.actor_id)
            if actor is None:
                warnings.warn(
                    "Spawned NPC %s could not be reacquired." % response.actor_id,
                    RuntimeWarning,
                )
                continue
            self.actors.append(actor)
            speeds = plan.get("npc_desired_speeds_kmh", [])
            distances = plan.get("npc_following_distances_m", [])
            speed = (
                speeds[position]
                if position < len(speeds)
                else self.rng.uniform(*self.config["traffic"]["npc_speed_range_kmh"])
            )
            following = (
                distances[position]
                if position < len(distances)
                else self.rng.uniform(
                    *self.config["traffic"]["npc_following_distance_range_m"]
                )
            )
            self._set_npc_traffic_manager(
                actor, speed, following, plan["npc_auto_lane_change"]
            )
            spawned += 1
        return spawned

    def _spawn_traffic(self):
        plan = self._planned_traffic()
        self.traffic_density = plan["traffic_density"]
        self.requested_npc_count = int(plan["requested_npc_count"])
        lead_count = self._spawn_structured_lead(plan)
        background_count = self._spawn_background_traffic(
            plan, self.requested_npc_count - lead_count
        )
        self.actual_npc_count = lead_count + background_count
        minimum = min(
            self.config["environment"]["minimum_actual_npcs"],
            self.requested_npc_count,
        )
        if self.actual_npc_count < minimum:
            raise RuntimeError(
                "Only %s of %s requested NPCs spawned; at least %s are required "
                "for a meaningful episode."
                % (self.actual_npc_count, self.requested_npc_count, minimum)
            )

    def _bounded_put(self, destination, item):
        try:
            destination.put_nowait(item)
        except queue.Full:
            try:
                destination.get_nowait()
            except queue.Empty:
                return
            destination.put_nowait(item)

    def _attach_sensors(self):
        token = self.episode_token
        library = self.world.get_blueprint_library()
        collision_blueprint = library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_blueprint, carla.Transform(), attach_to=self.ego
        )

        def on_collision(event):
            if token != self.episode_token:
                return
            self.collision = True
            other = getattr(event, "other_actor", None)
            self.collision_actor_type = other.type_id if other else "unknown"

        self.collision_sensor.listen(on_collision)
        self.sensors.append(self.collision_sensor)

        lane_blueprint = library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(
            lane_blueprint, carla.Transform(), attach_to=self.ego
        )

        def on_lane_invasion(event):
            if token == self.episode_token:
                self.lane_invasions += 1

        self.lane_sensor.listen(on_lane_invasion)
        self.sensors.append(self.lane_sensor)

        if self.mode == "render":
            video_config = self.config["video"]
            camera_blueprint = library.find("sensor.camera.rgb")
            camera_blueprint.set_attribute("image_size_x", str(video_config["width"]))
            camera_blueprint.set_attribute("image_size_y", str(video_config["height"]))
            camera_blueprint.set_attribute("fov", str(video_config["field_of_view"]))
            camera_blueprint.set_attribute(
                "sensor_tick", str(self.config["carla"]["fixed_delta_seconds"])
            )
            transform = carla.Transform(
                carla.Location(
                    x=video_config["camera_x"],
                    y=video_config["camera_y"],
                    z=video_config["camera_z"],
                ),
                carla.Rotation(pitch=video_config["camera_pitch"]),
            )
            self.camera = self.world.spawn_actor(
                camera_blueprint,
                transform,
                attach_to=self.ego,
                attachment_type=carla.AttachmentType.SpringArmGhost,
            )

            def on_camera(image):
                if token == self.episode_token:
                    self._bounded_put(self.camera_queue, image)

            self.camera.listen(on_camera)
            self.sensors.append(self.camera)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.closed:
            raise RuntimeError("This environment is closed; create a fresh instance.")
        self.episode_token += 1
        self.action_space.seed(seed)
        self._cleanup_actors()
        self._reset_episode_state()
        self.seed_value = self._episode_seed(seed)
        self._select_candidate()
        weather = (
            self.scenario["weather"]
            if self.scenario
            else self.config["traffic"]["training_weather"]
        )
        self.weather_name = weather
        self._set_weather(weather)
        try:
            self._spawn_ego()
            self._spawn_traffic()
            self._attach_sensors()
            self.world.tick()
            self.last_location = self.ego.get_location()
            self.current_state = self._build_state()
            observation = self._normalize_observation(self.current_state)
            return observation, self._step_info()
        except Exception:
            self._cleanup_actors()
            raise

    def _vehicle_speed_mps(self, actor):
        velocity = actor.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def _vehicle_speed_kmh(self, actor):
        return 3.6 * self._vehicle_speed_mps(actor)

    def _lane_waypoints(self, current_waypoint):
        return {
            "left": self._driving_adjacent(current_waypoint, "left"),
            "current": current_waypoint,
            "right": self._driving_adjacent(current_waypoint, "right"),
        }

    def _blank_lane_state(self, available=False, waypoint=None):
        radius = self.config["environment"]["neighbor_radius_m"]
        return {
            "available": bool(available),
            "road_id": waypoint.road_id if waypoint else None,
            "lane_id": waypoint.lane_id if waypoint else None,
            "section_id": waypoint.section_id if waypoint else None,
            "front_gap_m": radius,
            "front_relative_speed_mps": 0.0,
            "front_speed_mps": 0.0,
            "rear_gap_m": radius,
            "rear_relative_speed_mps": 0.0,
            "rear_speed_mps": 0.0,
        }

    def _classify_neighbors(self, lane_waypoints, ego_speed_mps):
        radius = self.config["environment"]["neighbor_radius_m"]
        ego_location = self.ego.get_location()
        lane_states = {
            name: self._blank_lane_state(waypoint is not None, waypoint)
            for name, waypoint in lane_waypoints.items()
        }
        vehicles = sorted(
            self.world.get_actors().filter("vehicle.*"), key=lambda actor: actor.id
        )
        for actor in vehicles:
            if actor.id == self.ego.id:
                continue
            location = actor.get_location()
            if location.distance(ego_location) > radius:
                continue
            actor_waypoint = self.map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if actor_waypoint is None:
                continue
            actor_speed = self._vehicle_speed_mps(actor)
            best_match = None
            for lane_name in ("left", "current", "right"):
                lane_waypoint = lane_waypoints[lane_name]
                if lane_waypoint is None or not self._same_direction(
                    lane_waypoint, actor_waypoint
                ):
                    continue
                tangent = lane_waypoint.transform.get_forward_vector()
                relative = location - lane_waypoint.transform.location
                longitudinal = relative.x * tangent.x + relative.y * tangent.y
                # Positive lateral projection points to the lane tangent's left.
                lateral = relative.x * -tangent.y + relative.y * tangent.x
                direct = (
                    actor_waypoint.road_id == lane_waypoint.road_id
                    and actor_waypoint.lane_id == lane_waypoint.lane_id
                )
                geometric = (
                    abs(lateral) <= max(2.5, lane_waypoint.lane_width * 0.75)
                    and abs(longitudinal) <= radius
                )
                if not direct and not geometric:
                    continue
                match_score = (
                    1 if direct else 0,
                    -abs(lateral),
                    -abs(longitudinal),
                    -actor.id,
                )
                if best_match is None or match_score > best_match[0]:
                    best_match = (
                        match_score,
                        lane_name,
                        longitudinal,
                        actor_speed,
                    )
            if best_match is None:
                continue
            lane_name = best_match[1]
            longitudinal = best_match[2]
            actor_speed = best_match[3]
            gap = min(radius, abs(longitudinal))
            lane_state = lane_states[lane_name]
            if longitudinal >= 0.0 and gap < lane_state["front_gap_m"]:
                lane_state["front_gap_m"] = gap
                lane_state["front_relative_speed_mps"] = actor_speed - ego_speed_mps
                lane_state["front_speed_mps"] = actor_speed
            elif longitudinal < 0.0 and gap < lane_state["rear_gap_m"]:
                lane_state["rear_gap_m"] = gap
                lane_state["rear_relative_speed_mps"] = actor_speed - ego_speed_mps
                lane_state["rear_speed_mps"] = actor_speed
        return lane_states

    def _signed_lateral_offset(self, waypoint, location):
        tangent = waypoint.transform.get_forward_vector()
        relative = location - waypoint.transform.location
        return relative.x * -tangent.y + relative.y * tangent.x

    def _normalize_angle(self, value):
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    def _build_state(self):
        location = self.ego.get_location()
        current_waypoint = self.map.get_waypoint(
            location,
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        projected_waypoint = self.map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        lane_reference = current_waypoint or projected_waypoint
        if lane_reference is None:
            lane_reference = self.target_waypoint
        lane_waypoints = self._lane_waypoints(lane_reference)
        speed_mps = self._vehicle_speed_mps(self.ego)
        lanes = self._classify_neighbors(lane_waypoints, speed_mps)
        lateral_offset = self._signed_lateral_offset(lane_reference, location)
        ego_yaw = math.radians(self.ego.get_transform().rotation.yaw)
        lane_yaw = math.radians(lane_reference.transform.rotation.yaw)
        heading_error = self._normalize_angle(ego_yaw - lane_yaw)
        lane_width = max(lane_reference.lane_width, 0.1)
        state = {
            "speed_kmh": speed_mps * 3.6,
            "speed_mps": speed_mps,
            "target_speed_kmh": self.target_speed_kmh,
            "lateral_offset_m": lateral_offset,
            "lateral_offset_normalized": np.clip(
                lateral_offset / lane_width, -2.0, 2.0
            ),
            "heading_error_rad": heading_error,
            "lane_change_active": self.lane_change_active,
            "current_road_id": lane_reference.road_id,
            "current_lane_id": lane_reference.lane_id,
            "target_lane_id": (
                self.target_lane_key[1]
                if self.target_lane_key
                else lane_reference.lane_id
            ),
            "left_of_initial_lane": lane_reference.lane_id != self.initial_lane_id,
            "lanes": lanes,
        }
        return state

    def _normalize_observation(self, state):
        return self.synthetic_observation(
            state, self.config["environment"]["neighbor_radius_m"]
        )

    def _rear_ttc(self, lane):
        closing_speed = lane["rear_relative_speed_mps"]
        if closing_speed <= 0.0:
            return math.inf
        return lane["rear_gap_m"] / closing_speed

    def _validate_lane_change(self, action, state):
        side = "left" if action == CHANGE_LEFT else "right"
        lane = state["lanes"][side]
        if self.lane_change_active:
            return False, "lane_change_active"
        if not lane["available"]:
            return False, "illegal_lane"
        lane_config = self.config["lane_change"]
        if lane["front_gap_m"] < lane_config["minimum_front_gap_m"]:
            return False, "front_gap"
        if lane["rear_gap_m"] < lane_config["minimum_rear_gap_m"]:
            return False, "rear_gap"
        if self._rear_ttc(lane) < lane_config["minimum_rear_ttc_seconds"]:
            return False, "rear_ttc"
        return True, ""

    def _apply_tactical_action(self, action, state):
        if action not in ACTION_NAMES:
            raise ValueError("Action must be an integer from 0 through 4.")
        self.last_lane_change_rejected = False
        self.last_lane_change_accepted = False
        self.last_action_name = ACTION_NAMES[action]
        self.action_counts[self.last_action_name] += 1
        environment_config = self.config["environment"]
        if action == ACCELERATE:
            self.target_speed_kmh = min(
                environment_config["max_target_speed_kmh"],
                self.target_speed_kmh
                + environment_config["speed_action_delta_kmh"],
            )
        elif action == DECELERATE:
            self.target_speed_kmh = max(
                environment_config["min_target_speed_kmh"],
                self.target_speed_kmh
                - environment_config["speed_action_delta_kmh"],
            )
        elif action in (CHANGE_LEFT, CHANGE_RIGHT):
            accepted, reason = self._validate_lane_change(action, state)
            if not accepted:
                self.last_lane_change_rejected = True
                self.unsafe_lane_changes += 1
                return MAINTAIN, reason
            side = "left" if action == CHANGE_LEFT else "right"
            lane = state["lanes"][side]
            self.target_lane_key = (lane["road_id"], lane["lane_id"])
            current_waypoint = self.map.get_waypoint(
                self.ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            self.target_waypoint = self._driving_adjacent(current_waypoint, side)
            self.lane_change_side = side
            self.lane_change_active = True
            self.lane_change_elapsed_ticks = 0
            self.lane_change_completion_counter = 0
            self.lane_changes_accepted += 1
            self.last_lane_change_accepted = True
        return action, ""

    def _advance_target_waypoint(self, state):
        speed_mps = state["speed_mps"]
        controller = self.config["controller"]
        lookahead = np.clip(
            controller["min_lookahead_m"]
            + controller["lookahead_speed_factor"] * speed_mps,
            controller["min_lookahead_m"],
            controller["max_lookahead_m"],
        )
        if self.lane_change_active:
            lookahead = max(
                lookahead, self.config["environment"]["lane_change_lookahead_m"]
            )
        ego_waypoint = self.map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        reference = ego_waypoint
        if self.lane_change_active and self.target_lane_key:
            candidates = [
                ego_waypoint,
                self._driving_adjacent(ego_waypoint, "left"),
                self._driving_adjacent(ego_waypoint, "right"),
            ]
            matching = [
                candidate
                for candidate in candidates
                if candidate is not None
                and candidate.road_id == self.target_lane_key[0]
                and candidate.lane_id == self.target_lane_key[1]
            ]
            if matching:
                reference = matching[0]
            elif self.target_waypoint is not None:
                reference = self.target_waypoint
        desired_lane_id = (
            self.target_lane_key[1]
            if self.target_lane_key
            else reference.lane_id
        )
        branches = reference.next(float(lookahead))
        candidate = self._choose_branch(reference, branches, desired_lane_id)
        if candidate is None:
            raise RuntimeError("Waypoint controller reached a road with no continuation.")
        ego_transform = self.ego.get_transform()
        forward = ego_transform.get_forward_vector()
        relative = candidate.transform.location - ego_transform.location
        if relative.x * forward.x + relative.y * forward.y <= 0.0:
            branches = candidate.next(float(lookahead))
            candidate = self._choose_branch(candidate, branches, desired_lane_id)
            if candidate is None:
                raise RuntimeError("Waypoint lookahead produced only points behind ego.")
        self.target_waypoint = candidate
        return candidate

    def _pid(self, error, integral_name, previous_name, kp, ki, kd, dt):
        raw_integral = getattr(self, integral_name) + error * dt
        clipped_integral = float(np.clip(raw_integral, -50.0, 50.0))
        derivative = (error - getattr(self, previous_name)) / dt
        setattr(self, integral_name, clipped_integral)
        setattr(self, previous_name, error)
        return kp * error + ki * clipped_integral + kd * derivative

    def _following_override(self, state, effective_target_kmh):
        following = self.config["controller"]["following"]
        is_override = False
        if not following["enabled"]:
            if getattr(self, "safety_override_active", False):
                self.safety_override_active = False
            return effective_target_kmh, 0.0, False

        lane = state["lanes"]["current"]
        if self.lane_change_active and self.target_lane_key:
            for candidate in state["lanes"].values():
                if (
                    candidate["road_id"] == self.target_lane_key[0]
                    and candidate["lane_id"] == self.target_lane_key[1]
                ):
                    lane = candidate
                    break
        gap = lane["front_gap_m"]
        relative_speed = lane["front_relative_speed_mps"]
        closing_speed = max(0.0, -relative_speed)
        ttc = gap / closing_speed if closing_speed > 0.0 else math.inf
        desired_gap = max(
            following["minimum_gap_m"],
            following["time_headway_seconds"] * state["speed_mps"],
        )
        emergency = (
            gap < following["emergency_gap_m"]
            or ttc < following["emergency_ttc_seconds"]
        )
        proportional_brake = 0.0
        if gap < desired_gap:
            is_override = True
            front_speed_kmh = max(0.0, lane["front_speed_mps"] * 3.6)
            effective_target_kmh = min(effective_target_kmh, front_speed_kmh)
            proportional_brake = np.clip(
                (desired_gap - gap) / max(desired_gap, 0.1), 0.0, 1.0
            )
        if emergency:
            is_override = True
            proportional_brake = 1.0

        if is_override:
            self.safety_override_active_ticks += 1
            if not getattr(self, "safety_override_active", False):
                self.safety_override_activation_events += 1
                self.emergency_interventions += 1
        self.safety_override_active = is_override
        self.safety_override_active_seconds = (
            self.safety_override_active_ticks
            * self.config["carla"]["fixed_delta_seconds"]
        )
        return effective_target_kmh, proportional_brake, emergency

    def _low_level_control(self, state):
        target_waypoint = self._advance_target_waypoint(state)
        transform = self.ego.get_transform()
        relative = target_waypoint.transform.location - transform.location
        target_angle = math.atan2(relative.y, relative.x)
        ego_angle = math.radians(transform.rotation.yaw)
        angular_error = self._normalize_angle(target_angle - ego_angle)
        cross_track = self._signed_lateral_offset(
            target_waypoint, transform.location
        )
        lateral_error = angular_error - 0.05 * cross_track
        controller = self.config["controller"]
        dt = self.config["carla"]["fixed_delta_seconds"]
        steer = self._pid(
            lateral_error,
            "lateral_integral",
            "lateral_previous_error",
            controller["lateral_kp"],
            controller["lateral_ki"],
            controller["lateral_kd"],
            dt,
        )
        steer = float(np.clip(steer, -controller["max_steer"], controller["max_steer"]))
        steer = float(
            np.clip(
                steer,
                self.previous_steer - controller["max_steer_change_per_tick"],
                self.previous_steer + controller["max_steer_change_per_tick"],
            )
        )
        self.previous_steer = steer

        effective_target, override_brake, emergency = self._following_override(
            state, self.target_speed_kmh
        )
        speed_error = effective_target - state["speed_kmh"]
        longitudinal = self._pid(
            speed_error,
            "speed_integral",
            "speed_previous_error",
            controller["speed_kp"],
            controller["speed_ki"],
            controller["speed_kd"],
            dt,
        )
        throttle = float(np.clip(longitudinal, 0.0, controller["max_throttle"]))
        brake = float(np.clip(-longitudinal, 0.0, controller["max_brake"]))
        brake = max(brake, override_brake * controller["max_brake"])
        if emergency:
            throttle = 0.0
            brake = controller["max_brake"]
        elif brake > 0.0:
            throttle = 0.0
        control = carla.VehicleControl(
            throttle=throttle, steer=steer, brake=brake
        )
        self.ego.apply_control(control)

    def _abort_lane_change(self, reason):
        self.lane_change_active = False
        self.target_lane_key = None
        self.lane_change_side = None
        self.lane_change_completion_counter = 0
        self.lane_change_elapsed_ticks = 0
        self.lane_change_aborted_count += 1
        if self.ego is not None and self.map is not None:
            waypoint = self.map.get_waypoint(
                self.ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is not None:
                self.target_waypoint = waypoint

    def _update_lane_change_completion(self):
        if not self.lane_change_active or not self.target_lane_key:
            return
        self.lane_change_elapsed_ticks += 1
        max_duration = float(
            self.config["environment"].get("lane_change_max_duration_seconds", 6.0)
        )
        max_ticks = int(round(max_duration / self.config["carla"]["fixed_delta_seconds"]))

        waypoint = self.map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            self.lane_change_completion_counter = 0
            if self.lane_change_elapsed_ticks >= max_ticks:
                self._abort_lane_change("offroad_or_invalid_waypoint")
            return

        if getattr(self, "lane_change_side", None) is not None:
            adj = self._driving_adjacent(waypoint, self.lane_change_side)
            if adj is not None:
                self.target_lane_key = (adj.road_id, adj.lane_id)

        matches = (
            waypoint.road_id == self.target_lane_key[0]
            and waypoint.lane_id == self.target_lane_key[1]
        )
        offset = abs(self._signed_lateral_offset(waypoint, self.ego.get_location()))
        if (
            matches
            and offset
            < self.config["environment"]["lane_change_completion_offset_m"]
        ):
            self.lane_change_completion_counter += 1
        else:
            self.lane_change_completion_counter = 0

        if (
            self.lane_change_completion_counter
            >= self.config["environment"]["lane_change_completion_ticks"]
        ):
            self.lane_change_active = False
            self.target_lane_key = None
            self.lane_change_side = None
            self.target_waypoint = waypoint
            self.lane_change_completion_counter = 0
            self.lane_change_elapsed_ticks = 0
            self.lane_changes_completed += 1
        elif self.lane_change_elapsed_ticks >= max_ticks:
            self._abort_lane_change("timeout_exceeded")

    def _capture_render_frame(self, simulation_frame):
        if self.mode != "render":
            return
        deadline = time.monotonic() + 2.0
        selected = None
        while time.monotonic() < deadline:
            try:
                image = self.camera_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if image.frame < simulation_frame:
                continue
            if image.frame == simulation_frame:
                selected = image
            else:
                self._bounded_put(self.camera_queue, image)
            break
        if selected is None:
            raise RuntimeError(
                "RGB camera did not deliver simulator frame %s within 2 seconds."
                % simulation_frame
            )
        array = np.frombuffer(selected.raw_data, dtype=np.uint8)
        array = array.reshape((selected.height, selected.width, 4))
        self.last_rgb_frame = array[:, :, :3][:, :, ::-1].copy()
        self.last_render_frame_id = int(selected.frame)
        self.render_frames.append(self.last_rgb_frame)

    def _update_metrics_tick(self, state):
        location = self.ego.get_location()
        if self.last_location is not None:
            distance = location.distance(self.last_location)
            maximum_plausible = max(
                5.0,
                state["speed_mps"] * self.config["carla"]["fixed_delta_seconds"] * 3.0,
            )
            if distance <= maximum_plausible:
                self.traveled_distance_m += distance
            else:
                warnings.warn(
                    "Ignored implausible ego location jump of %.2f m." % distance,
                    RuntimeWarning,
                )
        self.last_location = location
        self.speed_samples.append(state["speed_kmh"])
        current_lane = state["lanes"]["current"]
        self.minimum_front_gap_m = min(
            self.minimum_front_gap_m, current_lane["front_gap_m"]
        )
        closing_speed = max(0.0, -current_lane["front_relative_speed_mps"])
        if closing_speed > 0.0:
            self.minimum_ttc_seconds = min(
                self.minimum_ttc_seconds,
                current_lane["front_gap_m"] / closing_speed,
            )

    def _update_termination_flags(self, state):
        waypoint = self.map.get_waypoint(
            self.ego.get_location(),
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            self.offroad_ticks += 1
        else:
            self.offroad_ticks = 0
        self.offroad = (
            self.offroad_ticks > self.config["environment"]["offroad_grace_ticks"]
        )
        if (
            state["speed_kmh"] < self.config["environment"]["stuck_speed_kmh"]
            and self.elapsed_seconds >= 2.0
        ):
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        self.stuck = (
            self.stuck_ticks * self.config["carla"]["fixed_delta_seconds"]
            >= self.config["environment"]["stuck_timeout_seconds"]
        )
        route_complete = (
            self.traveled_distance_m
            >= self.config["environment"]["target_route_distance_m"]
        )
        self.success = route_complete and not self.collision and not self.offroad
        return self.collision or self.offroad or self.stuck or route_complete

    def step(self, action):
        if self.ego is None:
            raise RuntimeError("Call reset() before step().")
        action = int(action)
        self.render_frames.clear()
        pre_distance = self.traveled_distance_m
        state = self._build_state()
        executed_action, rejection_reason = self._apply_tactical_action(action, state)
        terminated = False
        for _ in range(self.config["environment"]["action_repeat_ticks"]):
            state = self._build_state()
            self._low_level_control(state)
            simulation_frame = self.world.tick()
            self._capture_render_frame(simulation_frame)
            self.total_ticks += 1
            self.elapsed_seconds = (
                self.total_ticks * self.config["carla"]["fixed_delta_seconds"]
            )
            state = self._build_state()
            self._update_metrics_tick(state)
            self._update_lane_change_completion()
            terminated = self._update_termination_flags(state)
            if terminated:
                break
        self.decision_steps += 1
        maximum_seconds = (
            self.scenario.get("max_episode_seconds")
            if self.scenario and self.scenario.get("max_episode_seconds") is not None
            else self.config["environment"]["max_episode_seconds"]
        )
        truncated = not terminated and self.elapsed_seconds >= maximum_seconds
        self.timeout = bool(truncated)
        delta_distance = self.traveled_distance_m - pre_distance
        components = calculate_reward_components(
            self.config,
            delta_distance,
            state["speed_kmh"],
            lane_change_rejected=self.last_lane_change_rejected,
            lane_change_accepted=self.last_lane_change_accepted,
            collision=self.collision and terminated,
            offroad=self.offroad and terminated,
            stuck=self.stuck and terminated,
        )
        reward = components["total_reward"]
        self.last_reward_components = components
        for name, value in components.items():
            self.reward_component_totals[name] += float(value)
        self.episode_return += reward
        self.current_state = self._build_state()
        observation = self._normalize_observation(self.current_state)
        info = self._step_info()
        info["executed_action"] = ACTION_NAMES[executed_action]
        info["lane_change_rejection_reason"] = rejection_reason
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_return,
                "l": self.decision_steps,
                "t": self.elapsed_seconds,
            }
            info["episode_metrics"] = self._episode_metrics()
        return observation, reward, terminated, truncated, info

    def _finite_or_radius(self, value):
        if math.isfinite(value):
            return float(value)
        return float(self.config["environment"]["neighbor_radius_m"])

    def _state_summary(self):
        if self.current_state is None:
            return {}
        return {
            "speed_kmh": round(self.current_state["speed_kmh"], 3),
            "target_speed_kmh": round(self.current_state["target_speed_kmh"], 3),
            "current_lane_id": self.current_state["current_lane_id"],
            "target_lane_id": self.current_state["target_lane_id"],
            "lane_change_active": self.current_state["lane_change_active"],
            "lanes": self.current_state["lanes"],
        }

    def _step_info(self):
        return {
            "seed": getattr(self, "seed_value", None),
            "traffic_density": getattr(self, "traffic_density", None),
            "weather": getattr(self, "weather_name", None),
            "requested_npc_count": self.requested_npc_count,
            "actual_npc_count": self.actual_npc_count,
            "reward_components": dict(self.last_reward_components),
            "lane_change_rejected": self.last_lane_change_rejected,
            "lane_change_accepted": self.last_lane_change_accepted,
            "safety_override_active": self.safety_override_active,
            "route_completion": min(
                1.0,
                self.traveled_distance_m
                / self.config["environment"]["target_route_distance_m"],
            ),
            "state": self._state_summary(),
        }

    def _episode_metrics(self):
        mean_speed = float(np.mean(self.speed_samples)) if self.speed_samples else 0.0
        max_speed = float(np.max(self.speed_samples)) if self.speed_samples else 0.0
        return {
            "success": self.success,
            "collision": self.collision,
            "collision_actor_type": self.collision_actor_type,
            "offroad": self.offroad,
            "stuck": self.stuck,
            "timeout": self.timeout,
            "route_completion": min(
                1.0,
                self.traveled_distance_m
                / self.config["environment"]["target_route_distance_m"],
            ),
            "traveled_distance_m": self.traveled_distance_m,
            "elapsed_seconds": self.elapsed_seconds,
            "completion_time_seconds": (
                self.elapsed_seconds if self.success else np.nan
            ),
            "mean_speed_kmh": mean_speed,
            "max_speed_kmh": max_speed,
            "minimum_ttc_seconds": self._finite_or_radius(
                self.minimum_ttc_seconds
            ),
            "minimum_front_gap_m": self._finite_or_radius(
                self.minimum_front_gap_m
            ),
            "accepted_lane_changes": self.lane_changes_accepted,
            "completed_lane_changes": self.lane_changes_completed,
            "unsafe_lane_change_requests": self.unsafe_lane_changes,
            "emergency_following_interventions": self.emergency_interventions,
            "requested_npc_count": self.requested_npc_count,
            "actual_npc_count": self.actual_npc_count,
            "episode_return": self.episode_return,
            "lane_invasion_events": self.lane_invasions,
            "action_counts": dict(self.action_counts),
            "reward_component_totals": dict(self.reward_component_totals),
        }

    def render(self):
        if self.mode != "render":
            return None
        return self.last_rgb_frame

    def drain_render_frames(self):
        frames = list(self.render_frames)
        self.render_frames.clear()
        return frames

    def start_recorder(self, path):
        if self.mode != "render":
            raise RuntimeError("CARLA recording is available only in render mode.")
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.start_recorder(str(path), True)
        self.recorder_path = path
        return path

    def stop_recorder(self):
        if self.recorder_path is not None:
            self.client.stop_recorder()
            self.recorder_path = None

    def _cleanup_actors(self):
        if self.recorder_path is not None:
            try:
                self.stop_recorder()
            except RuntimeError as exc:
                warnings.warn("Could not stop CARLA recorder: %s" % exc, RuntimeWarning)
        for sensor in list(reversed(self.sensors)):
            try:
                if sensor.is_alive:
                    sensor.stop()
            except RuntimeError as exc:
                warnings.warn("Could not stop sensor: %s" % exc, RuntimeWarning)
        for sensor in list(reversed(self.sensors)):
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except RuntimeError as exc:
                warnings.warn("Could not destroy sensor: %s" % exc, RuntimeWarning)
        self.sensors = []
        self.collision_sensor = None
        self.lane_sensor = None
        self.camera = None

        actor_ids = []
        for actor in reversed(self.actors):
            try:
                if actor is not None and actor.is_alive:
                    actor_ids.append(actor.id)
            except RuntimeError as exc:
                warnings.warn("Could not inspect actor during cleanup: %s" % exc)
        if actor_ids and self.client is not None:
            responses = self.client.apply_batch_sync(
                [carla.command.DestroyActor(actor_id) for actor_id in actor_ids],
                False,
            )
            for response in responses:
                if response.error:
                    warnings.warn(
                        "CARLA actor cleanup reported: %s" % response.error,
                        RuntimeWarning,
                    )
        self.actors = []
        self.ego = None

    def close(self):
        if self.closed:
            return
        self.episode_token += 1
        self._cleanup_actors()
        if self.traffic_manager is not None and getattr(self, "tm_sync_enabled", False):
            try:
                original_sync = getattr(self, "original_tm_sync", False)
                self.traffic_manager.set_synchronous_mode(bool(original_sync))
            except Exception as exc:
                warnings.warn(
                    "Could not restore Traffic Manager mode: %s" % exc,
                    RuntimeWarning,
                )
            self.tm_sync_enabled = False
        if (
            self.world is not None
            and self.owns_world_settings
            and self.original_world_settings is not None
        ):
            try:
                self.world.apply_settings(self.original_world_settings)
            except Exception as exc:
                warnings.warn(
                    "Could not restore original CARLA world settings: %s" % exc,
                    RuntimeWarning,
                )
            self.owns_world_settings = False
        self.closed = True
