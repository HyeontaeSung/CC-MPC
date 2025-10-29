from scipy import linalg
from scipy.stats import norm
import cvxpy as cp
import pickle
import time

"added by HYEONTAE"
"""
v8 does original approach with safe region
    - bicycle model with steering as control
    - LTV approximation about nominal path
    - Applies curved road boundaries using segmented polytopes.
"""
# Built-in libraries
import os
import logging
import collections
import weakref
import copy
import numbers
import math

# PyPI libraries
import numpy as np
import scipy.spatial
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cm
import matplotlib.patches as patches
import torch
import docplex.mp
import docplex.mp.model
import docplex.mp.utils

try:
    from utils.trajectory_utils import prediction_output_to_trajectories
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from ..plotting import (
    get_ovehicle_color_set,
    PlotPredictiveControl,
    PlotSimulation,
    PlotPIDController,
)
from ..util import compute_L4_outerapproximation
from ..ovehicle import OVehicle
from ..prediction import generate_vehicle_latents
from ...dynamics.bicycle_v2 import VehicleModel
from ...lowlevel.v4 import VehiclePIDController
from ....generate import AbstractDataCollector
from ....generate import create_semantic_lidar_blueprint
from ....generate.map import MapQuerier
from ....generate.scene import OnlineConfig
from ....generate.scene.v3_2.trajectron_scene import TrajectronPlusPlusSceneBuilder
from ....profiling import profile
from ....exception import InSimulationException

# Local libraries
import carla
import utility as util
import utility.npu
import carlautil
import carlautil.debug

# Local functions
from . import makeconstraint

class MidlevelAgent(AbstractDataCollector):

    Z_SENSOR_REL = 2.5

    def __create_segmentation_lidar_sensor(self):
        return self.__world.spawn_actor(
            create_semantic_lidar_blueprint(self.__world),
            carla.Transform(carla.Location(z=self.Z_SENSOR_REL)),
            attach_to=self.__ego_vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )

    def __make_global_params(self):
        """Get scenario wide parameters used across all loops"""
        params = util.AttrDict()
        # Slack variable for solver
        params.M_big = 10_000
        # Control variable for solver, setting max/min acceleration/speed
        params.max_a = 4.0 # 4 3.5
        params.min_a = -7.0 # -7
        params.max_v = 10 # 10
        # objective : util.AttrDict
        #   Parameters in objective function.
        params.objective = util.AttrDict(
            w_final= 6.0, # 3.0
            w_ref = 3.0,
            w_ch_accel=0.5, # 0.5
            w_ch_turning=2.0, # 2.0
            w_ch_joint=0.1, # 0.1
            w_accel=0.5, # 0.5
            w_turning=1.0, # 1.0
            w_joint=0.2, # 0.2
        )
        # Maximum steering angle
        physics_control = self.__ego_vehicle.get_physics_control()
        wheels = physics_control.wheels
        params.limit_delta = np.deg2rad(wheels[0].max_steer_angle)
        # Max steering
        #   We fix max turning angle to make reasonable planned turns.
        params.max_delta = 0.5 * params.limit_delta
        # longitudinal and lateral dimensions of car are normally 3.70 m, 1.79 m resp.
        bbox = util.AttrDict()
        bbox.lon, bbox.lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
        params.bbox = bbox
        # Number of faces of obstacle sets
        params.L = 4
        # Minimum distance from vehicle to avoid collision.
        #   Assumes that car is a circle.
        # TODO: remove this. Improve bounds instead
        params.diag = np.sqrt(bbox.lon**2 + bbox.lat**2) / 2.0
        return params

    def __setup_rectangular_boundary_conditions(self):
        # __road_seg_starting : np.array
        #   The position and the heading angle of the starting waypoint
        #   of the road of form [x, y, angle] in (meters, meters, radians).
        # __road_segment_enclosure : np.array
        #   Array of shape (4, 2) enclosing the road segment
        (
            self.__road_seg_starting,
            self.__road_seg_enclosure,
            self.__road_seg_params,
        ) = self.__map_reader.road_segment_enclosure_from_actor(self.__ego_vehicle)
        self.__road_seg_starting[1] *= -1  # need to flip about x-axis
        self.__road_seg_starting[2] = util.npu.reflect_radians_about_x_axis(
            self.__road_seg_starting[2]
        )  # need to flip about x-axis
        self.__road_seg_enclosure[:, 1] *= -1  # need to flip about x-axis
        # __goal
        #   Goal destination the vehicle should navigates to.
        self.__goal = util.AttrDict(x=50, y=0, is_relative=True)

    def __setup_curved_road_segmented_boundary_conditions(
        self, turn_choices, max_distance
    ):
        # __turn_choices : list of int
        #   List of choices of turns to make at intersections,
        #   starting with the first intersection to the last.
        self.__turn_choices = turn_choices
        # __max_distance : number
        #   Maximum distance from road
        self.__max_distance = max_distance
        # __road_segs : util.AttrDict
        #   Container of road segment properties.
        self.__road_segs = self.__map_reader.curved_road_segments_enclosure_from_actor(
            self.__ego_vehicle,
            self.__max_distance,
            choices=self.__turn_choices,
            flip_y=True,
        )
        logging.info(
            f"max curvature of planned path is {self.__road_segs.max_k}; "
            f"created {len(self.__road_segs.polytopes)} polytopes covering "
            f"a distance of {np.round(self.__max_distance, 2)} m in total."
        )
        x, y = self.__road_segs.spline(self.__road_segs.distances[-1])
        # __goal
        #   Not used for motion planning when using this BC.
        self.__goal = util.AttrDict(x=x, y=y, is_relative=False)

    def __setup_road_boundary_conditions(self, turn_choices, max_distance):
        """Set up a generic road boundary configuration.
        TODO: extend RoadBoundaryConstraint so it takes over the role of
        __setup_curved_road_segmented_boundary_conditions() and
        __setup_rectangular_boundary_conditions()
        """
        # __turn_choices : list of int
        #   List of choices of turns to make at intersections,
        #   starting with the first intersection to the last.
        self.__turn_choices = turn_choices
        # __max_distance : number
        #   Maximum distance from road
        self.__max_distance = max_distance
        # __road_boundary : RoadBoundaryConstraint
        #   Factory for road boundary constraints.
        self.__road_boundary = self.__map_reader.road_boundary_constraints_from_actor(
            self.__ego_vehicle,
            self.__max_distance,
            choices=self.__turn_choices,
            flip_y=True,
        )
        n_polytopes = len(self.__road_boundary.road_segs.polytopes)
        logging.info(
            f"created {n_polytopes} polytopes covering "
            f"a distance of {np.round(self.__max_distance, 2)} m in total."
        )
        x, y = self.__road_boundary.points[-1]
        # __goal
        #   Not used for motion planning when using this BC.
        self.__goal = util.AttrDict(x=x, y=y, is_relative=False)
        #logging.info(f"__setup_road_boundary_conditions_goal:{self.__goal}")

    def __init__(
        self,
        ego_vehicle,
        map_reader: MapQuerier,
        other_vehicle_ids,
        eval_stg,
        scene_builder_cls=TrajectronPlusPlusSceneBuilder,
        scene_config=OnlineConfig(),
        ##########################
        # Motion Planning settings
        n_burn_interval=4,
        n_predictions=100,
        prediction_horizon=8,
        control_horizon=6,
        step_horizon=1,
        road_boundary_constraints=False,
        angle_boundary_constraints=False,
        #######################
        # Logging and debugging
        log_cplex=True,
        log_agent=False,
        plot_simulation=False,
        plot_boundary=False,
        plot_scenario=False,
        plot_vertices=False,
        plot_overapprox=False,
        get_computeTime=True,
        #######################
        # Planned path settings
        turn_choices=[],
        max_distance=100,
        #######################
        **kwargs,
    ):
        assert control_horizon <= prediction_horizon
        # __ego_vehicle : carla.Vehicle
        #   The vehicle to control in the simulator.
        self.__ego_vehicle = ego_vehicle
        # __map_reader : MapQuerier
        #   To query map data.
        self.__map_reader = map_reader
        # __eval_stg : Trajectron
        #   Prediction Model to generate multi-agent forecasts.
        self.__eval_stg = eval_stg
        # __n_burn_interval : int
        #   Interval in prediction timest to skip prediction and control.
        self.__n_burn_interval = n_burn_interval
        # __n_predictions : int
        #   Number of predictions to generate on each control step.
        self.__n_predictions = n_predictions
        # __prediction_horizon : int
        #   Number of predictions timesteps to predict other vehicles over.
        self.__prediction_horizon = prediction_horizon
        # __control_horizon : int
        #   Number of predictions steps to optimize control over.
        self.__control_horizon = control_horizon
        # __step_horizon : int
        #   Number of predictions steps to execute at each iteration of MPC.
        self.__step_horizon = step_horizon
        self.__scene_builder = None
        self.__scene_builder_cls = scene_builder_cls
        self.__scene_config = scene_config
        # __first_frame : int
        #   First frame in simulation. Used to find current timestep.
        self.__first_frame = None
        self.__world = self.__ego_vehicle.get_world()
        vehicles = self.__world.get_actors(other_vehicle_ids)
        # __other_vehicles : list of carla.Vehicle
        #     List of IDs of vehicles not including __ego_vehicle.
        #     Use this to track other vehicles in the scene at each timestep.
        self.__other_vehicles = dict(zip(other_vehicle_ids, vehicles))
        # __steptime : float
        #   Time in seconds taken to complete one step of MPC.
        self.__steptime = (
            self.__scene_config.record_interval
            * self.__world.get_settings().fixed_delta_seconds
        )
        # __sensor : carla.Sensor
        #     Segmentation sensor. Data points will be used to construct overhead.
        self.__sensor = self.__create_segmentation_lidar_sensor()
        # __lidar_feeds : collections.OrderedDict
        #     Where int key is frame index and value
        #     is a carla.LidarMeasurement or carla.SemanticLidarMeasurement
        self.__lidar_feeds = collections.OrderedDict()
        # __U_warmstarting : ndarray
        #   Controls computed from last MPC step for warmstarting.
        self.__U_warmstarting = None
        self.__U_prev = []
        self.__X_warmstarting = None
        self.__local_planner = VehiclePIDController(self.__ego_vehicle)
        # __params : util.AttrDict
        #   Global parameters for optimization.
        self.__params = self.__make_global_params()
        self.__halfspace = None
        lon = self.__params.bbox.lon
        self.__vehicle_model = VehicleModel(
            self.__control_horizon, self.__steptime, l_r=0.5 * lon, L=lon
        )
        self.__setup_road_boundary_conditions(turn_choices, max_distance)
        self.road_boundary_constraints = road_boundary_constraints
        self.angle_boundary_constraints = angle_boundary_constraints
        self.log_cplex = log_cplex
        self.log_agent = log_agent
        self.plot_simulation = False
        self.plot_boundary = False
        self.plot_scenario = False
        self.plot_scenario_GMM = False
        self.plot_vertices = False
        self.plot_overapprox = False
        self.get_computeTime = get_computeTime
        if self.plot_simulation:
            self.__plot_simulation_data = util.AttrDict(
                actual_trajectory=collections.OrderedDict(),
                planned_trajectories=collections.OrderedDict(),
                planned_controls=collections.OrderedDict(),
                goals=collections.OrderedDict(),
                lowlevel=collections.OrderedDict(),
                halfspace=collections.OrderedDict(),
            )

    def get_vehicle_state_fromCarla(self, flip_x=False, flip_y=False):
        """Get the vehicle state as an ndarray. State consists of
        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z,
        length, width, height, pitch, yaw, roll] where pitch, yaw, roll are in
        radians."""
        return carlautil.actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(
            self.__ego_vehicle, flip_x=flip_x, flip_y=flip_y
        )

    def get_vehicle_state(self, flip_x=False, flip_y=False):
        """Get the vehicle state as an ndarray. State consists of
        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z,
        length, width, height, pitch, yaw, roll] where pitch, yaw, roll are in
        radians."""
        try:
            return self.x_init
        except:
            return carlautil.actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(
                self.__ego_vehicle, flip_x=flip_x, flip_y=flip_y
            )

    
    def get_goal(self):
        return copy.copy(self.__goal)
    
    def get_final_state(self):
        return self.planned_finalstate

    def set_goal(self, x=None, y=None, distance=None, is_relative=True, **kwargs):
        if x is not None and y is not None:
            self.__goal = util.AttrDict(x=x, y=y, is_relative=is_relative)
        elif distance is not None:
            point = self.__road_boundary.get_point_from_start(distance)
            self.__goal = util.AttrDict(x=point[0], y=point[1], is_relative=False)
        else:
            raise NotImplementedError("Unknown method of setting motion planner goal.")

    def start_sensor(self):
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.__sensor.listen(lambda image: type(self).parse_image(weak_self, image))

    def stop_sensor(self):
        """Stop the sensor."""
        self.__sensor.stop()

    @property
    def sensor_is_listening(self):
        return self.__sensor.is_listening

    def __plot_simulation(self):
        if len(self.__plot_simulation_data.planned_trajectories) == 0:
            return
        filename = f"agent{self.__ego_vehicle.id}_oa_simulation"
        bbox = self.__params.bbox
        ps = PlotSimulation(
            self.__prediction_horizon,
            self.__scene_builder.get_scene(),
            self.__map_reader.map_data,
            self.__plot_simulation_data.actual_trajectory,
            self.__plot_simulation_data.planned_trajectories,
            self.__plot_simulation_data.planned_controls,
            self.__plot_simulation_data.goals,
            self.__plot_simulation_data.lowlevel,
            self.__plot_simulation_data.halfspace,
            self.__road_boundary.road_segs,
            np.array([bbox.lon, bbox.lat]),
            self.__step_horizon,
            self.__steptime,
            filename=filename,
            road_boundary_constraints=self.road_boundary_constraints,
        )
        ps.plot_oa()
        ps.plot_overlapped_halfspaces()
        try:
            ps.plot_halfspace_areas()
        except:
            logging.info("area calculation failed")
        PlotPIDController(
            self.__plot_simulation_data.lowlevel,
            self.__world.get_settings().fixed_delta_seconds,
            filename=filename,
        ).plot()

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        self.__sensor.destroy()
        self.__sensor = None
        if self.plot_simulation:
            self.__plot_simulation()

    def do_prediction(self, frame):
        """Get processed scene object from scene builder,
        input the scene to a model to generate the predictions,
        and then return the predictions and the latents variables."""

        """Construct online scene"""
        scene = self.__scene_builder.get_scene()

        """Extract Predictions"""
        frame_id = int(
            (frame - self.__first_frame) / self.__scene_config.record_interval
        )
        timestep = frame_id  # we use this as the timestep
        timesteps = np.array([timestep])
        with torch.no_grad():
            (
                z,
                predictions,
                nodes,
                predictions_dict,
                latent_probs,
            ) = generate_vehicle_latents(
                self.__eval_stg,
                scene,
                timesteps,
                num_samples=self.__n_predictions,
                ph=self.__prediction_horizon,
                z_mode=False,
                gmm_mode=False,
                full_dist=False,
                all_z_sep=False,
            )
        # logging.info(f"shape(z){z.shape}, z[1].mean:{np.mean(z[1])}")
        # logging.info(f"z[1]:{z[1]}")
        # logging.info(f"prediction.shape:{predictions.shape}")
        # logging.info(f"latent_probs:{latent_probs[1]}")
        # logging.info(f"nodes:{nodes}")
        _, past_dict, ground_truth_dict = prediction_output_to_trajectories(
            predictions_dict,
            dt=scene.dt,
            max_h=10,
            ph=self.__prediction_horizon,
            map=None,
        )
        return util.AttrDict(
            scene=scene,
            timestep=timestep,
            nodes=nodes,
            predictions=predictions,
            z=z,
            latent_probs=latent_probs,
            past_dict=past_dict,
            ground_truth_dict=ground_truth_dict,
        )

    def make_ovehicles(self, result):
        scene, timestep, nodes = result.scene, result.timestep, result.nodes
        predictions, latent_probs, z = result.predictions, result.latent_probs, result.z
        past_dict, ground_truth_dict = result.past_dict, result.ground_truth_dict

        """Preprocess predictions"""
        minpos = np.array([scene.x_min, scene.y_min])
        ovehicles = []
        for idx, node in enumerate(nodes):
            if node.id == "ego":
                continue
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(
                self.__other_vehicles[int(node.id)]
            )
            veh_bbox = np.array([lon, lat])
            veh_gt = ground_truth_dict[timestep][node] + minpos
            veh_past = past_dict[timestep][node] + minpos
            veh_predict = predictions[idx] + minpos
            veh_latent_pmf = latent_probs[idx]
            n_states = veh_latent_pmf.size
            zn = z[idx]
            veh_latent_predictions = [[] for x in range(n_states)]
            for jdx, p in enumerate(veh_predict):
                veh_latent_predictions[zn[jdx]].append(p)
            for jdx in range(n_states):
                veh_latent_predictions[jdx] = np.array(veh_latent_predictions[jdx])
            ovehicle = OVehicle.from_trajectron(
                node,
                self.__prediction_horizon,
                veh_gt,
                veh_past,
                veh_latent_pmf,
                veh_latent_predictions,
                bbox=veh_bbox,
            )
            ovehicles.append(ovehicle)
        return ovehicles

    def get_current_velocity(self):
        """Get current velocity of vehicle in m/s."""
        v_0_x, v_0_y, _ = carlautil.actor_to_velocity_ndarray(
            self.__ego_vehicle, flip_y=True
        )
        return np.sqrt(v_0_x**2 + v_0_y**2)

    def make_local_params(self, frame, ovehicles,Tsh):
        """Get the local optimization parameters used for current MPC step."""

        """Get parameters to construct control and state variables."""
        params = util.AttrDict()
        params.frame = frame
        p_0_x, p_0_y, _ = carlautil.to_location_ndarray(self.__ego_vehicle, flip_y=True)
        _, psi_0, _ = carlautil.actor_to_rotation_ndarray(
            self.__ego_vehicle, flip_y=True
        )
        v_0_mag = self.get_current_velocity()
        # Using the predicted x state
        try:
            X_warmstarting = self.__X_warmstarting[0] # the second plannd state
            x_init = self.__X_warmstarting[0]
            # logging.info(f"using x warmstart: {x_init}")
        except TypeError:
            x_init = np.array([p_0_x, p_0_y, psi_0, v_0_mag])
            X_warmstarting = x_init
        logging.info
        # x_init = np.array([p_0_x, p_0_y, psi_0, v_0_mag])
        # X_warmstarting = x_init
        params.x_init = x_init # hyeontae
        self.x_init = x_init
        # self.save_x_init(x_init, frame, self.__ego_vehicle.id)
        initial_state = util.AttrDict(world=x_init, local=np.array([0, 0, 0, v_0_mag]))
        params.initial_state = initial_state
        # try:
        #     u_init = self.__U_warmstarting[0]
        #     logging.info(f"u_init:{u_init}")
        # except TypeError:
        #     logging.info("not using U_warnstarting")
        #     u_init = np.array([0., 0.])
        # Using previous control doesn't work
        u_init = np.array([0.0, 0.0])

        lon = self.__params.bbox.lon

        self.__vehicle_model = VehicleModel(
            Tsh, self.__steptime, l_r=0.5 * lon, L=lon
        )
        x_bar, u_bar, Gamma, nx, nu = self.__vehicle_model.get_optimization_ltv(
            X_warmstarting, u_init
        )
        params.x_bar, params.u_bar, params.Gamma = x_bar, u_bar, Gamma
        params.nx, params.nu = nx, nu

        """Get controls for other vehicles."""
        # O - number of obstacles
        params.O = len(ovehicles)
        # K - for each o=1,...,O K[o] is the number of outer approximations for vehicle o
        params.K = np.zeros(params.O, dtype=int)
        for idx, vehicle in enumerate(ovehicles):
            params.K[idx] = vehicle.n_states
        return params

    def __plot_segs_polytopes(self, params, segments, goal):
        fig, ax = plt.subplots(figsize=(7, 7))
        x_min, y_min = np.min(self.__road_boundary.points, axis=0)
        x_max, y_max = np.max(self.__road_boundary.points, axis=0)
        self.__map_reader.render_map(
            ax, extent=(x_min - 20, x_max + 20, y_min - 20, y_max + 20)
        )
        x, y, _ = carlautil.to_location_ndarray(self.__ego_vehicle, flip_y=True)
        ax.scatter(x, y, c="r", zorder=10)
        x, y = goal
        ax.scatter(x, y, c="g", marker="*", zorder=10)
        for (A, b), in_junction in zip(segments.polytopes, segments.mask):
            if in_junction:
                util.npu.plot_h_polyhedron(ax, A, b, fc="r", ec="r", alpha=0.3)
            else:
                util.npu.plot_h_polyhedron(ax, A, b, fc="b", ec="b", alpha=0.3)
        filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_boundary"
        fig.savefig(os.path.join("out", f"{filename}.png"))
        fig.clf()

    def compute_segs_polytopes_and_goal(self, params, Tsh):
        """Compute the road boundary constraints and goal.
        Returns
        =======
        util.AttrDict
            Payload of segment polytopes for optimization.
        ndarray
            Global (x, y) coordinates for car's destination at for the MPC step.
        """
        position = params.initial_state.world[:2]
        v_lim = min(self.__ego_vehicle.get_speed_limit() * 0.28, self.__params.max_v)
        distance = v_lim * self.__steptime * Tsh + 1
        segments = self.__road_boundary.collect_segs_polytopes_and_goal(
            position, distance
        )
        goal = segments.goal
        if self.plot_boundary:
            self.__plot_segs_polytopes(params, segments, goal)
        return segments, goal

    def compute_state_constraints(self, params, X):
        """Set velocity magnitude constraints.
        Usually street speed limits are 30 km/h == 8.33.. m/s.
        Speed limits can be 30, 40, 60, 90 km/h
        Parameters
        ==========
        params : util.AttrDict
        v : np.array of docplex.mp.vartype.VarType
        """
        # max_v = self.__ego_vehicle.get_speed_limit() # is m/s
        v = X[:, 3]
        max_v = self.__params.max_v
        constraints = []
        constraints.extend([z <= max_v for z in v])
        constraints.extend([z >= 0 for z in v])
        return constraints

    def __compute_vertices(self, params, ovehicles):
        """Compute verticles from predictions."""
        K, n_ov = params.K, params.O
        T = self.__prediction_horizon
        vertices = np.empty((T, np.max(K), n_ov), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                for t in range(T):
                    ps = ovehicle.pred_positions[latent_idx][:, t]
                    yaws = ovehicle.pred_yaws[latent_idx][:, t]
                    vertices[t][latent_idx][ov_idx] = util.npu.vertices_of_bboxes(
                            ps, yaws, ovehicle.bbox)

        return vertices

    def __plot_overapproximations(self, params, ovehicles, vertices, A_union, b_union):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        ax = axes[1]
        X = vertices[-1][0][0][:, 0:2].T
        ax.scatter(X[0], X[1], color="r", s=2)
        X = vertices[-1][0][0][:, 2:4].T
        ax.scatter(X[0], X[1], color="b", s=2)
        X = vertices[-1][0][0][:, 4:6].T
        ax.scatter(X[0], X[1], color="g", s=2)
        X = vertices[-1][0][0][:, 6:8].T
        ax.scatter(X[0], X[1], color="m", s=2)
        A = A_union[-1][0][0]
        b = b_union[-1][0][0]
        try:
            util.npu.plot_h_polyhedron(ax, A, b, fc="none", ec="k")
        except scipy.spatial.qhull.QhullError as e:
            print(f"Failed to plot polyhedron of OV")

        x, y, _ = carlautil.actor_to_location_ndarray(self.__ego_vehicle, flip_y=True)
        ax = axes[0]
        ax.scatter(x, y, marker="*", c="k", s=100)
        ovehicle_colors = get_ovehicle_color_set([ov.n_states for ov in ovehicles])
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                logging.info(f"Plotting OV {ov_idx} latent value {latent_idx}.")
                color = ovehicle_colors[ov_idx][latent_idx]
                for t in range(self.__prediction_horizon):
                    X = vertices[t][latent_idx][ov_idx][:, 0:2].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:, 2:4].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:, 4:6].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:, 6:8].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    A = A_union[t][latent_idx][ov_idx]
                    b = b_union[t][latent_idx][ov_idx]
                    try:
                        util.npu.plot_h_polyhedron(
                            ax, A, b, fc="none", ec=color
                        )  # , alpha=0.3)
                    except scipy.spatial.qhull.QhullError as e:
                        print(
                            f"Failed to plot polyhedron of OV {ov_idx} latent value {latent_idx} timestep t={t}"
                        )

        for ax in axes:
            ax.set_aspect("equal")
        filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_overapprox"
        fig.savefig(os.path.join("out", f"{filename}.png"))
        fig.clf()

    def __compute_overapproximations(self, params, ovehicles, vertices):
        """Compute the approximation of the union of obstacle sets.
        Parameters
        ==========
        vertices : ndarray
            Verticles to for overapproximations.
        params : util.AttrDict
            Parameters of motion planning problem.
        ovehicles : list of OVehicle
            Vehicles to compute overapproximations from.
        Returns
        =======
        ndarray
            Collection of A matrices of shape (T, max(K), O, L, 2).
            Where max(K) largest set of cluster.
        ndarray
            Collection of b vectors of shape (T, max(K), O, L).
            Where max(K) largest set of cluster.
        """
        K, n_ov = params.K, params.O
        T = self.__prediction_horizon
        shape = (T, np.max(K), n_ov,)
        A_union = np.empty(shape, dtype=object).tolist()
        b_union = np.empty(shape, dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                for t in range(T):
                    yaws = ovehicle.pred_yaws[latent_idx][:, t]
                    vertices_k = vertices[t][latent_idx][ov_idx]
                    mean_theta_k = np.mean(yaws)
                    A_union_k, b_union_k = compute_L4_outerapproximation(
                        mean_theta_k, vertices_k
                    )
                    A_union[t][latent_idx][ov_idx] = A_union_k
                    b_union[t][latent_idx][ov_idx] = b_union_k

        """Plot the overapproximation"""
        if self.plot_overapprox:
            self.__plot_overapproximations(
                params, ovehicles, vertices, A_union, b_union
            )

        return A_union, b_union

    def compute_road_boundary_constraints(self, params, X, Omicron, segments, Tsh):
        constraints = []
        T = Tsh
        M_big = self.__params.M_big
        print("allright?")
        # diag = self.__params.diag
        for t in range(T):
            for seg_idx, (A, b) in enumerate(segments.polytopes):
                # lhs = util.obj_matmul(A, X[t][:2]) - np.array(
                #     M_big * (1 - Omicron[seg_idx, t])
                # )
                lhs = util.obj_matmul(A, X[t,:2]) - np.array(
                    M_big * (1 - Omicron[seg_idx, t])
                )
                rhs = b  # - diag
                """Constraints on road boundaries"""
                
                constraints.extend([l <= r for (l, r) in zip(lhs, rhs)])
            constraints.extend([cp.sum(Omicron[:, t]) >= 1])
            # constraints.extend([cp.sum(Omicron[:, t]) >= 0])
        return constraints


    
    def get_OV_direction(
            self, posemean_x, posemean_y
    ):
        
        if posemean_x <= -98.122 and posemean_y <= -5:# or posemean_y >= -21.178: # in Town03_scene3
            direction = "RightDown"
        elif posemean_x <= -98.122 and posemean_y >= -5:# or posemean_y >= -21.178: # in Town03_scene3
            direction = "LeftDown"

        elif posemean_x <= -83.122 and posemean_y <= -21.178:
            direction = "rightParallel"
        elif (posemean_x >= -80.122 and posemean_y >= 10.178): 
            direction = "LeftParallel"
        else:
            direction = "rightParallel"

        return direction
    

    def compute_obstacle_constraints_GMM_Minkowski_idealprediction(
        self, params, ovehicles, Delta2, Omicron, temp_x, eps_ura, segments, Tsh, ref_traj     
    ):
        constraints = []
        M_big = self.__params.M_big
        T = Tsh
        L, K = self.__params.L, params.K
        truck = {'d': np.array([3.7, 1.79])}
        truck_d = truck['d']
        CAR_R = 4.47213 # this is actually a diameter
        R = 3.4 # EV radius + OV radius
        
        if self.road_boundary_constraints:
            # Auxiliary variable for masking and summing
            Z = cp.Variable(Omicron.shape)

            # Constraints for masking
            mask_constraints = [
                Z[i, j] == (Omicron[i, j] if not segments.mask[i] else 0)
                for i in range(Omicron.shape[0])
                for j in range(Omicron.shape[1])
            ]

            # Summing along axis=0
            Z_sum = M_big * cp.sum(Z, axis=0)

            # Variable and constraints for repeating
            S_big_repeated = cp.Variable((L, T))
            repeat_constraints = [S_big_repeated[i, :] == Z_sum for i in range(L)]

            constraints += mask_constraints
            constraints += repeat_constraints
        else:
            S_big_repeated = np.zeros((L, T), dtype=float)

        Tpred = self.__prediction_horizon

        logging.info(f"K:{params.K}")
        # ideal prediction
        if T < Tpred:
            ideal_trajs = self.predict_ideal(ovehicles, T, self.__ego_vehicle.id, params)

        # Let's check if the first prediction passes the intersection or not.
        O = params.O
        OV_injuction = np.empty((O,), dtype=object).tolist()
        direct = np.empty((O,), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):   
            for latent_idx in range(ovehicle.n_states):
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                posemean_x = np.mean(poseData[0::Tpred,0])
                posemean_y = np.mean(poseData[0::Tpred,1])
                if posemean_x >= 190 or posemean_y <=-80: # in Town03_scene4 T inersection.
                #if posemean_x <= -98.122 or (posemean_x <= -83.122 and posemean_y <= -21.178):# or posemean_y >= -21.178: # in Town03_scene3 Star intersection
                #    OV_injuction[ov_idx] = False
                #elif (posemean_x >= -80.122 and posemean_y >= 10.178): Star intersection
                    OV_injuction[ov_idx] = False 
                else:
                    OV_injuction[ov_idx] = True
                #    direct[ov_idx] = "InJunction"
        

        #logging.info(f"OV{OV_injuction}")

        #if OV_injuction:
        OVconstraint = False
        for ov_idx, ovehicle in enumerate(ovehicles):
            OVconstraint = OVconstraint or OV_injuction[ov_idx]

        # Save pose, angle [tau+1|tau] of OV
        posemean_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posemean_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawmean_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawcov_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()

        halfspace = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
        prob_lower_save = np.empty((T), dtype=object).tolist()

        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                # load data
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                yawData = np.vstack(ovehicle.pred_yaws[latent_idx])

                posemean_x_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,0])
                posemean_y_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,1])
                yawmean_save[ov_idx][latent_idx] = np.mean(yawData[:,0])
                posecov_x_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,0])
                posecov_y_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,1])
                yawcov_save[ov_idx][latent_idx] = np.cov(yawData[:,0])
        
        ovehicles_toSave_moments = []
        for ov_idx, ovehicle in enumerate(ovehicles):
            ovehicles_toSave_moments.append(copy.deepcopy(ovehicle))
                
        for ov_idx, ovehicle in enumerate(ovehicles):
            if True: #OV_injuction[ov_idx]:
                for latent_idx in range(ovehicle.n_states):
                    
                    if T < self.__prediction_horizon:
                        poseData = np.vstack(ideal_trajs[ov_idx][latent_idx])
                        ovehicles_toSave_moments[ov_idx].pred_positions[latent_idx] = ideal_trajs[ov_idx][latent_idx]
                        Tpred = T
                    else:
                        poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                    # logging.info(f"Tpred:{Tpred}")
                    # logging.info(f"p1_t0:{np.shape(poseData[0::Tpred,1])}")
                    for t in range(T):
                        p0_t = poseData[t::Tpred,0]
                        p1_t = poseData[t::Tpred,1]
                        mean = np.mean([p0_t, p1_t], axis = 1)
                        # logging.info(f"t:{t}, mean:{mean}")
                        prob_lower = 1.0
                        for tau_future in range(t):
                            # tau_future = 0
                            p0_tau = poseData[tau_future::Tpred,0]
                            p1_tau = poseData[tau_future::Tpred,1]
                                                
                            p_t_tau = [p0_t, p1_t, p0_tau, p1_tau]

                            # predict moments
                            cov_infer, cov_mu, cov_t = makeconstraint.predict_moments(p_t_tau)

                            # Minkowski sum
                            eps_ijt = eps_ura[ov_idx, latent_idx] / self.__prediction_horizon             
                            chi_risk_tol = scipy.stats.chi2.ppf(1 - eps_ijt, df = 2)
                            target_p = 0.9999
                            chi_target_p = scipy.stats.chi2.ppf(target_p, df = 2)

                            _, cov_mvoe = makeconstraint.compute_mvoe(cov_infer*chi_risk_tol , cov_mu*chi_target_p )

                            Sigma = np.identity(n=2)
                            _, cov_mvoe_R = makeconstraint.compute_mvoe(cov_mvoe , Sigma*R**2)

                            ref_x = [ref_traj[t][0], ref_traj[t][1]]
                            # n_star, d_star = makeconstraint.closest_tangent_line(mean, cov_mvoe, 1, ref_x)
                            
                            m = - (ref_traj[t][0] - mean[0]) / (ref_traj[t][1] - mean[1]) # fix tangent
                            n_star, d_star, _ = makeconstraint.choose_closest_tangent(mean , cov_mvoe_R, 1, m, ref_x)
                            
                            if n_star @ mean <= d_star:
                                constraints += [
                                    n_star.T @ cp.vstack([temp_x[t][0], temp_x[t][1]]) >= d_star \
                                        # + R#   + R*np.sqrt(m*m+1) 
                                ]
                                A =  - n_star
                                b =  - d_star
                            else:
                                constraints += [
                                    n_star.T @ cp.vstack([temp_x[t][0], temp_x[t][1]]) <= d_star \
                                        # - R#   - R*np.sqrt(m*m+1) 
                                ]
                                A =  n_star
                                b =  d_star

                            """calculate the lower bound for recursively feasible condition"""
                            prob_lower_temp = makeconstraint.compute_lower_bound(cov_infer, cov_mu, cov_t, eps_ijt)
                            prob_lower = min(prob_lower,prob_lower_temp)
                            # save halfspace
                            # halfspace[ov_idx][latent_idx][t] = [A[0], A[1], b]
                            # note: maybe we can optimize the buffer(0.5 * CAR_R ) 
                        prob_lower_save[t] = prob_lower
            else:
                constraints += []
        # constraints = []
        vertices = self.__compute_vertices(params, ovehicles)
        A_union, b_union = self.__compute_overapproximations(
            params, ovehicles, vertices
        )
        # self.__halfspace = halfspace
        ovStateMean_tau_1 = (posemean_x_save, posemean_y_save, yawmean_save)
        ovStateCov_tau_1 = (posecov_x_save, posecov_y_save, yawcov_save)

        # save moments
        self.save_moments(ovehicles_toSave_moments, params.O, K, T, self.__prediction_horizon, self.__ego_vehicle.id, params)
        if T == self.__prediction_horizon:
            self.__prob_lower_save = prob_lower_save
        logging.info(f"prob_lower_save:{prob_lower_save}")
        return constraints, vertices, A_union, b_union, OVconstraint, direct, ovStateMean_tau_1, ovStateCov_tau_1, 0

    def compute_obstacle_constraints_GMM(
        self, params, ovehicles, Delta2, Omicron, temp_x, eps_ura, segments, Tsh     
    ):
        constraints = []
        M_big = self.__params.M_big
        T = Tsh
        L, K = self.__params.L, params.K
        truck = {'d': np.array([3.7, 1.79])}
        truck_d = truck['d']
        CAR_R = 4.47213 # this is actually a diameter

        if self.road_boundary_constraints:
            # Auxiliary variable for masking and summing
            Z = cp.Variable(Omicron.shape)

            # Constraints for masking
            mask_constraints = [
                Z[i, j] == (Omicron[i, j] if not segments.mask[i] else 0)
                for i in range(Omicron.shape[0])
                for j in range(Omicron.shape[1])
            ]

            # Summing along axis=0
            Z_sum = M_big * cp.sum(Z, axis=0)

            # Variable and constraints for repeating
            S_big_repeated = cp.Variable((L, T))
            repeat_constraints = [S_big_repeated[i, :] == Z_sum for i in range(L)]

            constraints += mask_constraints
            constraints += repeat_constraints
        else:
            S_big_repeated = np.zeros((L, T), dtype=float)

        Tpred = self.__prediction_horizon
        # Let's check if the first prediction passes the intersection or not.
        O = params.O
        OV_injuction = np.empty((O,), dtype=object).tolist()
        direct = np.empty((O,), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):   
            for latent_idx in range(ovehicle.n_states):
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                posemean_x = np.mean(poseData[0::Tpred,0])
                posemean_y = np.mean(poseData[0::Tpred,1])
                if posemean_x >= 190 or posemean_y <=-80: # in Town03_scene4 T inersection.
                #if posemean_x <= -98.122 or (posemean_x <= -83.122 and posemean_y <= -21.178):# or posemean_y >= -21.178: # in Town03_scene3 Star intersection
                #    OV_injuction[ov_idx] = False
                #elif (posemean_x >= -80.122 and posemean_y >= 10.178): Star intersection
                    OV_injuction[ov_idx] = False 
                else:
                    OV_injuction[ov_idx] = True
                #    direct[ov_idx] = "InJunction"
        

        #logging.info(f"OV{OV_injuction}")
        logging.info(f"K:{params.K}")
        #if OV_injuction:
        OVconstraint = False
        for ov_idx, ovehicle in enumerate(ovehicles):
            OVconstraint = OVconstraint or OV_injuction[ov_idx]

        # Save pose, angle [tau+1|tau] of OV
        posemean_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posemean_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawmean_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawcov_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()

        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                # load data
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                yawData = np.vstack(ovehicle.pred_yaws[latent_idx])

                posemean_x_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,0])
                posemean_y_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,1])
                yawmean_save[ov_idx][latent_idx] = np.mean(yawData[:,0])
                posecov_x_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,0])
                posecov_y_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,1])
                yawcov_save[ov_idx][latent_idx] = np.cov(yawData[:,0])
        
        for ov_idx, ovehicle in enumerate(ovehicles):
            if OV_injuction[ov_idx]:
                for latent_idx in range(ovehicle.n_states):
                    for t in range(T):                      
                        yawData = np.vstack(ovehicle.pred_yaws[latent_idx])
                        poseData = np.vstack(ovehicle.pred_positions[latent_idx])

                        coeff1 = [-np.cos(yawData[:,t]), np.sin(yawData[:,t]), np.cos(yawData[:,t]) * poseData[t::Tpred,0] - np.sin(yawData[:,t]) * poseData[t::Tpred,1] + truck_d[1]/2]
                        coeff2 = [-np.sin(yawData[:,t]), -np.cos(yawData[:,t]), np.sin(yawData[:,t]) * poseData[t::Tpred,0] + np.cos(yawData[:,t]) * poseData[t::Tpred,1] + truck_d[0]/2]
                        coeff3 = [np.cos(yawData[:,t]), -np.sin(yawData[:,t]), -np.cos(yawData[:,t]) * poseData[t::Tpred,0] + np.sin(yawData[:,t]) * poseData[t::Tpred,1] + truck_d[1]/2]
                        coeff4 = [np.sin(yawData[:,t]), np.cos(yawData[:,t]), -np.sin(yawData[:,t]) * poseData[t::Tpred,0] - np.cos(yawData[:,t]) * poseData[t::Tpred,1] + truck_d[0]/2]

                        delta_ijt = Delta2[(ov_idx, latent_idx, t)]          
                        eps_ijt = eps_ura[ov_idx, latent_idx] / Tpred
                        Gamma_ijt = norm.ppf(1-eps_ijt) #np.sqrt((1-eps_ijt)/eps_ijt) #norm.pdf(norm.ppf(1-eps_ijt))/eps_ijt #norm.ppf(1-eps_ijt)
                    
                        mean1 = np.mean(coeff1, axis=1)
                        mean2 = np.mean(coeff2, axis=1)
                        mean3 = np.mean(coeff3, axis=1)
                        mean4 = np.mean(coeff4, axis=1)
                        
                        cov1 = linalg.sqrtm(np.cov(coeff1))
                        cov2 = linalg.sqrtm(np.cov(coeff2))
                        cov3 = linalg.sqrtm(np.cov(coeff3))
                        cov4 = linalg.sqrtm(np.cov(coeff4))

                        constraints += [
                            mean1.T @ temp_x[t] + Gamma_ijt * cp.norm(cov1 @ temp_x[t], p=2) + 0.5 * CAR_R <= M_big * (1-delta_ijt[0]) + S_big_repeated[0, t],
                            mean2.T @ temp_x[t] + Gamma_ijt * cp.norm(cov2 @ temp_x[t], p=2) + 0.5 * CAR_R <= M_big * (1-delta_ijt[1]) + S_big_repeated[1, t],
                            mean3.T @ temp_x[t] + Gamma_ijt * cp.norm(cov3 @ temp_x[t], p=2) + 0.5 * CAR_R <= M_big * (1-delta_ijt[2]) + S_big_repeated[2, t],
                            mean4.T @ temp_x[t] + Gamma_ijt * cp.norm(cov4 @ temp_x[t], p=2) + 0.5 * CAR_R <= M_big * (1-delta_ijt[3]) + S_big_repeated[3, t],
                        ]
                        constraints += [sum(delta_ijt) == 1]
            else:
                constraints += []
        # constraints = []
        vertices = self.__compute_vertices(params, ovehicles)
        A_union, b_union = self.__compute_overapproximations(
            params, ovehicles, vertices
        )

        ovStateMean_tau_1 = (posemean_x_save, posemean_y_save, yawmean_save)
        ovStateCov_tau_1 = (posecov_x_save, posecov_y_save, yawcov_save)
        return constraints, vertices, A_union, b_union, OVconstraint, direct, ovStateMean_tau_1, ovStateCov_tau_1


    def compute_obstacle_constraints_GMM_affine_ideal(
        self, params, ovehicles, Delta2, Omicron, temp_x, eps_ura, segments, Tsh, ref_traj     
    ):
        """"
            Choose only one constraint that is the most feasible using a reference trajectory
            ignore the heading uncertainty
        """
        constraints = []
        M_big = self.__params.M_big
        T = Tsh
        L, K = self.__params.L, params.K
        truck = {'d': np.array([3.7, 1.79])}
        truck_d = truck['d']
        CAR_R = 4.47213 # this is actually a diameter

        R = 3.4 # EV radius + OV radius

        if self.road_boundary_constraints:
            # Auxiliary variable for masking and summing
            Z = cp.Variable(Omicron.shape)

            # Constraints for masking
            mask_constraints = [
                Z[i, j] == (Omicron[i, j] if not segments.mask[i] else 0)
                for i in range(Omicron.shape[0])
                for j in range(Omicron.shape[1])
            ]

            # Summing along axis=0
            Z_sum = M_big * cp.sum(Z, axis=0)

            # Variable and constraints for repeating
            S_big_repeated = cp.Variable((L, T))
            repeat_constraints = [S_big_repeated[i, :] == Z_sum for i in range(L)]

            constraints += mask_constraints
            constraints += repeat_constraints
        else:
            S_big_repeated = np.zeros((L, T), dtype=float)

        Tpred = self.__prediction_horizon
        # ideal prediction
        if T < Tpred:
            ideal_trajs = self.predict_ideal(ovehicles, T, self.__ego_vehicle.id, params)

        # Let's check if the first prediction passes the intersection or not.
        O = params.O
        OV_injuction = np.empty((O,), dtype=object).tolist()
        direct = np.empty((O,), dtype=object).tolist()
        out_junction = False # if you want to decide if ovs are in juction
        if out_junction:
            for ov_idx, ovehicle in enumerate(ovehicles):   
                for latent_idx in range(ovehicle.n_states):
                    poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                    posemean_x = np.mean(poseData[0::Tpred,0])
                    posemean_y = np.mean(poseData[0::Tpred,1])
                    if posemean_x >= 190 or posemean_y <= -80: # in Town03_scene4 T inersection.
                    # if posemean_x <= -98.122 or (posemean_x <= -83.122 and posemean_y <= -21.178):# or posemean_y >= -21.178: # in Town03_scene3 Star intersection
                    #    OV_injuction[ov_idx] = False
                    # elif (posemean_x >= -80.122 and posemean_y >= 10.178): # Star intersection
                        OV_injuction[ov_idx] = False 
                    else:
                        OV_injuction[ov_idx] = True
                    #    direct[ov_idx] = "InJunction"
        

        #logging.info(f"OV{OV_injuction}")
        logging.info(f"K:{params.K}")
        #if OV_injuction:
        OVconstraint = False
        if out_junction:
            for ov_idx, ovehicle in enumerate(ovehicles):
                OVconstraint = OVconstraint or OV_injuction[ov_idx]

        # Save pose, angle [tau+1|tau] of OV
        posemean_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posemean_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawmean_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawcov_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()

        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                # load data
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                yawData = np.vstack(ovehicle.pred_yaws[latent_idx])

                posemean_x_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,0])
                posemean_y_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,1])
                yawmean_save[ov_idx][latent_idx] = np.mean(yawData[:,0])
                posecov_x_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,0])
                posecov_y_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,1])
                yawcov_save[ov_idx][latent_idx] = np.cov(yawData[:,0])

        halfspace = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()

        ovehicles_toSave_moments = []          
        for ov_idx, ovehicle in enumerate(ovehicles):
            ovehicles_toSave_moments.append(copy.deepcopy(ovehicle))
        
        for ov_idx, ovehicle in enumerate(ovehicles):
            if True: # OV_injuction[ov_idx]:
                for latent_idx in range(ovehicle.n_states):
                    if T < self.__prediction_horizon:
                        poseData = np.vstack(ideal_trajs[ov_idx][latent_idx])
                        ovehicles_toSave_moments[ov_idx].pred_positions[latent_idx] = ideal_trajs[ov_idx][latent_idx]
                        Tpred = T
                        # ovehicle.pred_positions[latent_idx] = ovehicles_toSave_moments[ov_idx].pred_positions[latent_idx]
                    else:
                        poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                    # logging.info(f"Tpred:{Tpred}")
                    # logging.info(f"shape:{np.shape(poseData[0::Tpred,0])}")
                    # logging.info(f"poseData.shape:{poseData[0::Tpred].shape}")
                    for t in range(T):                      
                        # yawData = np.vstack(ovehicle.pred_yaws[latent_idx])
                        # poseData = np.vstack(ovehicle.pred_positions[latent_idx])

                        # delta_ijt = Delta2[(ov_idx, latent_idx, t)]          
                        eps_ijt = eps_ura[ov_idx, latent_idx] / Tpred
                        Gamma_ijt = norm.ppf(1-eps_ijt) #np.sqrt((1-eps_ijt)/eps_ijt) #norm.pdf(norm.ppf(1-eps_ijt))/eps_ijt #norm.ppf(1-eps_ijt)

                        
                        p0 = poseData[t::Tpred,0]
                        p1 = poseData[t::Tpred,1]

                        mean_p0 = np.mean(p0)
                        mean_p1 = np.mean(p1)
                        
                        mean = np.array([mean_p0, mean_p1])
                        # logging.info(f"t:{t}, mean:{mean}")
                        cov = np.cov([p0, p1])
                        cov_sqrt = linalg.sqrtm(cov)
                        # logging.info(f"t:{t}, mean: {mean}, cov: {cov}")
                        m = - (ref_traj[t][0] - mean_p0) / (ref_traj[t][1] - mean_p1)
                        M = np.array([m, -1])

                        Sigma = np.identity(n=2)
                        ref_pose = np.array((ref_traj[t][0], ref_traj[t][1])) 

                        n_star, d_star, which_idx = makeconstraint.choose_closest_tangent(mean, Sigma, R, m, ref_pose)
                        
                        if n_star @ mean <= d_star:
                            constraints += [
                                n_star.T @ cp.vstack([temp_x[t][0], temp_x[t][1]]) >= d_star + Gamma_ijt * cp.norm(cov_sqrt @ M.T, p=2) \
                                # + R*np.sqrt(m*m+1)
                            ]
                            value = n_star[0] * ref_pose[0] + n_star[1] * ref_pose[1] - d_star - Gamma_ijt * np.linalg.norm(cov_sqrt @ M.T, 2)
                            # logging.info(f"{value}>=0")
                            A = - n_star
                            b =  - (d_star + Gamma_ijt * np.linalg.norm(cov_sqrt @ M.T, 2))
                        else:
                            constraints += [
                                n_star.T @ cp.vstack([temp_x[t][0], temp_x[t][1]]) <= d_star - Gamma_ijt * cp.norm(cov_sqrt @ M.T, p=2) \
                                # - R*np.sqrt(m*m+1)
                            ]
                            value = n_star[0] * ref_pose[0] + n_star[1] * ref_pose[1] - d_star + Gamma_ijt * np.linalg.norm(cov_sqrt @ M.T, 2)
                            # logging.info(f"{value}<=0")
                            A =  n_star
                            b =   (d_star - Gamma_ijt * np.linalg.norm(cov_sqrt @ M.T, 2))

                        # save halfspace
                        halfspace[ov_idx][latent_idx][t] = [A[0], A[1], b]

                        # val1 = m*(ref_traj[t][0] - mean_p0) - (ref_traj[t][1] - mean_p1)
                        # val2 = - m*(ref_traj[t][0] - mean_p0) + (ref_traj[t][1] - mean_p1)

                        # val = [val1, val2]
                        # const_idx = val.index(min(val))
                        # if 0==const_idx:
                        #     constraints += [m*(temp_x[t][0] - mean_p0) - (temp_x[t][1] - mean_p1) + R*np.sqrt(m*m+1)+ Gamma_ijt * cp.norm(cov_sqrt @ M.T, p=2) <= S_big_repeated[0, t]]
                        # elif 1==const_idx:
                        #     constraints += [-m*(temp_x[t][0] - mean_p0) + (temp_x[t][1] - mean_p1) + R*np.sqrt(m*m+1)+ Gamma_ijt * cp.norm(cov_sqrt @ M.T, p=2) <= S_big_repeated[0, t]]


            else:
                constraints += []
        # constraints = []
        vertices = self.__compute_vertices(params, ovehicles)
        A_union, b_union = self.__compute_overapproximations(
            params, ovehicles, vertices
        )
        self.__halfspace = halfspace
        # save moments
        self.save_moments(ovehicles_toSave_moments, params.O, K, T, self.__prediction_horizon, self.__ego_vehicle.id, params)

        ovStateMean_tau_1 = (posemean_x_save, posemean_y_save, yawmean_save)
        ovStateCov_tau_1 = (posecov_x_save, posecov_y_save, yawcov_save)
        return constraints, vertices, A_union, b_union, OVconstraint, direct, ovStateMean_tau_1, ovStateCov_tau_1, 0


    def compute_obstacle_constraints_GMM_affine_scale_ideal(
        self, params, ovehicles, Delta2, Omicron, temp_x, eps_ura, segments, Tsh, ref_traj, Relax = None  
    ):
        """"
            Choose only one constraint that is the most feasible using a reference trajectory
            Ignore the heading uncertainty
            Fix the heading angle in shrinking-horizon
            Scale the covariance
            Use ideal predictions
        """
        constraints = []
        M_big = self.__params.M_big
        T = Tsh
        L, K = self.__params.L, params.K
        truck = {'d': np.array([3.7, 1.79])}
        truck_d = truck['d']
        CAR_R = 4.47213 # this is actually a diameter

        R = 3.4 # EV radius + OV radius
        R = 3.4



        if self.road_boundary_constraints:
            # Auxiliary variable for masking and summing
            Z = cp.Variable(Omicron.shape)

            # Constraints for masking
            mask_constraints = [
                Z[i, j] == (Omicron[i, j] if not segments.mask[i] else 0)
                for i in range(Omicron.shape[0])
                for j in range(Omicron.shape[1])
            ]

            # Summing along axis=0
            Z_sum = M_big * cp.sum(Z, axis=0)

            # Variable and constraints for repeating
            S_big_repeated = cp.Variable((L, T))
            repeat_constraints = [S_big_repeated[i, :] == Z_sum for i in range(L)]

            constraints += mask_constraints
            constraints += repeat_constraints
        else:
            S_big_repeated = np.zeros((L, T), dtype=float)

        Tpred = self.__prediction_horizon
        # ideal prediction
        if T < Tpred:
            ideal_trajs = self.predict_ideal(ovehicles, T, self.__ego_vehicle.id, params)

        # logging.info(f"Tpred:{Tpred}")
        # Let's check if the first prediction passes the intersection or not.
        O = params.O
        OV_injuction = np.empty((O,), dtype=object).tolist()
        direct = np.empty((O,), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):   
            for latent_idx in range(ovehicle.n_states):
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                posemean_x = np.mean(poseData[0::Tpred,0])
                posemean_y = np.mean(poseData[0::Tpred,1])
                if posemean_x >= 190 or posemean_y <= -80: # in Town03_scene4 T inersection.
                # if posemean_x <= -98.122 or (posemean_x <= -83.122 and posemean_y <= -21.178):# or posemean_y >= -21.178: # in Town03_scene3 Star intersection
                #    OV_injuction[ov_idx] = False
                # elif (posemean_x >= -80.122 and posemean_y >= 10.178): # Star intersection
                    OV_injuction[ov_idx] = False 
                else:
                    OV_injuction[ov_idx] = True
                #    direct[ov_idx] = "InJunction"
        

        # logging.info(f"OV{OV_injuction}")
        logging.info(f"K:{params.K}")
        #if OV_injuction:
        OVconstraint = False
        for ov_idx, ovehicle in enumerate(ovehicles):
            OVconstraint = OVconstraint or OV_injuction[ov_idx]

        # Save pose, angle [tau+1|tau] of OV
        posemean_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posemean_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawmean_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawcov_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()

        # save mean_pos of OV and m(tangent)
        mean_p0p1 = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
        tangent = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
        cov_p0p1 = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
        const_idx_save = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()

        pose_data = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()

        halfspace = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()

        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                # load data
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                yawData = np.vstack(ovehicle.pred_yaws[latent_idx])

                posemean_x_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,0])
                posemean_y_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,1])
                yawmean_save[ov_idx][latent_idx] = np.mean(yawData[:,0])
                posecov_x_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,0])
                posecov_y_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,1])
                yawcov_save[ov_idx][latent_idx] = np.cov(yawData[:,0])

        # loaded Data 
        if T < self.__prediction_horizon:
            (mean_p0p1_loaded, tangent_loaded, _, _, const_idx_loaded) = self.load_data(params, self.__ego_vehicle.id)
            num_ov = np.shape(mean_p0p1_loaded)[0]
            num_mode = np.shape(mean_p0p1_loaded)[1]
            mean_differ = np.empty((num_mode,), dtype=object).tolist()
            tangent_saved = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
            const_idx_saved = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
            index_list = np.empty((np.max(O), np.max(K)), dtype=object).tolist()

            for ov_idx, ovehicle in enumerate(ovehicles):
                try:
                    if (mean_p0p1_loaded[ov_idx][0][0] != None).any:
                        for latent_idx in range(ovehicle.n_states):
                            
                            # poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                            poseData = np.vstack(ideal_trajs[ov_idx][latent_idx])
                            Tpred = T
                            for mode_idx in range(num_mode):
                                # logging.info(f"norm:{mean_p0p1_loaded[ov_idx][mode_idx][0]}")
                                mean_differ[mode_idx] = np.linalg.norm((params.x_init[:2]-mean_p0p1_loaded[ov_idx][mode_idx][0]))
                                for t in range(T):
                                    p0 = poseData[t::Tpred,0]
                                    p1 = poseData[t::Tpred,1]
                                    mean_p0 = np.mean(p0)
                                    mean_p1 = np.mean(p1)
                                    mean = np.array([mean_p0, mean_p1])

                                    mean_differ[mode_idx] += np.linalg.norm((mean-mean_p0p1_loaded[ov_idx][mode_idx][t+1]))
                
                            for rest in range(ovehicle.n_states, num_mode):
                                mean_differ[rest] = M_big
                            index_list[ov_idx][latent_idx] = np.argmin(mean_differ) # pick the minmun index
                            tangent_saved[ov_idx][latent_idx] = tangent_loaded[ov_idx][index_list[ov_idx][latent_idx]]
                            const_idx_saved[ov_idx][latent_idx] = const_idx_loaded[ov_idx][index_list[ov_idx][latent_idx]]
                except: # AttributeError: 'bool' object has no attribute 'any'
                    # logging.info(mean_p0p1_loaded[ov_idx][0][0] != None)
                    try:
                        if mean_p0p1_loaded[ov_idx][0][0] != None:

                            for latent_idx in range(ovehicle.n_states):
                                
                                # poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                                poseData = np.vstack(ideal_trajs[ov_idx][latent_idx])
                                Tpred = T
                                for mode_idx in range(num_mode):
                                    # logging.info(f"norm:{mean_p0p1_loaded[ov_idx][mode_idx][0]}")
                                    try:
                                        mean_differ[mode_idx] = np.linalg.norm((params.x_init[:2]-mean_p0p1_loaded[ov_idx][mode_idx][0]))
                                    except: # None
                                        mean_differ[mode_idx] = M_big
                                    for t in range(T):
                                        p0 = poseData[t::Tpred,0]
                                        p1 = poseData[t::Tpred,1]
                                        mean_p0 = np.mean(p0)
                                        mean_p1 = np.mean(p1)
                                        mean = np.array([mean_p0, mean_p1])

                                        mean_differ[mode_idx] += np.linalg.norm((mean-mean_p0p1_loaded[ov_idx][mode_idx][t+1]))
                    
                                for rest in range(ovehicle.n_states, num_mode):
                                    mean_differ[rest] = M_big
                                index_list[ov_idx][latent_idx] = np.argmin(mean_differ) # pick the minmun index
                                tangent_saved[ov_idx][latent_idx] = tangent_loaded[ov_idx][index_list[ov_idx][latent_idx]]
                                const_idx_saved[ov_idx][latent_idx] = const_idx_loaded[ov_idx][index_list[ov_idx][latent_idx]]
                    except:
                        if mean_p0p1_loaded[ov_idx][0][0][0] != None:

                            for latent_idx in range(ovehicle.n_states):
                                
                                # poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                                poseData = np.vstack(ideal_trajs[ov_idx][latent_idx])
                                Tpred = T
                                for mode_idx in range(num_mode):
                                    # logging.info(f"norm:{mean_p0p1_loaded[ov_idx][mode_idx][0]}")
                                    try:
                                        mean_differ[mode_idx] = np.linalg.norm((params.x_init[:2]-mean_p0p1_loaded[ov_idx][mode_idx][0]))
                                        for t in range(T):
                                            p0 = poseData[t::Tpred,0]
                                            p1 = poseData[t::Tpred,1]
                                            mean_p0 = np.mean(p0)
                                            mean_p1 = np.mean(p1)
                                            mean = np.array([mean_p0, mean_p1])

                                            mean_differ[mode_idx] += np.linalg.norm((mean-mean_p0p1_loaded[ov_idx][mode_idx][t+1]))
 
                                    except: # None
                                        mean_differ[mode_idx] = M_big

                   
                                for rest in range(ovehicle.n_states, num_mode):
                                    mean_differ[rest] = M_big
                                index_list[ov_idx][latent_idx] = np.argmin(mean_differ) # pick the minmun index
                                tangent_saved[ov_idx][latent_idx] = tangent_loaded[ov_idx][index_list[ov_idx][latent_idx]]
                                const_idx_saved[ov_idx][latent_idx] = const_idx_loaded[ov_idx][index_list[ov_idx][latent_idx]]

        ovehicles_toSave_moments = []          
        for ov_idx, ovehicle in enumerate(ovehicles):
            ovehicles_toSave_moments.append(copy.deepcopy(ovehicle))
        
        for ov_idx, ovehicle in enumerate(ovehicles):
            if True: #OV_injuction[ov_idx]:
                for latent_idx in range(ovehicle.n_states):
                    # pose_data[ov_idx][latent_idx] = np.vstack(ovehicle.pred_positions[latent_idx])
                    if T < self.__prediction_horizon:
                        poseData = np.vstack(ideal_trajs[ov_idx][latent_idx])
                        ovehicles_toSave_moments[ov_idx].pred_positions[latent_idx] = ideal_trajs[ov_idx][latent_idx]
                        Tpred = T
                        # ovehicle.pred_positions[latent_idx] = ovehicles_toSave_moments[ov_idx].pred_positions[latent_idx]
                    else:
                        poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                    
                    # here?
                    pose_data[ov_idx][latent_idx] = np.vstack(ovehicle.pred_positions[latent_idx])
                    # logging.info(f"Tpred:{Tpred}")
                    # logging.info(f"shape:{np.shape(poseData[0::Tpred,0])}")
                    for t in range(T):   
                        
                        const_idx = -1 # This is for checking if we loaded const_idx                   
                        # yawData = np.vstack(ovehicle.pred_yaws[latent_idx])
                        # here?
                        # pose_data[ov_idx][latent_idx] = poseData

                        # delta_ijt = Delta2[(ov_idx, latent_idx, t)]          
                        eps_ijt = eps_ura[ov_idx, latent_idx] / self.__prediction_horizon
                        Gamma_ijt = norm.ppf(1-eps_ijt) #np.sqrt((1-eps_ijt)/eps_ijt) #norm.pdf(norm.ppf(1-eps_ijt))/eps_ijt #norm.ppf(1-eps_ijt)
                    
                        
                        p0 = poseData[t::Tpred,0]
                        p1 = poseData[t::Tpred,1]
                        # logging.info(f"t:{t}")
                        # logging.info(f"Tpred:{Tpred}")
                        # logging.info(f"p1_t:{np.shape(p1)}")
                        mean_p0 = np.mean(p0)
                        mean_p1 = np.mean(p1)
                        
                        mean = np.array([mean_p0, mean_p1])

                        ## scale
                        scale = 1.0
                        cov_infer_0 = np.identity(n=2)
                    
                        for tau_future in range(t):
                        # if t >= 1:
                            # tau_future = t - 1
                            p0_tau = poseData[tau_future::Tpred,0]
                            p1_tau = poseData[tau_future::Tpred,1]
                            # logging.info(f"p1_tau:{np.shape(p1_tau)}")
                            p_t_tau = [p0, p1, p0_tau, p1_tau]

                            # predict moments
                            cov_infer, cov_mu, cov_t = makeconstraint.predict_moments(p_t_tau)

                            # scaling
                            scale_temp = makeconstraint.compute_scale(cov_infer, cov_mu, cov_t, Gamma_ijt, target_p = 0.9999)
                            # logging.info(f"tau_future:{tau_future}, scale_temp:{scale_temp}")
                            scale = np.max((scale_temp, scale))
                            if tau_future == 0:
                                cov_infer_0 =cov_infer

                        cov = np.cov([p0, p1])
                        cov_fro = np.linalg.norm(cov, 'fro')
                        cov_fro_sqrt = np.sqrt(cov_fro)
                        # logging.info(f"t:{t}, mean: {mean}, cov: {cov}")
                        # logging.info(f"cof_infer: {cov_infer_0}")
                        cov_fro_infer = np.linalg.norm(cov_infer_0, 'fro')
                        cov_fro_sqrt_infer = np.sqrt(cov_fro_infer)
                        # logging.info(f"scale:{scale}, cov:{cov_fro_sqrt}, cov_infer:{cov_fro_sqrt_infer}")
                        # scale = 1.0
                        cov = scale*np.cov([p0, p1])
                        cov_fro = np.linalg.norm(cov, 'fro')
                        cov_fro_sqrt = np.sqrt(cov_fro)

                        if T == self.__prediction_horizon:
                            m = - (ref_traj[t][0] - mean_p0) / (ref_traj[t][1] - mean_p1)
                            const_idx = None
                        else:
                            try:
                                if (mean_p0p1_loaded[ov_idx][0][0] != None).any():
                                    m = tangent_saved[ov_idx][latent_idx][t+1]
                                    const_idx = const_idx_saved[ov_idx][latent_idx][t+1]
                                    # logging.info(f"m:{m}")
                                else:
                                    m = - (ref_traj[t][0] - mean_p0) / (ref_traj[t][1] - mean_p1)
                            except:
                                try: 
                                    if mean_p0p1_loaded[ov_idx][0][0] != None:
                                        m = tangent_saved[ov_idx][latent_idx][t+1]
                                        const_idx = const_idx_saved[ov_idx][latent_idx][t+1]
                                        # logging.info(f"m:{m}")
                                    else:
                                        m = - (ref_traj[t][0] - mean_p0) / (ref_traj[t][1] - mean_p1)
                                except:
                                    if mean_p0p1_loaded[ov_idx][0][0][0] != None:
                                        m = tangent_saved[ov_idx][latent_idx][t+1]
                                        const_idx = const_idx_saved[ov_idx][latent_idx][t+1]
                                        # logging.info(f"m:{m}")
                                    else:
                                        m = - (ref_traj[t][0] - mean_p0) / (ref_traj[t][1] - mean_p1)

                        M = np.array([m, -1])

                        
                        # circle
                        Sigma = np.identity(n=2)
                        ref_pose = np.array((ref_traj[t][0], ref_traj[t][1])) 
                        # logging.info(f"ref_pose:{ref_pose}, m:{m}, mean: {mean}")
                        # logging.info(f"mean:{mean}")
                        n_star, d_star, const_idx = makeconstraint.choose_closest_tangent(mean, Sigma, R, m, ref_pose, const_idx)
                        # logging.info(f"idx:{which_idx}")
                        # outer region of ov
                        if n_star @ mean <= d_star:
                            constraints += [
                                n_star.T @ cp.vstack([temp_x[t][0], temp_x[t][1]]) >= d_star + Gamma_ijt * cov_fro_sqrt * cp.norm(M.T, p=2) + S_big_repeated[0, t]\
                                # - Relax[latent_idx,t]# + R*np.sqrt(m*m+1) 
                            ]
                            value = n_star[0] * ref_pose[0] + n_star[1] * ref_pose[1] - d_star - Gamma_ijt * cov_fro_sqrt * np.linalg.norm(M.T, 2)
                            # logging.info(f"which_idx:{which_idx}")
                            # logging.info(f"{value}>=0")
                            A = - n_star
                            b = -( d_star + Gamma_ijt * cov_fro_sqrt * np.linalg.norm(M.T, 2) + S_big_repeated[0, t])
                        else:
                            constraints += [
                                n_star.T @ cp.vstack([temp_x[t][0], temp_x[t][1]]) <= d_star - Gamma_ijt * cov_fro_sqrt * cp.norm(M.T, p=2) + S_big_repeated[0, t]\
                                # + Relax[latent_idx,t]# - R*np.sqrt(m*m+1)
                            ]
                            value = n_star[0] * ref_pose[0] + n_star[1] * ref_pose[1] - d_star + Gamma_ijt * cov_fro_sqrt * np.linalg.norm(M.T, 2)
                            # logging.info(f"which_idx:{which_idx}")
                            # logging.info(f"{value}<=0")
                            A =  n_star
                            b = ( d_star - Gamma_ijt * cov_fro_sqrt *  np.linalg.norm(M.T, 2) + S_big_repeated[0, t])
                        
                        # save halfspace

                        halfspace[ov_idx][latent_idx][t] = [A[0], A[1], b]
                        # logging.info(f"t={t}, halfspace={halfspace}")
                        # val1 = m*(ref_traj[t][0] - mean_p0) - (ref_traj[t][1] - mean_p1)
                        # val2 = - m*(ref_traj[t][0] - mean_p0) + (ref_traj[t][1] - mean_p1)

                        # val = [val1  + R*np.sqrt(m*m+1)+ Gamma_ijt * cov_fro_sqrt, val2 + R*np.sqrt(m*m+1)+ Gamma_ijt * cov_fro_sqrt]
                        # logging.info(f"val:{val}")
                        # if const_idx == -1:
                        #     const_idx = val.index(min(val))

                        # if 0==const_idx:
                        #     constraints += [m*(temp_x[t][0] - mean_p0) - (temp_x[t][1] - mean_p1) + R*np.sqrt(m*m+1)+ Gamma_ijt * cov_fro_sqrt * cp.norm(M.T, p=2) <= S_big_repeated[0, t]]
                        # elif 1==const_idx:
                        #     constraints += [-m*(temp_x[t][0] - mean_p0) + (temp_x[t][1] - mean_p1) + R*np.sqrt(m*m+1)+ Gamma_ijt * cov_fro_sqrt * cp.norm(M.T, p=2) <= S_big_repeated[0, t]]

                        # save mean and tangent
                        mean_p0p1[ov_idx][latent_idx][t] = mean
                        tangent[ov_idx][latent_idx][t] = m
                        cov_p0p1[ov_idx][latent_idx][t] = cov/scale # save the original cov
                        const_idx_save[ov_idx][latent_idx][t] = const_idx
                        # constraints += [m*(temp_x[t][0] - mean_p0) - (temp_x[t][1] - mean_p1) + R*np.sqrt(m*m+1)+ Gamma_ijt * cov_fro_sqrt * cp.norm(M.T, p=2) <= S_big_repeated[0, t]]

            else:
                constraints += []
        self.__halfspace = halfspace
        # constraints = []
        vertices = self.__compute_vertices(params, ovehicles)
        A_union, b_union = self.__compute_overapproximations(
            params, ovehicles, vertices
        )

        # save moments
        self.save_moments(ovehicles_toSave_moments, params.O, K, T, self.__prediction_horizon, self.__ego_vehicle.id, params)

        

        ovStateMean_tau_1 = (posemean_x_save, posemean_y_save, yawmean_save)
        ovStateCov_tau_1 = (posecov_x_save, posecov_y_save, yawcov_save)
        meanNtangent = (mean_p0p1, tangent, cov_p0p1, 0, const_idx_save)
        return constraints, vertices, A_union, b_union, OVconstraint, direct, ovStateMean_tau_1, ovStateCov_tau_1, meanNtangent

    def compute_obstacle_constraints_GMM_affine_robust(
        self, params, ovehicles, Delta2, Omicron, temp_x, eps_ura, segments, Tsh, ref_traj     
    ):
        """"
            Choose only one constraint that is the most feasible using a reference trajectory
            ignore the heading uncertainty
            Fix the heading angle in shrinking-horizon
        """
        constraints = []
        M_big = self.__params.M_big
        T = Tsh
        L, K = self.__params.L, params.K
        truck = {'d': np.array([3.7, 1.79])}
        truck_d = truck['d']
        CAR_R = 4.47213 # this is actually a diameter

        R = 3.4 # EV radius + OV radius

        if self.road_boundary_constraints:
            # Auxiliary variable for masking and summing
            Z = cp.Variable(Omicron.shape)

            # Constraints for masking
            mask_constraints = [
                Z[i, j] == (Omicron[i, j] if not segments.mask[i] else 0)
                for i in range(Omicron.shape[0])
                for j in range(Omicron.shape[1])
            ]

            # Summing along axis=0
            Z_sum = M_big * cp.sum(Z, axis=0)

            # Variable and constraints for repeating
            S_big_repeated = cp.Variable((L, T))
            repeat_constraints = [S_big_repeated[i, :] == Z_sum for i in range(L)]

            constraints += mask_constraints
            constraints += repeat_constraints
        else:
            S_big_repeated = np.zeros((L, T), dtype=float)

        Tpred = self.__prediction_horizon
        # ideal prediction
        if T < Tpred:
            ideal_trajs = self.predict_ideal(ovehicles, T, self.__ego_vehicle.id, params)

        # logging.info(f"Tpred:{Tpred}")
        # Let's check if the first prediction passes the intersection or not.
        O = params.O
        OV_injuction = np.empty((O,), dtype=object).tolist()
        direct = np.empty((O,), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):   
            for latent_idx in range(ovehicle.n_states):
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                posemean_x = np.mean(poseData[0::Tpred,0])
                posemean_y = np.mean(poseData[0::Tpred,1])
                if posemean_x >= 190 or posemean_y <= -80: # in Town03_scene4 T inersection.
                # if posemean_x <= -98.122 or (posemean_x <= -83.122 and posemean_y <= -21.178):# or posemean_y >= -21.178: # in Town03_scene3 Star intersection
                #    OV_injuction[ov_idx] = False
                # elif (posemean_x >= -80.122 and posemean_y >= 10.178): # Star intersection
                    OV_injuction[ov_idx] = False 
                else:
                    OV_injuction[ov_idx] = True
                #    direct[ov_idx] = "InJunction"
        

        #logging.info(f"OV{OV_injuction}")
        logging.info(f"K:{params.K}")
        #if OV_injuction:
        OVconstraint = False
        for ov_idx, ovehicle in enumerate(ovehicles):
            OVconstraint = OVconstraint or OV_injuction[ov_idx]

        # Save pose, angle [tau+1|tau] of OV
        posemean_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posemean_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawmean_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawcov_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()

        # save mean_pos of OV and m(tangent)
        mean_p0p1 = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
        tangent = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
        cov_p0p1 = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
        const_idx_save = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()

        pose_data = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()

        halfspace = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()

        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                # load data
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                yawData = np.vstack(ovehicle.pred_yaws[latent_idx])

                posemean_x_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,0])
                posemean_y_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,1])
                yawmean_save[ov_idx][latent_idx] = np.mean(yawData[:,0])
                posecov_x_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,0])
                posecov_y_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,1])
                yawcov_save[ov_idx][latent_idx] = np.cov(yawData[:,0])

        # loaded Data 
        if T < self.__prediction_horizon:
            (mean_p0p1_loaded, tangent_loaded, _, _, const_idx_loaded) = self.load_data(params, self.__ego_vehicle.id)
            num_ov = np.shape(mean_p0p1_loaded)[0]
            num_mode = np.shape(mean_p0p1_loaded)[1]
            mean_differ = np.empty((num_mode,), dtype=object).tolist()
            tangent_saved = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
            const_idx_saved = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
            index_list = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
            for ov_idx, ovehicle in enumerate(ovehicles):
                try:
                    if (mean_p0p1_loaded[ov_idx][0][0] != None).any:
                        for latent_idx in range(ovehicle.n_states):
                            
                            # poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                            poseData = np.vstack(ideal_trajs[ov_idx][latent_idx])
                            Tpred = T
                            for mode_idx in range(num_mode):
                                # logging.info(f"norm:{mean_p0p1_loaded[ov_idx][mode_idx][0]}")
                                mean_differ[mode_idx] = np.linalg.norm((params.x_init[:2]-mean_p0p1_loaded[ov_idx][mode_idx][0]))
                                for t in range(T):
                                    p0 = poseData[t::Tpred,0]
                                    p1 = poseData[t::Tpred,1]
                                    mean_p0 = np.mean(p0)
                                    mean_p1 = np.mean(p1)
                                    mean = np.array([mean_p0, mean_p1])

                                    mean_differ[mode_idx] += np.linalg.norm((mean-mean_p0p1_loaded[ov_idx][mode_idx][t+1]))
                
                            for rest in range(ovehicle.n_states, num_mode):
                                mean_differ[rest] = M_big
                            index_list[ov_idx][latent_idx] = np.argmin(mean_differ) # pick the minmun index
                            tangent_saved[ov_idx][latent_idx] = tangent_loaded[ov_idx][index_list[ov_idx][latent_idx]]
                            const_idx_saved[ov_idx][latent_idx] = const_idx_loaded[ov_idx][index_list[ov_idx][latent_idx]]
                except: # AttributeError: 'bool' object has no attribute 'any'
                    # logging.info(mean_p0p1_loaded[ov_idx][0][0] != None)
                    try:
                        if mean_p0p1_loaded[ov_idx][0][0] != None:

                            for latent_idx in range(ovehicle.n_states):
                                
                                # poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                                poseData = np.vstack(ideal_trajs[ov_idx][latent_idx])
                                Tpred = T
                                for mode_idx in range(num_mode):
                                    # logging.info(f"norm:{mean_p0p1_loaded[ov_idx][mode_idx][0]}")
                                    try:
                                        mean_differ[mode_idx] = np.linalg.norm((params.x_init[:2]-mean_p0p1_loaded[ov_idx][mode_idx][0]))
                                    except: # None
                                        mean_differ[mode_idx] = M_big
                                    for t in range(T):
                                        p0 = poseData[t::Tpred,0]
                                        p1 = poseData[t::Tpred,1]
                                        mean_p0 = np.mean(p0)
                                        mean_p1 = np.mean(p1)
                                        mean = np.array([mean_p0, mean_p1])

                                        mean_differ[mode_idx] += np.linalg.norm((mean-mean_p0p1_loaded[ov_idx][mode_idx][t+1]))
                    
                                for rest in range(ovehicle.n_states, num_mode):
                                    mean_differ[rest] = M_big
                                index_list[ov_idx][latent_idx] = np.argmin(mean_differ) # pick the minmun index
                                tangent_saved[ov_idx][latent_idx] = tangent_loaded[ov_idx][index_list[ov_idx][latent_idx]]
                                const_idx_saved[ov_idx][latent_idx] = const_idx_loaded[ov_idx][index_list[ov_idx][latent_idx]]
                    except:
                        if mean_p0p1_loaded[ov_idx][0][0][0] != None:

                            for latent_idx in range(ovehicle.n_states):
                                
                                # poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                                poseData = np.vstack(ideal_trajs[ov_idx][latent_idx])
                                Tpred = T
                                for mode_idx in range(num_mode):
                                    # logging.info(f"norm:{mean_p0p1_loaded[ov_idx][mode_idx][0]}")
                                    try:
                                        mean_differ[mode_idx] = np.linalg.norm((params.x_init[:2]-mean_p0p1_loaded[ov_idx][mode_idx][0]))
                                        for t in range(T):
                                            p0 = poseData[t::Tpred,0]
                                            p1 = poseData[t::Tpred,1]
                                            mean_p0 = np.mean(p0)
                                            mean_p1 = np.mean(p1)
                                            mean = np.array([mean_p0, mean_p1])

                                            mean_differ[mode_idx] += np.linalg.norm((mean-mean_p0p1_loaded[ov_idx][mode_idx][t+1]))
 
                                    except: # None
                                        mean_differ[mode_idx] = M_big

                   
                                for rest in range(ovehicle.n_states, num_mode):
                                    mean_differ[rest] = M_big
                                index_list[ov_idx][latent_idx] = np.argmin(mean_differ) # pick the minmun index
                                tangent_saved[ov_idx][latent_idx] = tangent_loaded[ov_idx][index_list[ov_idx][latent_idx]]
                                const_idx_saved[ov_idx][latent_idx] = const_idx_loaded[ov_idx][index_list[ov_idx][latent_idx]]
        
        ovehicles_toSave_moments = []          
        for ov_idx, ovehicle in enumerate(ovehicles):
            ovehicles_toSave_moments.append(copy.deepcopy(ovehicle))    

        
        for ov_idx, ovehicle in enumerate(ovehicles):
            if True: #OV_injuction[ov_idx]:
                for latent_idx in range(ovehicle.n_states):
                    # pose_data[ov_idx][latent_idx] = np.vstack(ovehicle.pred_positions[latent_idx])
                    if T < self.__prediction_horizon:
                        poseData = np.vstack(ideal_trajs[ov_idx][latent_idx])
                        ovehicles_toSave_moments[ov_idx].pred_positions[latent_idx] = ideal_trajs[ov_idx][latent_idx]
                        Tpred = T
                        # ovehicle.pred_positions[latent_idx] = ovehicles_toSave_moments[ov_idx].pred_positions[latent_idx]
                    else:
                        poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                    
                    for t in range(T):   
                        const_idx = -1 # This is for checking if we loaded const_idx                   
                        # yawData = np.vstack(ovehicle.pred_yaws[latent_idx])
                        # poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                        # pose_data[ov_idx][latent_idx][t] = poseData

                        # delta_ijt = Delta2[(ov_idx, latent_idx, t)]          
                        eps_ijt = eps_ura[ov_idx, latent_idx] / self.__prediction_horizon
                        Gamma_ijt = norm.ppf(1-eps_ijt) #np.sqrt((1-eps_ijt)/eps_ijt) #norm.pdf(norm.ppf(1-eps_ijt))/eps_ijt #norm.ppf(1-eps_ijt)
                    
                        
                        p0 = poseData[t::Tpred,0]
                        p1 = poseData[t::Tpred,1]
                        

                        mean_p0 = np.mean(p0)
                        mean_p1 = np.mean(p1)
                        
                        mean = np.array([mean_p0, mean_p1])

                        cov = np.cov([p0, p1])
                        cov_fro = np.linalg.norm(cov, 'fro')
                        cov_fro_sqrt = np.sqrt(cov_fro)

                        if T == self.__prediction_horizon:
                            m = - (ref_traj[t][0] - mean_p0) / (ref_traj[t][1] - mean_p1)
                            const_idx = None
                        else:
                            try:
                                if (mean_p0p1_loaded[ov_idx][0][0] != None).any():
                                    m = tangent_saved[ov_idx][latent_idx][t+1]
                                    const_idx = const_idx_saved[ov_idx][latent_idx][t+1]
                                    # logging.info(f"m:{m}")
                                else:
                                    m = - (ref_traj[t][0] - mean_p0) / (ref_traj[t][1] - mean_p1)
                            except:
                                try: 
                                    if mean_p0p1_loaded[ov_idx][0][0] != None:
                                        m = tangent_saved[ov_idx][latent_idx][t+1]
                                        const_idx = const_idx_saved[ov_idx][latent_idx][t+1]
                                        # logging.info(f"m:{m}")
                                    else:
                                        m = - (ref_traj[t][0] - mean_p0) / (ref_traj[t][1] - mean_p1)
                                except:
                                    if mean_p0p1_loaded[ov_idx][0][0][0] != None:
                                        m = tangent_saved[ov_idx][latent_idx][t+1]
                                        const_idx = const_idx_saved[ov_idx][latent_idx][t+1]
                                        # logging.info(f"m:{m}")
                                    else:
                                        m = - (ref_traj[t][0] - mean_p0) / (ref_traj[t][1] - mean_p1)

                        M = np.array([m, -1])

                        # val1 = m*(ref_traj[t][0] - mean_p0) - (ref_traj[t][1] - mean_p1)
                        # val2 = - m*(ref_traj[t][0] - mean_p0) + (ref_traj[t][1] - mean_p1)

                        # val = [val1, val2]
                        # if const_idx == -1:
                        #     const_idx = val.index(min(val))

                        # if 0==const_idx:
                        #     constraints += [m*(temp_x[t][0] - mean_p0) - (temp_x[t][1] - mean_p1) + R*np.sqrt(m*m+1)+ Gamma_ijt * cov_fro_sqrt * cp.norm(M.T, p=2) <= S_big_repeated[0, t]]
                        # elif 1==const_idx:
                        #     constraints += [-m*(temp_x[t][0] - mean_p0) + (temp_x[t][1] - mean_p1) + R*np.sqrt(m*m+1)+ Gamma_ijt * cov_fro_sqrt * cp.norm(M.T, p=2) <= S_big_repeated[0, t]]


                        Sigma = np.identity(n=2)
                        ref_pose = np.array((ref_traj[t][0], ref_traj[t][1])) 

                        n_star, d_star, const_idx = makeconstraint.choose_closest_tangent(mean, Sigma, R, m, ref_pose, const_idx)
                        # logging.info(f"m:{m}")
                        if n_star @ mean <= d_star:
                            constraints += [
                                n_star.T @ cp.vstack([temp_x[t][0], temp_x[t][1]]) >= d_star + Gamma_ijt * cov_fro_sqrt * cp.norm(M.T, p=2) \
                                # + R*np.sqrt(m*m+1)
                            ]
                            value = n_star[0] * ref_pose[0] + n_star[1] * ref_pose[1] - d_star - Gamma_ijt * cov_fro_sqrt * np.linalg.norm(M.T, 2)
                            # logging.info(f"{value}>=0")
                            # logging.info(f"-Gamma_ijt * cov_fro_sqrt * np.linalg.norm(M.T, 2):{-Gamma_ijt * cov_fro_sqrt * np.linalg.norm(M.T, 2)}")
                            # logging.info(f"cov_fro:{cov_fro_sqrt}")
                            A = -n_star
                            b = - ( d_star + Gamma_ijt * cov_fro_sqrt *  np.linalg.norm(M.T, 2))
                        else:
                            constraints += [
                                n_star.T @ cp.vstack([temp_x[t][0], temp_x[t][1]]) <= d_star - Gamma_ijt * cov_fro_sqrt * cp.norm(M.T, p=2) \
                                # - R*np.sqrt(m*m+1)
                            ]
                            value = n_star[0] * ref_pose[0] + n_star[1] * ref_pose[1] - d_star + Gamma_ijt * cov_fro_sqrt * np.linalg.norm(M.T, 2)
                            # logging.info(f"{value}<=0")                           
                            # logging.info(f"Gamma_ijt * cov_fro_sqrt * np.linalg.norm(M.T, 2):{Gamma_ijt * cov_fro_sqrt * np.linalg.norm(M.T, 2)}")
                            # logging.info(f"cov_fro:{cov_fro_sqrt}")
                            A =  n_star
                            b = ( d_star - Gamma_ijt * cov_fro_sqrt *  np.linalg.norm(M.T, 2) )
                        
                        # save halfspace
                        halfspace[ov_idx][latent_idx][t] = [A[0], A[1], b]

                        # save mean and tangent
                        mean_p0p1[ov_idx][latent_idx][t] = mean
                        tangent[ov_idx][latent_idx][t] = m
                        cov_p0p1[ov_idx][latent_idx][t] = cov
                        const_idx_save[ov_idx][latent_idx][t] = const_idx
                        # constraints += [m*(temp_x[t][0] - mean_p0) - (temp_x[t][1] - mean_p1) + R*np.sqrt(m*m+1)+ Gamma_ijt * cov_fro_sqrt * cp.norm(M.T, p=2) <= S_big_repeated[0, t]]

            else:
                constraints += []
        # constraints = []
        self.__halfspace = halfspace
        
        vertices = self.__compute_vertices(params, ovehicles)
        A_union, b_union = self.__compute_overapproximations(
            params, ovehicles, vertices
        )

        # save moments
        self.save_moments(ovehicles_toSave_moments, params.O, K, T, self.__prediction_horizon, self.__ego_vehicle.id, params)

        ovStateMean_tau_1 = (posemean_x_save, posemean_y_save, yawmean_save)
        ovStateCov_tau_1 = (posecov_x_save, posecov_y_save, yawcov_save)
        meanNtangent = (mean_p0p1, tangent, cov_p0p1, 0, const_idx_save)
        return constraints, vertices, A_union, b_union, OVconstraint, direct, ovStateMean_tau_1, ovStateCov_tau_1, meanNtangent

    def compute_objective(self, X, U, goal, Tsh):
        """Set the objective."""
        obj = self.__params.objective
        T = Tsh
        R1 = cp.Constant([[obj.w_accel, obj.w_joint], [obj.w_joint, obj.w_turning]])
        R2 = cp.Constant([[obj.w_ch_accel, obj.w_ch_joint], [obj.w_ch_joint, obj.w_ch_turning]])

        # final destination objective
        cost = (
            obj.w_final * (X[-1, 0] - goal[0]) ** 2
            + obj.w_final * (X[-1, 1] - goal[1]) ** 2
        )
        # Control effort
        cost += cp.sum([cp.quad_form(U[t], R1) for t in range(T)])
        # Control change
        cost += cp.sum([cp.quad_form(U[t] - U[t-1], R2) for t in range(1, T)])
        return cost
    
    def compute_objective_referenceTraj(self, X, U, ref_traj, goal, Tsh):
        """Set the objective."""
        obj = self.__params.objective
        T = Tsh
        R1 = cp.Constant([[obj.w_accel, obj.w_joint], [obj.w_joint, obj.w_turning]])
        R2 = cp.Constant([[obj.w_ch_accel, obj.w_ch_joint], [obj.w_ch_joint, obj.w_ch_turning]])

        cost = (
            obj.w_final * (X[-1, 0] - goal[0]) ** 2
            + obj.w_final * (X[-1, 1] - goal[1]) ** 2
        )
        # Cost reference trajectory
        for t in range(Tsh):
        
            if t < len(ref_traj):
                cost += (
                    obj.w_ref* (X[t, 0] - ref_traj[t][0]) ** 2
                    + obj.w_ref * (X[t, 1] - ref_traj[t][1]) ** 2
                )
            else:
                cost += (
                    obj.w_ref* (X[t, 0] - ref_traj[-1][0]) ** 2
                    + obj.w_ref * (X[t, 1] - ref_traj[-1][1]) ** 2
                )
        # Control effort
        cost += cp.sum([cp.quad_form(U[t], R1) for t in range(T)])
        # Control change
        cost += cp.sum([cp.quad_form(U[t] - U[t-1], R2) for t in range(1, T)])

        return cost

    def compute_objective_referenceTraj_Relax(self, X, U, ref_traj, goal, Tsh, Relax):
        """Set the objective."""
        obj = self.__params.objective
        T = Tsh
        R1 = cp.Constant([[obj.w_accel, obj.w_joint], [obj.w_joint, obj.w_turning]])
        R2 = cp.Constant([[obj.w_ch_accel, obj.w_ch_joint], [obj.w_ch_joint, obj.w_ch_turning]])

        cost = (
            obj.w_final * (X[-1, 0] - goal[0]) ** 2
            + obj.w_final * (X[-1, 1] - goal[1]) ** 2
        )
        # Cost reference trajectory
        for t in range(Tsh):
        
            if t < len(ref_traj):
                cost += (
                    obj.w_ref* (X[t, 0] - ref_traj[t][0]) ** 2
                    + obj.w_ref * (X[t, 1] - ref_traj[t][1]) ** 2
                )
            else:
                cost += (
                    obj.w_ref* (X[t, 0] - ref_traj[-1][0]) ** 2
                    + obj.w_ref * (X[t, 1] - ref_traj[-1][1]) ** 2
                )

        # Control effort
        cost += cp.sum([cp.quad_form(U[t], R1) for t in range(T)])
        # Control change
        cost += cp.sum([cp.quad_form(U[t] - U[t-1], R2) for t in range(1, T)])
        # Relax
        # cost += Relax[0,0]
        for latent_idx in range(1):
            for t in range(T):
                cost += 100_000_000*Relax[latent_idx, t]**2

        
        return cost

    def load_data(self, params, ego_vehicle_id):
        filename = "agent" + str(ego_vehicle_id) + "_frame" + str(params.frame-10) + "_cov"
        filepath = os.path.join("out/data", filename)
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
                meanNtangent = loaded_data['meanNtangent']
        except Exception as e:
            logging.info(f"Error while loading: {str(e)}")
        
        return meanNtangent

    def save_data(self, data_save, params, ego_vehicle_id):
        filename_cov = f"agent{ego_vehicle_id}_frame{params.frame}_cov"
        #filename_cov = f"100_frame{params.frame}_cov"
        filepath_cov = os.path.join("out/data",filename_cov)
        try:
            with open(filepath_cov,"wb") as f:
                pickle.dump(data_save,f)
        except Exception as e:
            logging.info(f"Error while saving: {str(e)}")

    # def save_ovehicles(self, ovehicles, params, ego_vehicle_id):
    #     """
    #     Save ovehicles for loading offline ovehicles
    #     """
    #     filename_ovehicle = f"agent{}"

    def save_moments(self, ovehicles, O, K, T, Tpred, ego_vehicle_id, params):
        """
        Save mean, covariance, and cross-covariance of predictions
        """
        # estimate moments
        mean_p0p1 = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
        cov_p0p1 = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
        cross_cov = np.empty((np.max(O), np.max(K), T, T-1), dtype=object).tolist()

        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                for t in range(T):
                    p0_t = poseData[t::T,0]
                    p1_t = poseData[t::T,1]

                    p_t = [p0_t, p1_t]
                    mean_t = np.mean(p_t, axis = 1)
                    cov_t = np.cov(p_t)

                    mean_p0p1[ov_idx][latent_idx][t] = mean_t
                    cov_p0p1[ov_idx][latent_idx][t] = cov_t

                    for tau_future in range(t):
                        p0_tau = poseData[tau_future::T,0]
                        p1_tau = poseData[tau_future::T,1]
                        p_t_tau = [p0_t, p1_t, p0_tau, p1_tau]
                        cov = np.cov(p_t_tau)

                        # predict moments
                        cov_t_tau = cov[0:2, 2:4]   # Sigma^{(t|tau)(tau+1|tau)} = cov(mu^{t|tau+1})
                        cross_cov[ov_idx][latent_idx][t][tau_future] = cov_t_tau
        moments_save = {}
        moments_save['mean_p0p1'] = mean_p0p1
        moments_save['cov_p0p1'] = cov_p0p1
        moments_save['cross_cov'] = cross_cov

        filename = f"agent{ego_vehicle_id}_frame{params.frame}_moments"
        filepath = os.path.join("out/data",filename)
        try:
            with open(filepath, "wb") as f:
                pickle.dump(moments_save, f)
        except Exception as e:
            logging.info(f"Error while saving: {str(e)}")

    def predict_ideal(self, ovehicles, T, ego_vehicle_id, params):
        """
            Predict trajectories using saved moments
        """

        # loading moments
        filename = f"agent{ego_vehicle_id}_frame{params.frame-10}_moments"
        filepath = os.path.join("out/data", filename)
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
                mean_p0p1 = loaded_data['mean_p0p1']
                cov_p0p1 = loaded_data['cov_p0p1']
                cross_cov = loaded_data['cross_cov']
        except Exception as e:
            logging.info(f"Error while loading: {str(e)}")

        # generate samples
        Tsh = T
        traj_all = {}
        n_samples = 1_000_000
        for ov_idx, ovehicle in enumerate(ovehicles):

            # --- (A) Load the current position
            traj_all.setdefault(ov_idx, {})
            x_cur = ovehicle.past[-1]

            # how many modes
            n_latent = len(mean_p0p1[ov_idx])

            for latent_idx in range(ovehicle.n_states):
                # if latent_idx is out of range, fall back to latent_idx - 1
                if latent_idx < n_latent:
                    data_idx = latent_idx
                else:
                    data_idx = max(n_latent - 1, 0)
                    logging.info(f"K changed, latent_idx:{latent_idx}")

                # observe
                # x_cur_batch = x_cur*np.ones((n_samples, 2))
                
                # sample
                mean_0 = mean_p0p1[ov_idx][data_idx][0]   # (2,) vector
                cov_0  = cov_p0p1[ov_idx][data_idx][0]    # (2,2) matrix
                x_cur = np.random.multivariate_normal(mean_0, cov_0, size=1)
                x_cur_batch = x_cur*np.ones((n_samples, 2))         

                traj_batch = np.zeros((n_samples, T, 2))
                traj_batch[:, 0, :] = x_cur_batch

                #  --- (B) Iterate for t=0 ... Tsh-1 to update all samples at once ---
                for t in range(0, Tsh):
                    # (B1) Distribution parameters at previous time τ = t
                    mean_tau = mean_p0p1[ov_idx][data_idx][t]    # (2,)
                    cov_tau  = cov_p0p1[ov_idx][data_idx][t]     # (2,2)

                    # (B2) Distribution parameters at next time τ+1
                    mean_next = mean_p0p1[ov_idx][data_idx][t+1] # (2,)
                    cov_next  = cov_p0p1[ov_idx][data_idx][t+1]  # (2,2)

                    # (B3) Cross‐covariance: Cov(x_{τ+1}, x_{τ})
                    C_t_tp1 = cross_cov[ov_idx][data_idx][t+1][t]  # (2,2)

                    # (B4) Inverse of cov_tau
                    cov_tau_inv = np.linalg.inv(cov_tau)

                    # (B5) Vectorized conditional mean:
                    #     cond_mean_i = mean_next + C_t_tp1 @ inv(cov_tau) @ ( x_cur_batch_i - mean_tau )
                    diff = x_cur_batch - mean_tau                 # shape (n_samples, 2)
                    A = C_t_tp1 @ cov_tau_inv                      # shape (2,2)
                    cond_mean = mean_next + (diff @ A.T)           # shape (n_samples, 2)

                    # (B6) Conditional covariance (same for all samples)
                    cond_cov = cov_next - A @ C_t_tp1.T            # shape (2,2)

                    # (B7) cond_cov의 Cholesky 분해: L @ L^T = cond_cov
                    L = np.linalg.cholesky(cond_cov)               # shape (2,2)

                    # (B8) Draw standard normal noise Z in one shot: shape (n_samples,2)
                    rng = np.random.RandomState(seed=None)                  # 혹은 미리 만들어둔 시드
                    Z = rng.standard_normal((n_samples, 2))       # shape (n_samples, 2)

                    # (B9) Sample next x_cur_batch: cond_mean + Z @ L.T
                    x_cur_batch = cond_mean + (Z @ L.T)            # (n_samples, 2)

                    # (B10) Store into traj_batch
                    traj_batch[:, t, :] = x_cur_batch

                # --- (C) Save results ---
                traj_all[ov_idx][latent_idx] = traj_batch

        return traj_all


    def save_x_init(self, x_init, frame, ego_vehicle_id):
        filename_cov = f"agent{ego_vehicle_id}_frame{frame}_x_init"
        #filename_cov = f"100_frame{params.frame}_cov"
        filepath_cov = os.path.join("out/data",filename_cov)
        try:
            with open(filepath_cov,"wb") as f:
                pickle.dump(x_init,f)
        except Exception as e:
            logging.info(f"Error while saving: {str(e)}")

    def load_refT(self, offline_idx, Tsh, x_init):
        filename = "refT"
        filepath = os.path.join("out/data/referenceTrajectory/T", filename)
        try:
            with open(filepath,"rb") as f:
                ref = pickle.load(f)
        except Exception as e:
            logging.info(f"Error while saving: {str(e)}")
        
        diff = np.array(ref).T[0:2] - np.array([x_init[0]*np.ones((1,len(ref))),x_init[1]*np.ones((1,len(ref)))]).reshape(2,len(ref))
        norm2 = linalg.norm(diff.T, ord=2, axis=1)
        # index_min = np.argmin(norm2)
        # the closest ahead point
        sorted_index = np.argsort(norm2)
        if sorted_index[0] < sorted_index[2]:
            index_min = sorted_index[0]
        elif sorted_index[0] > sorted_index[2]:
            index_min = sorted_index[1]

        self.refT = ref[index_min:index_min+ Tsh]

        # self.refT =  ref[int(offline_idx):int(offline_idx + Tsh)]

    def enforceCondition(self, mean0, mean1, cov0, cov1_fro_sqrt,Gamma):

        cov0_fro = np.linalg.norm(cov0, 'fro')
        cov0_fro_sqrt = np.sqrt(cov0_fro)

        mean_shift = linalg.norm(mean0-mean1,ord=2,axis=0)
        cov_shift = cov0_fro_sqrt - cov1_fro_sqrt
        condition = -mean_shift + Gamma * cov_shift

        if condition >= 0:
            alpha = 1
            logging.info("satify condition")
            return cov1_fro_sqrt, alpha
        else:
            alpha = 1-mean_shift/(Gamma * cov0_fro_sqrt)
            logging.info("covariance is adjusted")
            if alpha <= 0:
                alpha = 0
                logging.info("can't enforce the condition, mean shift is too big.")
            return alpha * cov0_fro_sqrt, alpha

    def load_refT(self, offline_idx, Tsh, x_init):
        filename = "refT"
        filepath = os.path.join("out/data/referenceTrajectory/T", filename)
        try:
            with open(filepath,"rb") as f:
                ref = pickle.load(f)
        except Exception as e:
            logging.info(f"Error while saving: {str(e)}")
        
        diff = np.array(ref).T[0:2] - np.array([x_init[0]*np.ones((1,len(ref))),x_init[1]*np.ones((1,len(ref)))]).reshape(2,len(ref))
        norm2 = linalg.norm(diff.T, ord=2, axis=1)
        # index_min = np.argmin(norm2)
        # the closest ahead point
        sorted_index = np.argsort(norm2)
        if sorted_index[0] < sorted_index[2]:
            index_min = sorted_index[0]
        elif sorted_index[0] > sorted_index[2]:
            index_min = sorted_index[1]

        self.refT = ref[index_min:index_min+ Tsh]

    def load_ovehicles(self, offline_idx):
        filename = "agent90_frame"+str(460+10*(offline_idx)) + "_cov"
        # filename = str(offline_idx)
        logging.info(filename)
        filepath = os.path.join("out/data/T_ov2_straight", filename)
        try:
            with open(filepath,"rb") as f:
                loaded_data = pickle.load(f)
                loaded_ovehicles = loaded_data["ovehicles"]
                params = loaded_data["params"]
                # __params = loaded_data["__params"]

        except Exception as e:
            logging.info(f"Error while saving: {str(e)}")

        return loaded_ovehicles , params,# __params


    def do_highlevel_control(self, params, ovehicles, Tsh, shrinking):
        timeLimit = 120
        apply_robust = True
        load_ov = False
        # initialize
        self.__halfspace = None
        # logging.info("HERE")
        self.load_refT(int(self.offline_index/10) + 1, Tsh, params.x_init)
        x_init_save=params.x_init
        # logging.info(f"refT:{self.refT}")
        if params.O > 0:
            if load_ov:
                # params = []
                # ovehicles = []
            
                # ovehicles, params, self.__params= self.load_ovehicles(int(self.offline_index/10-1))
                ovehicles, params= self.load_ovehicles(int(self.offline_index/10-1))
                params.x_init = x_init_save
                # ovehicles, _, _= self.load_ovehicles(int(self.offline_index/10))
                # ovehicles= self.load_ovehicles(int(self.offline_index/10-1))
                logging.info("load ovs")

        """Load the saved datas"""
        """
        filename = "agent86_frame1198_cov"
        path = "out/cov/try"
        filepath = os.path.join(path,filename)
        with open(filepath,'rb') as f:
            loaded_data = pickle.load(f)
            params = loaded_data["params"] 
            ovehicles = loaded_data["ovehicles"]
        """
        """Compute road segments and goal."""
        segments, goal = self.compute_segs_polytopes_and_goal(params, Tsh)

        # save the system matrix at the initial step       
        if Tsh == self.__prediction_horizon:
            self.__matrix_dynamics_full = (
                params.x_bar.copy(),    # shape (nx*T,)
                params.u_bar.copy(),    # shape (nu*T,)
                params.Gamma.copy(),    # shape (nx*T, nu*T)
            )

        """Apply motion planning problem""" 
        nx, nu = params.nx, params.nu
        T = Tsh
        T_prev = self.__prediction_horizon - Tsh   # steps already executed
        logging.info(f"T:{T}, Shrinking:{shrinking}")
        max_a, min_a = self.__params.max_a, self.__params.min_a
        max_delta = self.__params.max_delta

        """Applying the same Gamma in the shrinking horion to Guarantee the same dynamics"""
        # store the full matrix

        if Tsh < self.__prediction_horizon:
            x_bar_full, u_bar_full, Gamma_full = self.__matrix_dynamics_full

            row_off = nx * T_prev          # how many *state* rows to drop
            col_off = nu * T_prev          # how many *control* cols to drop

            x_bar  = x_bar_full[row_off:]          # shape (nx*T ,)
            u_bar  = u_bar_full[col_off:]          # shape (nu*T ,)
            Gamma  = Gamma_full[row_off:, :]       # keep all columns for now
        else:
            x_bar, u_bar, Gamma = params.x_bar, params.u_bar, params.Gamma

        """Model, control and state variables"""
        min_u = np.vstack((np.full(T, min_a), np.full(T, -max_delta))).T.ravel()
        max_u = np.vstack((np.full(T, max_a), np.full(T, max_delta))).T.ravel()
        # Slack variables for control
        u = cp.Variable(shape=(nu * T), name="u")
        constraints = [u >= min_u, u <= max_u]
        u_delta = u - u_bar[:nu*T]

        """Applying the same Gamma in the shrinking horion to Guarantee the same dynamics"""
        if T_prev:
            Gamma_future = Gamma[: nx*T, col_off : col_off + nu*T] 
            x = util.obj_matmul(Gamma_future, u_delta) + x_bar[: nx*T]

            # add past 
            Gamma_past = Gamma[: nx*T, : col_off] 
            u_prev_vec = np.concatenate(self.__U_prev).flatten()
            x += util.obj_matmul(Gamma_past, u_prev_vec)
        else:
            x = util.obj_matmul(Gamma[:nx*T,:nu*T], u_delta) + x_bar[:nx*T] # original
        
        X = x.reshape(T, nx)
        U = cp.reshape(u, (T, nu))

        """DEBUG: relaxation variable"""
        # assume params.O = 1
        # relax = cp.Variable(shape=(int(T * params.K)), name="relax")
        # Relax = cp.reshape(relax, ( int(params.K), T))

        """Apply state constraints"""
        state_constraints = self.compute_state_constraints(params, X)
        constraints.extend(state_constraints)

        """Apply road boundary constraints"""
        if self.road_boundary_constraints:
            I = len(segments.polytopes)
            # Slack variables from road obstacles
            Omicron = cp.Variable((I * T), boolean=True, name="omicron")
            Omicron = cp.reshape(Omicron, (I, T))
            constraints.extend(
                # self.compute_road_boundary_constraints(params, self.refT , Omicron, segments, T)
                self.compute_road_boundary_constraints(params, X , Omicron, segments, T)
            )
        else:
            Omicron = None

        if params.O > 0:
            L, K = self.__params.L, params.K
            eps = 0.05
            maxK = int(max(K))  # Max number of modes among all OVs
            eps_ura = np.zeros((params.O, maxK))

            for i in range(params.O):
                for k in range(int(K[i])):
                    eps_ura[i, k] = eps / (params.O)
            
            ## >>>added by HYEONTAE
            Delta2 = {(i, j, t): cp.Variable((L), boolean=True, name="delta2") for i in range(params.O) for j in range(len(eps_ura[0])) for t in range(T)}
            
            # time horizon 0~7, T = 8
            temp_x = {t: cp.vstack([X[t,0], X[t,1], 1]) for t in range(T)}
            
            if shrinking and apply_robust == True:

                logging.info(f"Shrinking horizon, T = {T}, robustification = {apply_robust}")
                (
                    GMM_constraints,
                    vertices,
                    A_union,
                    b_union,
                    OVconstraint,
                    direct,
                    ovStateMean_tau_1,
                    ovStateCov_tau_1,
                    meanNtangent
                ) = self.compute_obstacle_constraints_GMM_Minkowski_idealprediction(
                    params, ovehicles, Delta2, Omicron, temp_x, eps_ura, segments, T, self.refT #, Relax
                )

                # if len(ovehicles) == 1: # this is for T ov2 scenario
                #     GMM_constraints = [False]
                
            else:
                if shrinking:
                    logging.info(f"Shrinking horizon, T = {T}, robustification = {apply_robust}")
                else:
                    logging.info(f"Receding horizon, T = {T}")
                (
                    GMM_constraints,
                    vertices,
                    A_union,
                    b_union,
                    OVconstraint,
                    direct,
                    ovStateMean_tau_1,
                    ovStateCov_tau_1,
                    _
                ) = self.compute_obstacle_constraints_GMM_affine(
                    params, ovehicles, Delta2, Omicron, temp_x, eps_ura, segments, T, self.refT
                )
                # if shrinking == False: # this is for T ov2 scenario
                #     GMM_constraints = []
                # if shrinking == True:
                #     if len(ovehicles) == 1:
                #         GMM_constraints = [False]
            # GMM_constraints = []
                
            saveData = True # save data indicator
            data_save = {}
            data_save["direct"] = direct
            # logging.info(f"{data_save['direct']}")
            data_save["MeanCov"] = True
            data_save["OVconstraint"] = OVconstraint
            data_save["ovStateMean_tau_1"] = ovStateMean_tau_1
            data_save["ovStateCov_tau_1"] = ovStateCov_tau_1
            data_save["shrinking"] = shrinking
            if shrinking and apply_robust == True:
                # logging.info("Save meanNtangent")
                data_save["meanNtangent"] = meanNtangent
                if False: # save lower bound?? implemented only in Minkow
                    data_save["lowerBound"] = self.__prob_lower_save
            # data_save["ovehicles"] = ovehicles
            constraints.extend(GMM_constraints)
        else:
            data_save = {}
            data_save["MeanCov"] = False
            saveData = False # save data indicator
            vertices = A_union = b_union = None
        if load_ov:
            data_save["ovehicles"] = ovehicles
        data_save["params"] = params
        data_save["__params"] = self.__params
        #data_save["goal"] = goal
        #data_save["segments"] = segments
        """Compute and minimize objective"""
        # cost = self.compute_objective(X, U, goal,T)
        cost = self.compute_objective_referenceTraj(X, U, self.refT, goal,T)
        # cost = self.compute_objective_referenceTraj_Relax(X, U, self.refT, goal,T, Relax)
        logging.info(f"x_init:{params.x_init}")
        # for ov_idx, ovehicle in enumerate(ovehicles):
            # logging.info(f"cur location of ov:{ovehicle.past[-1]}")

        objective = cp.Minimize(cost)
        prob = cp.Problem(objective, constraints)
        # save halfspace for plot
        if self.plot_scenario:
            self.__plot_simulation_data.halfspace[params.frame] = self.__halfspace
            # logging.info(f"Save halfspace:{self.__halfspace}")
        # data_save = {}
        try:
            if self.get_computeTime:
                start = time.time()
                if self.log_cplex:
                    # prob.solve(solver=cp.CPLEX, cplex_params={"timelimit": timeLimit}, verbose=True)
                    prob.solve(solver=cp.CPLEX) #, cplex_params=n{"timelimit": timeLimit}, verbose=True)

                else:
                    prob.solve(solver=cp.CPLEX, cplex_params={"timelimit": timeLimit}, verbose=True)

                process_time = time.time() - start
                solve_time = prob.solver_stats.solve_time
                data_save['solve_time'] = solve_time
                data_save['process_time'] = process_time
            else:
                if self.log_cplex:
                    prob.solve(solver=cp.CPLEX, verbose=True)
                else:
                    prob.solve(solver=cp.CPLEX, verbose=False)
            data_save["x_init"] = params.x_init
        except cp.error.SolverError as e:
            logging.warning(f"CPLEX failed (will continue): {e}")


        """Extract solution"""
        try:
            cost = cost.value
            f = lambda x: x if isinstance(x, numbers.Number) else x.value
            U_star = util.obj_vectorize(f, U.value)
            X_star = util.obj_vectorize(f, X)
            # Relax_Star = util.obj_vectorize(f, Relax.value)
            # logging.info(f"Relax:{Relax_Star}")
            data_save['cost'] = cost
            data_save["goal"] = goal
            data_save["U_star"] = U_star
            data_save["X_star"] = X_star
            data_save["timeout"] = False
            data_save["infeasible"] = False
            # data_save["x_init"] = params.x_init

            if solve_time >= timeLimit:
                if saveData:
                    data_save["timeout"] = True
                    self.save_data(data_save, params, self.__ego_vehicle.id) 
                print("Solver reached the time limit!")
                return (
                    util.AttrDict(
                        cost=cost,
                        U_star=U_star,
                        X_star=X_star,
                        goal=goal,
                        A_union=A_union,
                        b_union=b_union,
                        vertices=vertices,
                        segments=segments,
                        halfspace = self.__halfspace
                    ),
                    True, # Indicator of timeout
                    None,
                )
            else:
                if saveData:
                    self.save_data(data_save, params, self.__ego_vehicle.id) 
                print("Solution found in", solve_time, "seconds.")
                return (
                    util.AttrDict(
                        cost=cost,
                        U_star=U_star,
                        X_star=X_star,
                        goal=goal,
                        A_union=A_union,
                        b_union=b_union,
                        vertices=vertices,
                        segments=segments,
                        halfspace = self.__halfspace
                    ),
                    False, # Indicator of timeout
                    None,
                )
        except:
            if saveData:
                data_save["timeout"] = False
                data_save["infeasible"] = True
                self.save_data(data_save, params, self.__ego_vehicle.id)
            # Handle CVXPY-specific errors here
            return util.AttrDict(
                cost=None, U_star=None, X_star=None,
                goal=goal, A_unions=A_union, b_unions=b_union,
                vertices=vertices, segments=segments, halfspace = self.__halfspace
            ), False, InSimulationException("Optimizer failed to find a solution")

    def __plot_scenario(self, pred_result, ovehicles, params, ctrl_result, error=None):
        lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
        ego_bbox = np.array([lon, lat])
        params.update(self.__params)
        if error:
            filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_oa_fail"
            PlotPredictiveControl(
                pred_result,
                ovehicles,
                params,
                ctrl_result,
                self.__control_horizon,
                ego_bbox,
            ).plot_oa_failure(filename=filename)
        else:
            filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_oa_predict"
            PlotPredictiveControl(
                pred_result,
                ovehicles,
                params,
                ctrl_result,
                self.__control_horizon,
                ego_bbox,
            ).plot_oa_prediction(filename=filename)

    def __plot_scenario_GMM(self, pred_result, ovehicles, params, ctrl_result, error=None):
        lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
        ego_bbox = np.array([lon, lat])
        params.update(self.__params)
        if error:
            filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_oa_fail"
            PlotPredictiveControl(
                pred_result,
                ovehicles,
                params,
                ctrl_result,
                self.__control_horizon,
                ego_bbox,
            ).plot_oa_failure(filename=filename)
        else:
            filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_oa_predict"
            PlotPredictiveControl(
                pred_result,
                ovehicles,
                params,
                ctrl_result,
                self.__control_horizon,
                ego_bbox,
            ).plot_oa_prediction(filename=filename)

    # @profile(sort_by='cumulative', lines_to_print=50, strip_dirs=True)
    def __compute_prediction_controls(self, frame, Tsh, shrinking):
        pred_result = self.do_prediction(frame)
        ovehicles = self.make_ovehicles(pred_result)
        params = self.make_local_params(frame, ovehicles,Tsh)
        ctrl_result, timeout, error = self.do_highlevel_control(params, ovehicles, Tsh, shrinking)
        
        # To plot scenario, save the planning horizon length
        self.__control_horizon = Tsh
        if self.plot_scenario:
            """Plot scenario"""
            self.__plot_scenario(
                pred_result, ovehicles, params, ctrl_result, error=error
            )
        if error:
            raise error
        
        if self.plot_scenario_GMM:
            """Plot scenario"""
            self.__plot_scenario_GMM(
                pred_result, ovehicles, params, ctrl_result, error=error
            )
        if error:
            raise error

        """use control input next round for warm starting."""
        self.__U_warmstarting = ctrl_result.U_star
        # logging.info(f"saved u star:{ctrl_result.U_star}")
        self.__U_prev.append(ctrl_result.U_star.T.ravel()[:params.nu])

        # logging.info(f"saved U star:{ctrl_result.U_star[0]}")
        self.__X_warmstarting = ctrl_result.X_star
        # logging.info(f"saved X star:{ctrl_result.X_star}")

        if self.plot_simulation:
            """Save planned trajectory for final plotting"""
            X = np.concatenate((params.initial_state.world[None], ctrl_result.X_star))
            # logging.info(f"planned traj:{X}")
            self.__plot_simulation_data.planned_trajectories[frame] = X
            self.__plot_simulation_data.planned_controls[frame] = ctrl_result.U_star
            # self.__plot_simulation_data.halfspace[frame] = ctrl_result.halfspace
            # logging.info(f"Save halfspace:{ctrl_result.halfspace}")
        "Save planned trajectory for check the distance from goal"
        #X = np.concatenate((params.initial_state.world[None], ctrl_result.X_star))
        #self.planned_finalstate= X[-1]
        """Get trajectory and velocity"""
        angles = util.npu.reflect_radians_about_x_axis(ctrl_result.X_star[:, 2])
        speeds = ctrl_result.X_star[:, 3]
        return speeds, angles, timeout

    def do_first_step(self, frame):
        self.__first_frame = frame
        self.__scene_builder = self.__scene_builder_cls(
            self,
            self.__map_reader,
            self.__ego_vehicle,
            self.__other_vehicles,
            self.__lidar_feeds,
            "test",
            self.__first_frame,
            scene_config=self.__scene_config,
            debug=False,
        )

    def run_step(self, frame, offline_index = 0 ,Tsh=6, shrinking = False, control=None):
        """Run motion planner step. Should be called whenever carla.World.click() is called.
        Parameters
        ==========
        frame : int
            Current frame of the simulation.
        control: carla.VehicleControl (optional)
            Optional control to apply to the motion planner. Used to move the vehicle
            while burning frames in the simulator before doing motion planning.
        """
        self.offline_index = offline_index

        #logging.debug(f"In LCSSHighLevelAgent.run_step() with frame = {frame}")
        if self.__first_frame is None:
            self.do_first_step(frame)

        self.__scene_builder.capture_trajectory(frame)
        timeout = False
        if (frame - self.__first_frame) % self.__scene_config.record_interval == 0:
            """We only motion plan every `record_interval` frames
            (e.g. every 0.5 seconds of simulation)."""
            frame_id = int(
                (frame - self.__first_frame) / self.__scene_config.record_interval
            )
            if frame_id < self.__n_burn_interval:
                """Initially collect data without doing any control to the vehicle."""
                pass
            elif (frame_id - self.__n_burn_interval) % self.__step_horizon == 0:
                speeds, angles, timeout = self.__compute_prediction_controls(frame, Tsh, shrinking)
                self.__local_planner.set_plan(
                    speeds, angles, self.__scene_config.record_interval
                )
            if self.plot_simulation:
                """Save actual trajectory for final plotting"""
                payload = carlautil.actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(
                    self.__ego_vehicle, flip_y=True
                )
                payload = np.array(
                    [
                        payload[0],
                        payload[1],
                        payload[13],
                        self.get_current_velocity(),
                    ]
                )
                try:
                    self.__plot_simulation_data.actual_trajectory[frame] = self.x_init
                except:
                    self.__plot_simulation_data.actual_trajectory[frame] = payload
                self.__plot_simulation_data.goals[frame] = self.get_goal()

        if not control:
            control = self.__local_planner.step()
        self.__ego_vehicle.apply_control(control)
        if self.plot_simulation:
            payload = self.__local_planner.get_current()
            self.__plot_simulation_data.lowlevel[frame] = payload
        
        return timeout

    def remove_scene_builder(self, first_frame):
        raise Exception(
            f"Can't remove scene builder from {util.classname(first_frame)}."
        )

    @staticmethod
    def parse_image(weak_self, image):
        """Pass sensor image to each scene builder.
        Parameters
        ==========
        image : carla.SemanticLidarMeasurement
        """
        self = weak_self()
        if not self:
            return
        # logging.debug(
        #     f"in DataCollector.parse_image() player = {self.__ego_vehicle.id} frame = {image.frame}"
        # )
        self.__lidar_feeds[image.frame] = image
        if self.__scene_builder:
            self.__scene_builder.capture_lidar(image)
            
