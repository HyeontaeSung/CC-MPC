import os
import time
import math
import logging

import numpy as np
import pandas as pd

import carla
import utility as util
import carlautil

try:
    # imports from trajectron-plus-plus/trajectron
    from environment import Environment, Scene
    from model import Trajectron
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from tests import (
    LoopEnum,
    ScenarioParameters,
    CtrlParameters,
    attach_camera_to_spectator,
    shift_spawn_point
)
from collect.generate import get_all_vehicle_blueprints
from collect.generate import NaiveMapQuerier
from collect.in_simulation.midlevel.v8 import MidlevelAgent
from collect.generate.scene import OnlineConfig
from collect.generate.scene.v3_2.trajectron_scene import TrajectronPlusPlusSceneBuilder
from collect.exception import InSimulationException

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

class MonteCarloScenario(object):

    DEBUG_SETTINGS = util.AttrDict(
        plot_boundary=False,
        log_agent=False,
        log_cplex=False,
        plot_scenario=False,
        plot_simulation=False,
        plot_overapprox=False,
    )

    TOL = 6
    TOL_turningShrinking = 10 # T = 6 -> 10 in OV close, 6 in OV far
    def __init__(
        self,
        scenario_params: ScenarioParameters,
        ctrl_params: CtrlParameters,
        carla_synchronous: tuple,
        eval_env: Environment,
        eval_stg: Trajectron,
        motion_planner_cls: MidlevelAgent,
        scene_builder_cls: TrajectronPlusPlusSceneBuilder,
        n_simulations = 10
    ):
        self.client, self.world, self.carla_map, self.traffic_manager = carla_synchronous
        self.scenario_params = scenario_params
        self.ctrl_params = ctrl_params
        self.eval_env = eval_env
        self.eval_stg = eval_stg
        self.motion_planner_cls = motion_planner_cls
        self.scene_builder_cls = scene_builder_cls
        self.n_simulations = 1
        # self.n_simulations = 1
        # Shrinking-horizon setting, manually selected Window
        self.combineShrinking = True
        self.shTstart = 6
        self.shTend = 14
        self.shrinking_manually = False # we select the time window of shrinking horizon
        self.shrinking_dist = True              # start shrinking hoizon when |X_star(-1)-goal|^2 < C 

        self.map_reader = NaiveMapQuerier(self.world, self.carla_map, debug=True)
        self.online_config = OnlineConfig(record_interval=10, node_type=self.eval_env.NodeType)
        
        # Mock vehicles
        self.spawn_points = self.carla_map.get_spawn_points()
        self.blueprints = get_all_vehicle_blueprints(self.world)
        self.blueprint_audi_a2 = self.world.get_blueprint_library().find('vehicle.audi.a2')
        

    def episode(self, episode_idx):

        logging.info(f"doing episode {episode_idx}")
        ego_vehicle = None
        agent = None
        other_vehicles = []
        stats = util.AttrDict(
            success=False,
            infeasibility=False,
            steps=0,
            plan_steps=0,
            timeOver = False,
            initiallyFeasible = False,
        )

        try:
            # settings = self.world.get_settings()
            # settings.synchronous_mode = True
            # settings.fixed_delta_seconds = 0.05
            # self.world.apply_settings(settings)
            
            # fix ov trajectory
            # 1) Wolrd sync + fixed delta
            SEED = 41 # go straight in T
            # SEED = 2 # turn right in T
            settings = self.world.get_settings()
            settings.synchronus_mode = True
            # settings.fixed_delta_seconds = 0.05
            # self.world.apply_settings(settings)
            
            # 2) Seed Python & Numpy
            # random.seed(SEED)
            # np.random.seed(SEED)

            # 3) TM deterministic mode
            # self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_random_device_seed(SEED)

            shrinking = True
            # if it starts shrinking horizon when the scene begins, you have to modify it manually 
            # in midlevel/
            shrinkIndex = (self.ctrl_params.control_horizon + 1) * 10 - 1
            
            spawn_indices = [self.scenario_params.ego_spawn_idx] + self.scenario_params.other_spawn_ids
            other_vehicle_ids = []
            for k, spawn_idx in enumerate(spawn_indices):
                if k == 0:
                    blueprint = self.blueprint_audi_a2
                else:
                    # blueprint = np.random.choice(self.blueprints)
                    # blueprint = self.blueprints[7]
                    # logging.info(f"len blue prints:{len(self.blueprints)}") : 21
                    blueprint = self.world.get_blueprint_library().find('vehicle.nissan.micra')
                spawn_point = self.spawn_points[spawn_idx]

                if k >= 1 and episode_idx % 10 == 0 and episode_idx != 0:
                    self.scenario_params.spawn_shifts[k] = \
                        self.scenario_params.spawn_shifts[k] + 1*(2/10)
                # if k == 1:
                    # logging.info(f"spawn_shifts:{self.scenario_params.spawn_shifts[k]}")


                spawn_point = shift_spawn_point(
                    self.carla_map, k, self.scenario_params.spawn_shifts, spawn_point
                )
                # Prevent collision with road.
                spawn_point.location += carla.Location(0, 0, 0.5)
                vehicle = self.world.spawn_actor(blueprint, spawn_point)
                if k == 0:
                    ego_vehicle = vehicle
                else:
                    vehicle.set_autopilot(True, self.traffic_manager.get_port())
                    # init_settings = self.world.get_settings()#
                    # settings = self.world.get_settings()#
                    # settings.synchronous_mode = True#
                    if self.scenario_params.ignore_signs:
                        self.traffic_manager.ignore_signs_percentage(vehicle, 100.)
                    if self.scenario_params.ignore_lights:
                        self.traffic_manager.ignore_lights_percentage(vehicle, 100.)
                    if self.scenario_params.ignore_vehicles:
                        self.traffic_manager.ignore_vehicles_percentage(vehicle, 100.)
                    if not self.scenario_params.auto_lane_change:
                        self.traffic_manager.auto_lane_change(vehicle, False)
                        # 50% faster than speed limit
                    # self.traffic_manager.vehicle_percentage_speed_difference(vehicle, - 29 % 10) 
                    # self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 1 ) 
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, - (episode_idx % 10)) 
                    # logging.info(f"set speed:{- episode_idx % 10}")
                    # import pdb
                    # pdb.set_trace()
                    # logging.info(f"set speed:{ episode_idx % 10}")
                    # self.world.apply_settings(init_settings)
                    
                    other_vehicles.append(vehicle)
                    other_vehicle_ids.append(vehicle.id)
            
            frame = self.world.tick()
            agent = self.motion_planner_cls(
                ego_vehicle,
                self.map_reader,
                other_vehicle_ids,
                self.eval_stg,
                scene_builder_cls=self.scene_builder_cls,
                scene_config=self.online_config,
                **self.scenario_params,
                **self.ctrl_params,
                **self.DEBUG_SETTINGS,
            )
            agent.start_sensor()
            assert agent.sensor_is_listening
            if self.scenario_params.goal:
                agent.set_goal(**self.scenario_params.goal)

            """Setup vehicle routes"""
            if "CARLANAME" in os.environ and os.environ["CARLANAME"] == "carla-0.9.13":
                for k, vehicle in enumerate(other_vehicles):
                    route = None
                    try:
                        route = self.scenario_params.other_routes[k]
                        len(route)
                    except (TypeError, IndexError) as e:
                        continue
                    self.traffic_manager.set_route(vehicle, route)
            else:
                logging.info("Skipping setting up OV routes.")
            
            if episode_idx == 0:
                """Move the spectator to the ego vehicle.
                The positioning is a little off"""
                goal = agent.get_goal()
                goal.x = 167.174698
                goal.y = -81.759842
                #goal.x = 190
                #goal.y = -80
                goal_x, goal_y = goal.x, -goal.y
                state = agent.get_vehicle_state()
                if goal.is_relative:
                    location = carla.Location(
                            x=state[0] + goal_x /2.,
                            y=state[1] - goal_y /2.,
                            z=state[2] + 80)
                else:
                    location = carla.Location(
                            x=(state[0] + goal_x) /2.,
                            y=(state[1] + goal_y) /2.,
                            z=state[2] + 80)
                # rotation = carla.Rotation(pitch=-70, yaw=-90, roll=20)
                rotation = carla.Rotation(pitch=-90, yaw=0, roll=0)
                # configure the spectator
                self.world.get_spectator().set_transform(
                    carla.Transform(location, rotation)
                )
                location = carla.Location(goal_x, goal_y, state[2])
                carlautil.debug_point(
                    self.world, location, t=60.0, label="goal"
                )

            n_burn_frames = self.scenario_params.n_burn_interval*self.online_config.record_interval
            if self.ctrl_params.loop_type == LoopEnum.CLOSED_LOOP:
                run_frames = self.scenario_params.run_interval*self.online_config.record_interval
            else:
                run_frames = self.ctrl_params.control_horizon*self.online_config.record_interval - 1
            for idx in range(n_burn_frames):
                control = None
                for ctrl in self.scenario_params.controls:
                    if ctrl.interval[0] <= idx and idx <= ctrl.interval[1]:
                        control = ctrl.control
                        break
                agent.run_step(frame, control=control)
                frame = self.world.tick()

            # Whether using robustification and shrinking-horizon
            T = self.ctrl_params.control_horizon
            OnceShrink = False
            offline_index = 0
            for idx in range(run_frames):
                if self.combineShrinking:
                    if self.shrinking_manually:
                        if self.shTstart*10 < idx <= self.shTend*10:
                            # max(1, 6 - (idx - 51) // 10) if idx in range(51, 111) else None
                            T = max(1, self.ctrl_params.control_horizon - (idx - (self.shTstart*10+1)) // 10) if idx in range(self.shTstart*10+1, self.shTend*10+1) else None
                            stats.timeOver = agent.run_step(frame, T)
                            frame = self.world.tick()
                            stats.steps += 1
                            state = agent.get_vehicle_state(flip_y=True)
                            goal = agent.get_goal()
                            #goal.x = 167.174698
                            #goal.y = -81.759842
                        
                            dist = math.sqrt((state[0] - goal.x)**2 + (state[1] - goal.y)**2)
                            if stats.timeOver == True:
                                break
                            if dist < self.TOL:
                                stats.success = True
                                break
                        else:
                            stats.timeOver = agent.run_step(frame, self.ctrl_params.control_horizon)
                            frame = self.world.tick()
                            stats.steps += 1
                            state = agent.get_vehicle_state(flip_y=True)
                            goal = agent.get_goal()
                            #goal.x = 167.174698
                            #goal.y = -81.759842
                            dist = math.sqrt((state[0] - goal.x)**2 + (state[1] - goal.y)**2)
                    
                            if stats.timeOver == True:
                                break
                            if dist < self.TOL:
                                stats.success = True
                                break
                                
                    elif self.shrinking_dist:
                        if shrinking == False: # receding horizon
                            stats.timeOver = agent.run_step(frame,offline_index, T,shrinking)
                            offline_index += 1
                            frame = self.world.tick()
                            stats.steps += 1
                            state = agent.get_vehicle_state(flip_y=True)
                            goal = agent.get_goal()
                            goal.x = 167.174698
                            goal.y = -81.759842

                            #finalState = agent.get_final_state()

                            #dist_final = math.sqrt((finalState[0] - goal.x)**2 + (finalState[1] - goal.y)**2)

                            dist = math.sqrt((state[0] - goal.x)**2 + (state[1] - goal.y)**2)
                            # logging.info(f"dist:{dist}")
                            if stats.timeOver == True:
                                break
                            if dist < self.TOL:
                                stats.success = True
                                break
                            '''
                            if OnceShrink == False: # we do SH only once
                                if dist_final < self.TOL_turningShrinking:
                                    stats.initiallyFeasible = True
                                    shrinking = True
                            '''
                            if OnceShrink == False: # we do SH only once
                                if dist < 36: # it was 32
                                    # stats.initiallyFeasible = True
                                    shrinking = True

                        else: # shrinking horizon
                            T = max(1,shrinkIndex//10)

                            if T <= self.ctrl_params.control_horizon -1 :
                                stats.initiallyFeasible = True
                            stats.timeOver = agent.run_step(frame,offline_index,T,shrinking)
                            offline_index += 1
                            frame = self.world.tick()
                            stats.steps += 1
                            state = agent.get_vehicle_state(flip_y=True)
                            goal = agent.get_goal()
                            goal.x = 167.174698
                            goal.y = -81.759842
                            dist = math.sqrt((state[0] - goal.x)**2 + (state[1] - goal.y)**2)
                            # logging.info(f"dist:{dist}")
                            if stats.timeOver == True:
                                break
                            if dist < self.TOL:
                                stats.success = True
                                break
                            # initially feasible at the first step of SH
                  
                            #logging.info("initially?")
                            
                            shrinkIndex = shrinkIndex - 1
                            if shrinkIndex//10 < 1:
                                T = self.ctrl_params.control_horizon
                                OnceShrink = True
                                shrinking = False # return to a receding-horizon
                                shrinkIndex = self.ctrl_params.control_horizon * 10 - 1


                                
                else:
                    T = 6
                    stats.timeOver = agent.run_step(frame, T)
                    frame = self.world.tick()
                    stats.steps += 1
                    state = agent.get_vehicle_state(flip_y=True)
                    goal = agent.get_goal()
                    #goal.x = 167.174698
                    #goal.y = -81.759842
                    dist = math.sqrt((state[0] - goal.x)**2 + (state[1] - goal.y)**2)
                    if stats.timeOver == True:
                        break
                    if dist < self.TOL:
                        stats.initiallyFeasible = True
                        stats.success = True
                        break

            if stats.timeOver == True:
                logging.info("Time is too long, start again. in optimizer TimeOut")
                if agent:
                    agent.destroy()
                if ego_vehicle:
                    ego_vehicle.destroy()
                for other_vehicle in other_vehicles:
                    other_vehicle.destroy()

        except InSimulationException as e:
            stats.infeasibility = True

        finally:
            if stats.timeOver != True:
                if agent:
                    agent.destroy()
                if ego_vehicle:
                    ego_vehicle.destroy()
                for other_vehicle in other_vehicles:
                    other_vehicle.destroy()
            stats.plan_steps = stats.steps / self.online_config.record_interval
            time.sleep(1)
        
        logging.info(
            f"episode succeeded? {stats.success}. "
            f"Infeasibility? {stats.infeasibility}. "
            f"Ran planner over {stats.steps} total steps; "
            f"{stats.plan_steps} planning steps."
        )
        return pd.Series(stats)

    def run(self):
        logging.info(f"Running {self.n_simulations} episodes.")
        stats = []
        episode_idx = 0
        while len(stats) <= self.n_simulations-1:
            stat = self.episode(episode_idx)
            if stat.timeOver == True:
                continue
            elif True: #stat.initiallyFeasible == True: # countin feasibility only when initially feasible
                
                stats.append(stat)
                episode_idx = episode_idx + 1
            try:
                stats_cur = pd.DataFrame(stats)
                frac_initially= stats_cur["initiallyFeasible"].mean()
                frac_success = stats_cur["success"].mean()
                frac_infeasi = stats_cur["infeasibility"].mean()
                mean_steps   = stats_cur[stats_cur["success"]]["steps"].mean()
                mean_plan    = stats_cur[stats_cur["success"]]["plan_steps"].mean()
                logging.info(f"frac of initially feasible: {frac_initially}")
                logging.info(f"frac of success: {frac_success}")
                logging.info(f"frac of recursive feasibility: {(1-frac_infeasi)/frac_initially}")
                logging.info(f"success mean steps: {mean_steps}")
                logging.info(f"success mean planning steps: {mean_plan}")
            except:
                continue

        stats = pd.DataFrame(stats)
        frac_success = stats["success"].mean()
        frac_infeasi = stats["infeasibility"].mean()
        mean_steps   = stats[stats["success"]]["steps"].mean()
        mean_plan    = stats[stats["success"]]["plan_steps"].mean()
        logging.info(f"frac of success: {frac_success}")
        logging.info(f"frac of infeasibility: {frac_infeasi}")
        logging.info(f"success mean steps: {mean_steps}")
        logging.info(f"success mean planning steps: {mean_plan}")


class PlannerScenario(object):

    DEBUG_SETTINGS = util.AttrDict(
        plot_boundary=False,
        log_agent=False,
        log_cplex=False,
        plot_scenario=True,
        plot_simulation=True,
        # plot_scenario=False,
        # plot_simulation=False,
        plot_overapprox=False,
    )

    def __init__(
        self,
        scenario_params: ScenarioParameters,
        ctrl_params: CtrlParameters,
        carla_synchronous: tuple,
        eval_env: Environment,
        eval_stg: Trajectron,
        motion_planner_cls: MidlevelAgent,
        scene_builder_cls: TrajectronPlusPlusSceneBuilder
    ):
        self.client, self.world, self.carla_map, self.traffic_manager = carla_synchronous
        self.scenario_params = scenario_params
        self.ctrl_params = ctrl_params
        self.eval_env = eval_env
        self.eval_stg = eval_stg
        self.motion_planner_cls = motion_planner_cls
        self.scene_builder_cls = scene_builder_cls

    def run(self):
        ego_vehicle = None
        agent = None
        spectator_camera = None
        other_vehicles = []
        record_spectator = False

        try:
            map_reader = NaiveMapQuerier(self.world, self.carla_map, debug=True)
            online_config = OnlineConfig(record_interval=10, node_type=self.eval_env.NodeType)

            # Mock vehicles
            spawn_points = self.carla_map.get_spawn_points()
            blueprints = get_all_vehicle_blueprints(self.world)
            spawn_indices = [self.scenario_params.ego_spawn_idx] + self.scenario_params.other_spawn_ids
            other_vehicle_ids = []
            for k, spawn_idx in enumerate(spawn_indices):
                if k == 0:
                    blueprint = self.world.get_blueprint_library().find('vehicle.audi.a2')
                else:
                    blueprint = np.random.choice(blueprints)
                spawn_point = spawn_points[spawn_idx]
                spawn_point = shift_spawn_point(
                    self.carla_map, k, self.scenario_params.spawn_shifts, spawn_point
                )
                # Prevent collision with road.
                spawn_point.location += carla.Location(0, 0, 0.5)
                vehicle = self.world.spawn_actor(blueprint, spawn_point)
                if k == 0:
                    ego_vehicle = vehicle
                else:
                    vehicle.set_autopilot(True, self.traffic_manager.get_port())
                    if self.scenario_params.ignore_signs:
                        self.traffic_manager.ignore_signs_percentage(vehicle, 100.)
                    if self.scenario_params.ignore_lights:
                        self.traffic_manager.ignore_lights_percentage(vehicle, 100.)
                    if self.scenario_params.ignore_vehicles:
                        self.traffic_manager.ignore_vehicles_percentage(vehicle, 100.)
                    if not self.scenario_params.auto_lane_change:
                        self.traffic_manager.auto_lane_change(vehicle, False)
                    other_vehicles.append(vehicle)
                    other_vehicle_ids.append(vehicle.id)

            frame = self.world.tick()
            agent = self.motion_planner_cls(
                ego_vehicle,
                map_reader,
                other_vehicle_ids,
                self.eval_stg,
                scene_builder_cls=self.scene_builder_cls,
                scene_config=online_config,
                **self.scenario_params,
                **self.ctrl_params,
                **self.DEBUG_SETTINGS,
            )
            agent.start_sensor()
            assert agent.sensor_is_listening

            """Setup vehicle routes"""
            if "CARLANAME" in os.environ and os.environ["CARLANAME"] == "carla-0.9.13":
                for k, vehicle in enumerate(other_vehicles):
                    route = None
                    try:
                        route = self.scenario_params.other_routes[k]
                        len(route)
                    except (TypeError, IndexError) as e:
                        continue
                    self.traffic_manager.set_route(vehicle, route)
            else:
                logging.info("Skipping setting up OV routes.")
            
            """Move the spectator to the ego vehicle.
            The positioning is a little off"""
            state = agent.get_vehicle_state()
            goal = agent.get_goal()
            goal_x, goal_y = goal.x, -goal.y
            if goal.is_relative:
                location = carla.Location(
                        x=state[0] + goal_x /2.,
                        y=state[1] - goal_y /2.,
                        z=state[2] + 50)
            else:
                location = carla.Location(
                        x=(state[0] + goal_x) /2.,
                        y=(state[1] + goal_y) /2.,
                        z=state[2] + 50)
            # configure the spectator
            self.world.get_spectator().set_transform(
                carla.Transform(
                    location, carla.Rotation(pitch=-70, yaw=-90, roll=20)
                )
            )
            record_spectator = False
            if record_spectator:
                # attach camera to spectator
                spectator_camera = attach_camera_to_spectator(self.world, frame)

            n_burn_frames = self.scenario_params.n_burn_interval*online_config.record_interval
            if self.ctrl_params.loop_type == LoopEnum.CLOSED_LOOP:
                run_frames = self.scenario_params.run_interval*online_config.record_interval
            else:
                run_frames = self.ctrl_params.control_horizon*online_config.record_interval - 1
            for idx in range(n_burn_frames + run_frames):
                control = None
                for ctrl in self.scenario_params.controls:
                    if ctrl.interval[0] <= idx and idx <= ctrl.interval[1]:
                        control = ctrl.control
                        break
                frame = self.world.tick()
                agent.run_step(frame, control=control)
    
        finally:
            if spectator_camera:
                spectator_camera.destroy()
            if agent:
                agent.destroy()
            if ego_vehicle:
                ego_vehicle.destroy()
            for other_vehicle in other_vehicles:
                other_vehicle.destroy()
        
            if record_spectator == True:
                time.sleep(5)
            else:
                time.sleep(1)
