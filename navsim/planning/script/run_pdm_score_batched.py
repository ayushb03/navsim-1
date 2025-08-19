"""
GPU-optimized batched version of PDM score evaluation.
This script processes multiple scenes simultaneously for significant speedup.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from hydra._internal.utils import _locate
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from nuplan.common.actor_state.state_representation import StateSE2

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.batched_runner import BatchProcessor
from navsim.common.dataclasses import AgentInput
from navsim.common.dataloader import SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.caching import MetricCacheProcessor
from nuplan.common.geometry.convert import relative_to_absolute_poses
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def run_pdm_score_batched(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    Run PDM score evaluation with GPU batching for improved throughput.
    
    :param args: Arguments for running PDM score evaluation
    :return: List of DataFrames with evaluation results
    """
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    results = []
    
    for exp_idx, exp_arg in enumerate(args):
        exp_list = exp_arg["exp_list"]
        cfg = exp_arg["cfg"]
        
        logger.info(f"Running experiment {exp_idx + 1}/{len(args)} with {len(exp_list)} configurations")
        
        for exp_dir in exp_list:
            logger.info(f"Processing experiment directory: {exp_dir}")
            
            # Initialize components
            scenario_filter = _locate(cfg.scenario_filter._target_)(cfg.scenario_filter)
            scene_loader = SceneLoader(cfg.common_cfg, scenario_filter)
            metric_cache_loader = MetricCacheProcessor(
                metric_cache_base_path=cfg.metric_cache_base_path,
                metric_cache_map_location=cfg.metric_cache_map_location,
            )
            
            # Load agent
            agent_checkpoint_path = os.path.join(exp_dir, "checkpoints", "epoch=*.ckpt")
            agent_lightning_module = AgentLightningModule.load_from_checkpoint(agent_checkpoint_path)
            agent: AbstractAgent = agent_lightning_module.agent
            agent.initialize()
            agent = agent.to(device)
            agent.eval()
            
            # Get traffic agents policy and simulator
            traffic_agents_policy_stage_one = _locate(cfg.traffic_agents_policy_stage_one._target_)(
                cfg.traffic_agents_policy_stage_one
            )
            simulator = _locate(cfg.simulator._target_)(cfg.simulator)
            scorer = _locate(cfg.scorer._target_)(cfg.scorer)
            
            # Load tokens and filter
            scene_loader_tokens_stage_one = scene_loader.scene_dict.keys()
            metric_cache_loader.initialize()
            tokens_to_evaluate_stage_one = list(
                set(scene_loader_tokens_stage_one) & set(metric_cache_loader.tokens)
            )
            
            logger.info(f"Evaluating {len(tokens_to_evaluate_stage_one)} scenes")
            
            # Prepare batched data
            agent_inputs = []
            metric_caches = []
            valid_tokens = []
            
            for token in tokens_to_evaluate_stage_one:
                try:
                    metric_cache = metric_cache_loader.get_from_token(token)
                    agent_input = scene_loader.get_agent_input_from_token(token)
                    
                    agent_inputs.append(agent_input)
                    metric_caches.append(metric_cache)
                    valid_tokens.append(token)
                except Exception as e:
                    logger.warning(f"Failed to load data for token {token}: {e}")
                    continue
            
            if not agent_inputs:
                logger.warning("No valid scenes to process")
                continue
            
            # Batch processing setup
            batch_size = cfg.get("batch_size", 8)
            if torch.cuda.is_available():
                # Tune batch size based on GPU memory
                try:
                    runner = BatchProcessor(agent, device=device, batch_size=batch_size)
                    optimal_batch_sizes = runner.profile_batch_sizes(
                        agent_inputs[:min(32, len(agent_inputs))],
                        batch_sizes=[1, 2, 4, 8, 16, 32]
                    )
                    if optimal_batch_sizes:
                        # Choose batch size with highest throughput
                        optimal_batch_size = max(optimal_batch_sizes.keys(), 
                                               key=lambda k: optimal_batch_sizes[k])
                        logger.info(f"Optimal batch size: {optimal_batch_size} "
                                  f"({optimal_batch_sizes[optimal_batch_size]:.1f} scenes/sec)")
                        batch_size = optimal_batch_size
                except Exception as e:
                    logger.warning(f"Failed to profile batch sizes: {e}")
            
            # Benchmark removed to keep minimal implementation
            
            # Run batched inference
            logger.info(f"Running batched inference with batch_size={batch_size}")
            runner = BatchProcessor(agent, device=device, batch_size=batch_size)
            
            start_time = time.time()
            trajectories = runner.run_batched(agent_inputs)
            inference_time = time.time() - start_time
            
            throughput = len(agent_inputs) / inference_time
            logger.info(f"Processed {len(agent_inputs)} scenes in {inference_time:.2f}s "
                       f"({throughput:.1f} scenes/sec)")
            
            # Process results
            score_rows = []
            for idx, (trajectory, metric_cache, token) in enumerate(zip(trajectories, metric_caches, valid_tokens)):
                try:
                    score_row_stage_one, ego_simulated_states = pdm_score(
                        metric_cache=metric_cache,
                        model_trajectory=trajectory,
                        future_sampling=simulator.proposal_sampling,
                        simulator=simulator,
                        scorer=scorer,
                        traffic_agents_policy=traffic_agents_policy_stage_one,
                    )
                    
                    score_row_stage_one["valid"] = True
                    score_row_stage_one["log_name"] = metric_cache.log_name
                    score_row_stage_one["frame_type"] = metric_cache.scene_type
                    score_row_stage_one["start_time"] = metric_cache.timepoint.time_s
                    
                    end_pose = StateSE2(
                        x=trajectory.poses[-1, 0],
                        y=trajectory.poses[-1, 1],
                        heading=trajectory.poses[-1, 2],
                    )
                    absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
                    score_row_stage_one["endpoint_x"] = absolute_endpoint.x
                    score_row_stage_one["endpoint_y"] = absolute_endpoint.y
                    score_row_stage_one["start_point_x"] = metric_cache.ego_state.rear_axle.x
                    score_row_stage_one["start_point_y"] = metric_cache.ego_state.rear_axle.y
                    score_row_stage_one["ego_simulated_states"] = [ego_simulated_states]
                    
                    score_rows.append(score_row_stage_one)
                    
                except Exception as e:
                    logger.error(f"Failed to process results for scene {idx}: {e}")
                    # Create empty/invalid score row
                    score_row_stage_one = {"valid": False, "token": token}
                    score_rows.append(score_row_stage_one)
            
            # Convert to DataFrame
            df_results = pd.DataFrame(score_rows)
            results.append(df_results)
            
            logger.info(f"Completed experiment directory: {exp_dir}")
    
    return results


def run_pdm_score_batched_simple(
    agent: AbstractAgent,
    agent_inputs: List[AgentInput],
    metric_caches: List,
    simulator,
    scorer,
    traffic_agents_policy,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Simplified batched PDM score evaluation for direct use.
    
    :param agent: Agent to evaluate
    :param agent_inputs: List of AgentInput objects
    :param metric_caches: List of metric cache objects
    :param simulator: Simulator object
    :param scorer: Scorer object  
    :param traffic_agents_policy: Traffic agents policy
    :param batch_size: Batch size for processing
    :param device: Device to run on
    :return: DataFrame with evaluation results
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = agent.to(device)
    
    # Run batched inference
    runner = BatchProcessor(agent, device=device, batch_size=batch_size)
    trajectories = runner.run_batched(agent_inputs)
    
    # Process results
    score_rows = []
    for trajectory, metric_cache in zip(trajectories, metric_caches):
        try:
            score_row, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            score_row["valid"] = True
            score_rows.append(score_row)
        except Exception as e:
            logger.error(f"Failed to score trajectory: {e}")
            score_rows.append({"valid": False})
    
    return pd.DataFrame(score_rows)


if __name__ == "__main__":
    from navsim.planning.script.utils import set_default_path
    set_default_path()
    
    # Register configuration
    cs = ConfigStore.instance()
    cs.repo = CONFIG_PATH
    
    # Example usage - you would typically call this from a hydra app
    logger.info("Use this module as a library or integrate with hydra configuration")
