#!/usr/bin/env python3
"""NavSim GPU batched processing performance test."""

import time
import torch
import numpy as np
from typing import List, Dict

from navsim.agents.ego_status_mlp_agent import EgoStatusMLPAgent
from navsim.common.batched_runner import BatchProcessor
from navsim.common.dataclasses import AgentInput, EgoStatus, Camera, Cameras, Lidar
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def create_test_data(num_scenes: int) -> List[AgentInput]:
    """Create test data."""
    agent_inputs = []
    for i in range(num_scenes):
        ego_statuses = [EgoStatus(
            ego_pose=np.array([float(i), float(j), 0.0], dtype=np.float32),
            ego_velocity=np.array([5.0, 0.0], dtype=np.float32),
            ego_acceleration=np.array([0.1, 0.0], dtype=np.float32),
            driving_command=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ) for j in range(11)]
        
        cameras = [Cameras(
            cam_f0=Camera(), cam_l0=Camera(), cam_l1=Camera(), cam_l2=Camera(),
            cam_r0=Camera(), cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera(),
        ) for _ in range(11)]
        
        lidars = [Lidar() for _ in range(11)]
        agent_inputs.append(AgentInput(ego_statuses, cameras, lidars))
    
    return agent_inputs


def test_device(device: torch.device, num_scenes: int = 512) -> Dict[str, float]:
    """Test performance on a specific device."""
    try:
        agent = EgoStatusMLPAgent(hidden_layer_dim=64, lr=1e-3, 
                                 trajectory_sampling=TrajectorySampling(4, 0.5))
        agent = agent.to(device)
        agent_inputs = create_test_data(num_scenes)
        runner = BatchProcessor(agent, device, 32)
        
        # Warmup with better error handling for MPS
        agent.eval()
        try:
            for _ in range(2):
                agent.compute_trajectory(agent_inputs[0])
                runner.run_batched(agent_inputs[:32])
        except RuntimeError as e:
            if device.type == "mps" and ("Placeholder storage" in str(e) or "MPS" in str(e)):
                # MPS has compatibility issues with this specific model
                return {'speedup': 0.0, 'throughput': 0.0, 'error': 'MPS compatibility issue'}
            raise
        
        # Measure serial
        runner._synchronize()
        start = time.time()
        for agent_input in agent_inputs:
            agent.compute_trajectory(agent_input)
        runner._synchronize()
        serial_time = time.time() - start
        
        # Measure batched
        runner._synchronize()
        start = time.time()
        runner.run_batched(agent_inputs)
        runner._synchronize()
        batched_time = time.time() - start
        
        speedup = serial_time / batched_time
        throughput = num_scenes / batched_time
        
        return {'speedup': speedup, 'throughput': throughput, 'time': batched_time}
    
    except Exception as e:
        return {'error': str(e)}


def main():
    print("🚀 NavSim Batched Processing Performance")
    print("-" * 40)
    
    # Test available devices
    devices = []
    if torch.cuda.is_available():
        devices.append(("CUDA", torch.device("cuda")))
    if torch.backends.mps.is_available():
        devices.append(("MPS", torch.device("mps")))
    devices.append(("CPU", torch.device("cpu")))
    
    results = {}
    for name, device in devices:
        result = test_device(device)
        if result and 'error' not in result:
            results[name] = result
            print(f"{name:4s}: {result['speedup']:.1f}x speedup, {result['throughput']:6.0f} scenes/sec")
        elif result and 'error' in result:
            if 'compatibility' in result['error']:
                print(f"{name:4s}: Available but incompatible with current model")
            else:
                print(f"{name:4s}: Error - {result['error'][:30]}...")
        else:
            print(f"{name:4s}: Not available")
    
    if len(results) > 0:
        working_results = {k: v for k, v in results.items() if v['speedup'] > 0}
        if working_results:
            best = max(working_results.keys(), key=lambda k: working_results[k]['speedup'])
            print(f"\n🏆 Best: {best} ({working_results[best]['speedup']:.1f}x speedup)")


if __name__ == "__main__":
    main()
