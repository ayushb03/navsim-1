"""Batched processing for NavSim agents."""

import time
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, Trajectory


class CollateFunction:
    def __init__(self, agent: AbstractAgent):
        self.agent = agent
    def __call__(self, batch):
        return collate_navsim_features(batch, self.agent)


class NavSimDataset(Dataset):
    def __init__(self, agent_inputs: List[AgentInput], scenes: Optional[List[Scene]] = None):
        self.agent_inputs = agent_inputs
        self.scenes = scenes
    def __len__(self) -> int:
        return len(self.agent_inputs)
    def __getitem__(self, idx: int) -> Union[AgentInput, tuple[AgentInput, Scene]]:
        return (self.agent_inputs[idx], self.scenes[idx]) if self.scenes else self.agent_inputs[idx]


def collate_navsim_features(batch, agent: AbstractAgent) -> Dict[str, torch.Tensor]:
    """Build and batch features from AgentInputs with padding for variable shapes."""
    agent_inputs = [item[0] if isinstance(batch[0], tuple) else item for item in batch]
    
    # Build features per scene
    per_scene_features = []
    for agent_input in agent_inputs:
        scene_features = {}
        for builder in agent.get_feature_builders():
            features = builder.compute_features(agent_input)
            for key, value in features.items():
                scene_features[key] = torch.as_tensor(value) if not isinstance(value, torch.Tensor) else value
        per_scene_features.append(scene_features)
    
    if not per_scene_features:
        return {}
    
    batched_features = {}
    for key in sorted(per_scene_features[0].keys()):
        tensors = [scene_feats[key] for scene_feats in per_scene_features]
        shapes = [tuple(t.shape) for t in tensors]
        
        if all(shape == shapes[0] for shape in shapes):
            batched_features[key] = torch.stack(tensors, dim=0)
        else:
            # Pad variable shapes
            max_shape = [max(t.shape[i] for t in tensors) for i in range(tensors[0].dim())]
            padded_tensors, masks = [], []
            
            for tensor in tensors:
                pad_sizes = []
                for i in range(tensor.dim() - 1, -1, -1):
                    pad_sizes.extend([0, max_shape[i] - tensor.shape[i]])
                padded_tensors.append(torch.nn.functional.pad(tensor, pad_sizes))
                
                if tensor.dim() > 0:
                    mask = torch.zeros(max_shape[0], dtype=torch.bool)
                    mask[:tensor.shape[0]] = True
                    masks.append(mask)
                else:
                    masks.append(torch.tensor(True, dtype=torch.bool))
            
            batched_features[key] = torch.stack(padded_tensors, dim=0)
            if masks:
                batched_features[f"{key}_mask"] = torch.stack(masks, dim=0)
    
    return batched_features


class BatchProcessor:
    """GPU-optimized batch processor for NavSim agents."""
    
    def __init__(self, agent: AbstractAgent, device: Optional[torch.device] = None, batch_size: int = 8):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device
        self.agent = agent.to(self.device)
        self.batch_size = batch_size
    
    def _synchronize(self):
        """Synchronize GPU operations for accurate timing."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
    
    def run_batched(self, agent_inputs: List[AgentInput], scenes: Optional[List[Scene]] = None) -> List[Trajectory]:
        """Run batched inference on AgentInputs."""
        if scenes and len(scenes) != len(agent_inputs):
            raise ValueError("Number of scenes must match number of agent inputs")
        
        dataloader = DataLoader(
            NavSimDataset(agent_inputs, scenes),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=CollateFunction(self.agent),
        )
        
        trajectories = []
        self.agent.eval()
        
        with torch.inference_mode():
            for batch_features in dataloader:
                device_features = {k: v.to(self.device) for k, v in batch_features.items()}
                predictions = self.agent.forward(device_features)
                batch_trajectories = predictions["trajectory"].cpu().numpy()
                
                for i in range(batch_trajectories.shape[0]):
                    trajectories.append(Trajectory(batch_trajectories[i], self.agent._trajectory_sampling))
        
        return trajectories
    
    def profile_batch_sizes(self, agent_inputs: List[AgentInput], scenes: Optional[List[Scene]] = None, 
                          batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]) -> Dict[int, float]:
        """Profile different batch sizes to find optimal throughput."""
        results = {}
        original_batch_size = self.batch_size
        max_scenes = min(len(agent_inputs), 64)
        profile_inputs = agent_inputs[:max_scenes]
        profile_scenes = scenes[:max_scenes] if scenes else None
        
        for batch_size in batch_sizes:
            if batch_size > len(profile_inputs):
                continue
            self.batch_size = batch_size
            
            try:
                # Warmup
                for _ in range(3):
                    self.run_batched(profile_inputs[:batch_size], profile_scenes[:batch_size] if profile_scenes else None)
                
                # Measure
                self._synchronize()
                start_time = time.time()
                self.run_batched(profile_inputs, profile_scenes)
                self._synchronize()
                
                results[batch_size] = len(profile_inputs) / (time.time() - start_time)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                raise
        
        self.batch_size = original_batch_size
        return results