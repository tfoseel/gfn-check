"""
Copied and Adapted from https://github.com/Tikquuss/GflowNets_Tutorial
"""

from typing import Literal, Tuple

import torch
from einops import rearrange

from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.gym.helpers.preprocessors import KHotPreprocessor, OneHotPreprocessor
from gfn.preprocessors import EnumPreprocessor, IdentityPreprocessor
from gfn.states import DiscreteStates, States
from torchtyping import TensorType as TT

class Tree(DiscreteEnv):
    def __init__(self, max_depth: int, max_value: int, preprocessor: Literal['khot', 'one hot', 'identity'] = 'identity'):
        pass

    def step(self, states: DiscreteStates, actions: Actions, action_idx: int) -> torch.Tensor:
        pass

    def backward_step(self, states: States, actions: Actions) -> torch.Tensor:
        pass
    
    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        pass


        

        