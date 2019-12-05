# Copyright (c) 2019 Alibaba Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from easy_rl.models.model_base import Model
from easy_rl.models.policy_gradient import PGModel
from easy_rl.models.ppo_model import PPOModel
from easy_rl.models.dqn import DQNModel
from easy_rl.models.v_trace import VTraceModel
from easy_rl.models.ddpg import DDPGModel
from easy_rl.models.evolution_strategy import EvolutionStrategy
from easy_rl.models.marwil import MarwilModel
from easy_rl.models.linucb import LinUCBModel

models = dict(
    PG=PGModel,
    PPO=PPOModel,
    DQN=DQNModel,
    DDPG=DDPGModel,
    Vtrace=VTraceModel,
    ES=EvolutionStrategy,
    Marwil=MarwilModel,
    LinUCB=LinUCBModel)

__all__ = [
    "Model", "PGModel", "PPOModel", "DQNModel", "DDPGModel", "VTraceModel",
    "EvolutionStrategy", "MarwilModel", "LinUCBModel"
]
