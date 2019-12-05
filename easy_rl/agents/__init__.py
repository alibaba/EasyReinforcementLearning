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

from easy_rl.agents.agent_base import AgentBase
from easy_rl.agents.agent import Agent
from easy_rl.agents.actor_learner_agent import ActorLearnerAgent
from easy_rl.agents.async_agent import AsyncAgent
from easy_rl.agents.apex_agent import ApexAgent
from easy_rl.agents.impala_agent import ImpalaAgent
from easy_rl.agents.a3c_agent import A3CAgent
from easy_rl.agents.sync_agent import SyncAgent
from easy_rl.agents.dppo_agent import DPPOAgent
from easy_rl.agents.es_agent import ESAgent

agents = dict(
    Agent=Agent,
    Apex=ApexAgent,
    Impala=ImpalaAgent,
    A3C=A3CAgent,
    DPPO=DPPOAgent,
    ES=ESAgent)

__all__ = [
    "AgentBase", "Agent", "ActorLearnerAgent", "AsyncAgent", "ApexAgent",
    "ImpalaAgent", "A3CAgent", "SyncAgent", "DPPOAgent", "ESAgent"
]
