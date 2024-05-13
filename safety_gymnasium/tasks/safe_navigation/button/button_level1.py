# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Button task 1."""

from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.mocaps import Gremlins
from safety_gymnasium.tasks.safe_navigation.button.button_level0 import ButtonLevel0
import numpy as np
from dataclasses import dataclass

import numpy as np

from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.assets.geoms import Hardwalls
class ButtonLevel1(ButtonLevel0):
    """An agent must press a goal button while avoiding hazards and gremlins.

    And while not pressing any of the wrong buttons.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # self.set_seed(10)
        # self.random_generator.set_random_seed(54)
        self.placements_conf.extents = [-2.0, -2.0, 2.0, 2.0]
        self._add_geoms(Hazards(num=8, keepout=0.05))
        self._add_mocaps(Gremlins(num=2, travel=0.2, keepout=0.4))
        self._add_geoms(Hardwalls(num=4, locate_factor=2.0, keepout=0.18))
        self.buttons.is_constrained = True  # pylint: disable=no-member
        self.original_button_locations = self.buttons.locations
        self.mechanism_conf.randomize_layout = False
        # self.mechanism_conf.continue_goal = False  # change this if we're using the goal button to help guide the agent
        self.lidar_conf.max_dist = 6  # large enough distance so all objects will be detected
        self.agent_min = -2.
        self.agent_max = 2.

    # @property
    # def goal_achieved(self):
    #     # override to prevent the environment from blocking observations
    #     return False
    
    def specific_reset(self):
        """Reset the buttons timer."""
        # import pdb; pdb.set_trace()
        # self.set_seed(10)
        self.buttons.timer = 0  # pylint: disable=no-member
        # move the agent back to a position within the map if it goes out of boundaries

        