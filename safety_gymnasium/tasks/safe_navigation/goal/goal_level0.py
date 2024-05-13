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
"""Goal level 0."""

from safety_gymnasium.assets.geoms import GoalRed, GoalBlue, GoalGreen, GoalPurple
from safety_gymnasium.bases.base_task import BaseTask


class GoalLevel0(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-2, -2, 2, 2]

        self._add_geoms(
            GoalRed(keepout=0.305),
            GoalBlue(keepout=0.305),
            GoalGreen(keepout=0.305),
            GoalPurple(keepout=0.305),
        )
        self.last_dist_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal_red()
        reward += (self.last_dist_goal_red - dist_goal) * self.goal_red.reward_distance
        self.last_dist_goal_red = dist_goal

        if self.goal_achieved:
            reward += self.goal_red.reward_goal

        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        # self.build_goal_position()
        self.last_dist_goal_red = self.dist_goal_red()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal_red() <= self.goal_red.size
    
    @property
    def each_goal_achieved(self):
        # pylint: disable-next=no-member
        achieved = {}
        achieved['r'] = self.dist_goal_red() <= self.goal_red.size
        achieved['g'] = self.dist_goal_green() <= self.goal_green.size
        achieved['b'] = self.dist_goal_blue() <= self.goal_blue.size
        achieved['p'] = self.dist_goal_purple() <= self.goal_purple.size
        return achieved


    def dist_goal_red(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal_red'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(self.goal_red.pos)  # pylint: disable=no-member

    def dist_goal_blue(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal_blue'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(self.goal_blue.pos)  # pylint: disable=no-member

    def dist_goal_green(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal_green'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(self.goal_green.pos)  # pylint: disable=no-member

    def dist_goal_purple(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal_purple'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(self.goal_purple.pos)  # pylint: disable=no-member
