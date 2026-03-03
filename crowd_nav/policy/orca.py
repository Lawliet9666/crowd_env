import numpy as np
import rvo2
from crowd_nav.policy.policy import Policy
from crowd_sim.env.robot.action import ActionXY


class ORCA(Policy):
    def __init__(self, config):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__(config)
        cfg = config.get("orca", config) if isinstance(config, dict) else getattr(config, "orca", {})
        if not isinstance(cfg, dict):
            cfg = {}
        self.name = 'ORCA'
        self.max_neighbors = None
        self.radius = None
        self.max_speed = 1 # the ego agent assumes that all other agents have this max speed
        self.sim = None
        self.neighbor_dist = float(cfg.get("neighbor_dist", 5.0))
        self.time_horizon = float(cfg.get("time_horizon", 0.5))
        self.time_horizon_obst = float(cfg.get("time_horizon_obst", 1.0))
        self.safety_space = float(cfg.get("safety_space", 0.1))


    def predict(self, state):
        return self.predict_from_states(state.self_state, state.human_states)

    def predict_from_states(self, self_state, human_states):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param self_state:
        :param human_states:
        :return:
        """
        # max number of humans = current number of humans
        self.max_neighbors = len(human_states)
        self.radius = self_state.radius
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            self.sim.addAgent((self_state.px, self_state.py), *params, self_state.radius + 0.01 + self.safety_space,
                              self_state.v_pref, (self_state.vx, self_state.vy))
            for human_state in human_states:
                max_speed = getattr(human_state, "v_pref", self.max_speed)
                self.sim.addAgent((human_state.px, human_state.py), *params, human_state.radius + 0.01 + self.safety_space,
                                  max_speed, (human_state.vx, human_state.vy))
        else:
            self.sim.setAgentPosition(0, (self_state.px, self_state.py))
            self.sim.setAgentVelocity(0, (self_state.vx, self_state.vy))
            for i, human_state in enumerate(human_states):
                self.sim.setAgentPosition(i + 1, (human_state.px, human_state.py))
                self.sim.setAgentVelocity(i + 1, (human_state.vx, human_state.vy))

        # Set preferred velocity toward goal, scaled by v_pref (max preferred speed).
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        if speed > 1e-6:
            pref_speed = min(speed, self_state.v_pref)
            pref_vel = velocity / speed * pref_speed
        else:
            pref_vel = velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = (self_state, human_states)

        return action
