"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
# import torch.nn.functional as F
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from network import FeedForwardNN

	
class CBFLayer(nn.Module):
	"""
		A differentiable Control Barrier Function (CBF) layer using cvxpylayers.
		Solves: 
			min ||u - u_nom||^2 + lambda * epsilon^2
			s.t. 
				h_dot + alpha * h >= -epsilon  (CBF constraint with slack)
				||u|| <= u_max                 (Velocity constraint)
				epsilon >= 0                   (Slack non-negativity)
	"""
	def __init__(self, action_dim=2, alpha=2.0, safe_dist=1.0, u_max=1.0, slack_penalty=10000.0):
		super(CBFLayer, self).__init__()
		self.alpha = alpha
		self.safe_dist = safe_dist
		self.u_max = u_max
		
		u = cp.Variable(action_dim)
		s = cp.Variable(1) # Slack variable
		
		u_nom = cp.Parameter(action_dim)
		
		# Parameters for CBF constraint: a^T u <= b + s
		# a = -2 * p_rel
		# b = alpha * h + 2 * p_rel * v_human
		a_cbf = cp.Parameter(action_dim) 
		b_cbf = cp.Parameter(1)
		
		obj = cp.Minimize(cp.sum_squares(u - u_nom) + slack_penalty * cp.sum_squares(s))
		
		# Constraints
		constraints = [
			a_cbf @ u <= b_cbf + s,  # CBF constraint with slack
			cp.sum_squares(u) <= u_max**2, # Velocity limit constraint
			s >= 0                   # Slack must be non-negative
		]
		
		prob = cp.Problem(obj, constraints)
		self.layer = CvxpyLayer(prob, parameters=[u_nom, a_cbf, b_cbf], variables=[u, s])

	def forward(self, u_nom, obs):
		# Handle input shapes (batch vs single)
		is_batch = obs.dim() > 1
		if not is_batch:
			obs = obs.unsqueeze(0)
			u_nom = u_nom.unsqueeze(0)

		# Extract relative position and velocity from observation
		# obs: [goal_x, goal_y, human_rel_x, human_rel_y, human_vx, human_vy]
		p_rel = obs[:, 2:4]   # p_human - p_robot
		v_human = obs[:, 4:6] # v_human

		# Barrier function h(x) = ||p_rel||^2 - R_safe^2
		dist_sq = torch.sum(p_rel**2, dim=1, keepdim=True)
		h = dist_sq - self.safe_dist**2

		# Constraint: 2 * p_rel * (u - v_human) >= -alpha * h
		# => 2 * p_rel * u >= 2 * p_rel * v_human - alpha * h
		# => -2 * p_rel * u <= alpha * h - 2 * p_rel * v_human
		# Let a_cbf * u <= b_cbf
		
		a_cbf = -2 * p_rel
		b_cbf = self.alpha * h - 2 * torch.sum(p_rel * v_human, dim=1, keepdim=True)
		
		# Solve QP
		# Note: CvxpyLayer expects batched inputs
		u_safe, s_val = self.layer(u_nom, a_cbf, b_cbf)

		if not is_batch:
			u_safe = u_safe.squeeze(0)
		
		return u_safe

class DiffCBF_NN(FeedForwardNN):
	"""
		A FeedForwardNN with a differentiable CBF layer at the end.
	"""
	def __init__(self, in_dim, out_dim, alpha=2.0, safe_dist=0.8, u_max=1.0, slack_penalty=1000.0):
		super(DiffCBF_NN, self).__init__(in_dim, out_dim)

		self.use_cbf = (out_dim > 1)
		if self.use_cbf:
			self.cbf_layer = CBFLayer(action_dim=out_dim, alpha=alpha, safe_dist=safe_dist, u_max=u_max, slack_penalty=slack_penalty)

	def forward(self, obs):
		u_nom = super().forward(obs)
		
		# Apply CBF correction if enabled
		if self.use_cbf:
			if isinstance(obs, np.ndarray):
				obs = torch.tensor(obs, dtype=torch.float)
			return self.cbf_layer(u_nom, obs)
		
		return u_nom




