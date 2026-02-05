from server_side.server_comm_api import DQNCommServer
from server_side.dqn_manager import DQN_AGENT_AB

S, H, N_BUFFER = 19, 7, 36000
dqn_model = DQN_AGENT_AB(S, H, [7], N_BUFFER, params=None)
dqn_model.load_model("/home/nxc/wjbang/server_side/checkpoint_util_80")