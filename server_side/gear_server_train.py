from server_side.server_comm_api import DQNCommServer
from server_side.dqn_manager import DQN_AGENT_AB

def train_gear_dqn_server():
    S, H, N_BUFFER = 19, 7, 36000
    dqn_model = DQN_AGENT_AB(S, H, [7], N_BUFFER, params=None)
    
    server = DQNCommServer(
        host="0.0.0.0",
        port=61103,
        dqn_model=dqn_model,
        max_connections=10,
        is_train=True
    )
    
    # Set callbacks
    def on_state_received(state, client_id):
        print(f"[Callback] State received from {client_id}: {state[:3]}... (showing first 3)")
    
    def on_reward_received(reward, done, client_id):
        print(f"[Callback] Reward from {client_id}: {reward:.4f}, Done: {done}")
    
    server.set_on_state_received_callback(on_state_received)
    server.set_on_reward_received_callback(on_reward_received)
    
    # Start server (blocking)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()
        
if __name__ == "__main__":
    print("Starting DQN Communication Server for Gear Training...")
    print("Press Ctrl+C to stop\n")
    train_gear_dqn_server()