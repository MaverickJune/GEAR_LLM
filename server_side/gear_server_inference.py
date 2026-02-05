from server_side.server_comm_api import DQNCommServer
from server_side.dqn_manager import DQN_AGENT_AB

def inference_gear_dqn_server():
    S, H, N_BUFFER = 19, 7, 36000
    
    # Set path to pre-trained model based on the desired target CPU utilization level
    target_util_model_path = "/home/nxc/wjbang/server_side/checkpoint_util_80"
    
    # Load Pre-trained DQN model
    dqn_model = DQN_AGENT_AB(S, H, [7], N_BUFFER, params=None)
    dqn_model.load_model(target_util_model_path)
    dqn_model.policy_net.eval()
    
    server = DQNCommServer(
        host="0.0.0.0",
        port=61103,
        dqn_model=dqn_model,
        max_connections=10,
        is_train=False,
        set_inference_mode_action=True
    )
    
    # Set callbacks
    def on_state_received(state, client_id):
        print(f"[Callback] State received from {client_id}: {state[:3]}... (showing first 3)")
    
    server.set_on_state_received_callback(on_state_received)
    
    # Start server (blocking)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()
        
if __name__ == "__main__":
    print("Starting DQN Communication Server for Gear Inference...")
    print("Press Ctrl+C to stop\n")
    inference_gear_dqn_server()