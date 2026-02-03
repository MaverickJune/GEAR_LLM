"""
Server-side communication API for DQN
Receives state information from Jetson clients and sends back actions
"""

import socket
import json
import struct
import threading
import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable
import queue


class DQNCommServer:
    """
    Server for receiving states from Jetson clients and sending actions
    Integrates with DQN model for action selection
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000, 
                 dqn_model=None, max_connections: int = 10):
        """
        Initialize DQN communication server
        
        Args:
            host: Server host address (0.0.0.0 for all interfaces)
            port: Server port number
            dqn_model: DQN model instance with select_action() method
            max_connections: Maximum number of concurrent client connections
        """
        self.host = host
        self.port = port
        self.dqn_model = dqn_model
        self.max_connections = max_connections
        
        self.server_socket = None
        self.is_running = False
        self.client_threads = []
        self.client_count = 0
        self.client_lock = threading.Lock()
        
        # Statistics
        self.total_requests = 0
        self.total_rewards = 0
        self.stats_lock = threading.Lock()
        
        # Callback functions
        self.on_state_received_callback = None
        self.on_reward_received_callback = None
        
    def set_dqn_model(self, dqn_model):
        """Set or update DQN model"""
        self.dqn_model = dqn_model
        print("DQN model updated")
    
    def set_on_state_received_callback(self, callback: Callable):
        """
        Set callback function to be called when state is received
        Callback signature: callback(state: List[float], client_id: str)
        """
        self.on_state_received_callback = callback
    
    def set_on_reward_received_callback(self, callback: Callable):
        """
        Set callback function to be called when reward is received
        Callback signature: callback(reward: float, done: bool, client_id: str)
        """
        self.on_reward_received_callback = callback
    
    def start(self):
        """Start the server and listen for connections"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_connections)
            self.is_running = True
            
            print(f"DQN Server started on {self.host}:{self.port}")
            print(f"Waiting for client connections...")
            
            while self.is_running:
                try:
                    self.server_socket.settimeout(1.0)
                    client_socket, client_address = self.server_socket.accept()
                    
                    # Create thread to handle client
                    with self.client_lock:
                        self.client_count += 1
                        client_id = f"client_{self.client_count}"
                    
                    print(f"New connection from {client_address} (ID: {client_id})")
                    
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address, client_id),
                        daemon=True
                    )
                    client_thread.start()
                    self.client_threads.append(client_thread)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_running:
                        print(f"Error accepting connection: {e}")
                    
        except Exception as e:
            print(f"Failed to start server: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server"""
        print("Stopping server...")
        self.is_running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Wait for client threads to finish
        for thread in self.client_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        print("Server stopped")
    
    def _handle_client(self, client_socket: socket.socket, client_address, client_id: str):
        """
        Handle communication with a single client
        
        Args:
            client_socket: Client socket connection
            client_address: Client address tuple
            client_id: Unique client identifier
        """
        print(f"[{client_id}] Handler started")
        
        try:
            while self.is_running:
                # Receive message from client
                message = self._receive_message(client_socket)
                
                if message is None:
                    print(f"[{client_id}] Connection closed")
                    break
                
                # Process message based on type
                msg_type = message.get('type')
                
                if msg_type == 'state':
                    self._handle_state_request(client_socket, message, client_id)
                    
                elif msg_type == 'batch_state':
                    self._handle_batch_state_request(client_socket, message, client_id)
                    
                elif msg_type == 'reward':
                    self._handle_reward_feedback(message, client_id)
                    
                else:
                    print(f"[{client_id}] Unknown message type: {msg_type}")
                    
        except Exception as e:
            print(f"[{client_id}] Error handling client: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
            print(f"[{client_id}] Handler terminated")
    
    def _handle_state_request(self, client_socket: socket.socket, message: Dict, client_id: str):
        """
        Handle single state request and send action
        
        Args:
            client_socket: Client socket
            message: Request message containing state
            client_id: Client identifier
        """
        state = message.get('state')
        if state is None:
            print(f"[{client_id}] No state in message")
            return
        
        # Update statistics
        with self.stats_lock:
            self.total_requests += 1
        
        # Call callback if set
        if self.on_state_received_callback:
            try:
                self.on_state_received_callback(state, client_id)
            except Exception as e:
                print(f"[{client_id}] Error in state callback: {e}")
        
        # Get action from DQN model
        action = self._select_action(state, client_id)
        
        # Send response
        response = {
            'type': 'action',
            'action': action,
            'timestamp': time.time()
        }
        
        self._send_message(client_socket, response)
    
    def _handle_batch_state_request(self, client_socket: socket.socket, message: Dict, client_id: str):
        """
        Handle batch state request and send batch actions
        
        Args:
            client_socket: Client socket
            message: Request message containing batch of states
            client_id: Client identifier
        """
        states = message.get('states')
        if states is None:
            print(f"[{client_id}] No states in message")
            return
        
        batch_size = len(states)
        
        # Update statistics
        with self.stats_lock:
            self.total_requests += batch_size
        
        # Get actions for all states
        actions = []
        for state in states:
            action = self._select_action(state, client_id)
            actions.append(action)
        
        # Send response
        response = {
            'type': 'batch_action',
            'actions': actions,
            'batch_size': batch_size,
            'timestamp': time.time()
        }
        
        self._send_message(client_socket, response)
    
    def _handle_reward_feedback(self, message: Dict, client_id: str):
        """
        Handle reward feedback from client
        
        Args:
            message: Reward message
            client_id: Client identifier
        """
        reward = message.get('reward')
        done = message.get('done', False)
        
        if reward is None:
            print(f"[{client_id}] No reward in message")
            return
        
        # Update statistics
        with self.stats_lock:
            self.total_rewards += 1
        
        print(f"[{client_id}] Reward received: {reward:.4f}, Done: {done}")
        
        # Call callback if set
        if self.on_reward_received_callback:
            try:
                self.on_reward_received_callback(reward, done, client_id)
            except Exception as e:
                print(f"[{client_id}] Error in reward callback: {e}")
    
    def _select_action(self, state: List[float], client_id: str) -> int:
        """
        Select action using DQN model
        
        Args:
            state: State vector
            client_id: Client identifier
            
        Returns:
            Selected action index
        """
        if self.dqn_model is None:
            # Return random action if no model is set
            action = np.random.randint(0, 10)  # Assuming 10 possible actions
            print(f"[{client_id}] No DQN model, returning random action: {action}")
            return int(action)
        
        try:
            # Call DQN model's select_action method
            # Assumes model has a method: select_action(state) -> int
            if hasattr(self.dqn_model, 'select_action'):
                action = self.dqn_model.select_action(state)
            elif hasattr(self.dqn_model, 'predict'):
                action = self.dqn_model.predict(state)
            elif callable(self.dqn_model):
                action = self.dqn_model(state)
            else:
                raise AttributeError("DQN model must have select_action, predict method or be callable")
            
            return int(action)
            
        except Exception as e:
            print(f"[{client_id}] Error selecting action: {e}")
            # Return default action on error
            return 0
    
    def _send_message(self, client_socket: socket.socket, message: Dict[str, Any]) -> bool:
        """
        Send JSON message to client
        
        Args:
            client_socket: Client socket
            message: Dictionary to send as JSON
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert message to JSON and encode
            json_data = json.dumps(message)
            json_bytes = json_data.encode('utf-8')
            
            # Send message length first (4 bytes)
            msg_length = struct.pack('>I', len(json_bytes))
            client_socket.sendall(msg_length)
            
            # Send actual message
            client_socket.sendall(json_bytes)
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False
    
    def _receive_message(self, client_socket: socket.socket) -> Optional[Dict[str, Any]]:
        """
        Receive JSON message from client
        
        Args:
            client_socket: Client socket
            
        Returns:
            Parsed JSON dictionary or None if failed
        """
        try:
            # Receive message length (4 bytes)
            length_data = self._recv_exact(client_socket, 4)
            if not length_data:
                return None
            
            msg_length = struct.unpack('>I', length_data)[0]
            
            # Receive actual message
            json_bytes = self._recv_exact(client_socket, msg_length)
            if not json_bytes:
                return None
            
            # Decode and parse JSON
            json_data = json_bytes.decode('utf-8')
            message = json.loads(json_data)
            return message
        except Exception as e:
            # Connection likely closed
            return None
    
    def _recv_exact(self, client_socket: socket.socket, num_bytes: int) -> Optional[bytes]:
        """
        Receive exact number of bytes from socket
        
        Args:
            client_socket: Client socket
            num_bytes: Number of bytes to receive
            
        Returns:
            Received bytes or None if failed
        """
        data = b''
        while len(data) < num_bytes:
            try:
                chunk = client_socket.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data += chunk
            except Exception as e:
                return None
        return data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics"""
        with self.stats_lock:
            return {
                'total_requests': self.total_requests,
                'total_rewards': self.total_rewards,
                'active_clients': sum(1 for t in self.client_threads if t.is_alive())
            }
    
    def print_statistics(self):
        """Print server statistics"""
        stats = self.get_statistics()
        print("\n" + "=" * 50)
        print("SERVER STATISTICS")
        print("=" * 50)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Total Rewards: {stats['total_rewards']}")
        print(f"Active Clients: {stats['active_clients']}")
        print("=" * 50 + "\n")


class MockDQNModel:
    """
    Mock DQN model for testing
    Replace with actual DQN implementation
    """
    
    def __init__(self, state_dim: int = 19, num_actions: int = 10):
        """
        Initialize mock DQN model
        
        Args:
            state_dim: State vector dimension
            num_actions: Number of possible actions
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        print(f"MockDQNModel initialized: state_dim={state_dim}, num_actions={num_actions}")
    
    def select_action(self, state: List[float]) -> int:
        """
        Select action based on state
        
        Args:
            state: State vector
            
        Returns:
            Selected action index
        """
        # Simple rule-based mock policy
        # In real implementation, this would use neural network
        
        # Example: select action based on CPU utilization (first 8 dimensions)
        if len(state) >= 8:
            avg_cpu_util = np.mean(state[:8])
            
            # High utilization -> conservative action
            if avg_cpu_util > 0.8:
                action = 0  # Conservative action
            elif avg_cpu_util > 0.5:
                action = self.num_actions // 2  # Moderate action
            else:
                action = self.num_actions - 1  # Aggressive action
        else:
            action = np.random.randint(0, self.num_actions)
        
        return int(action)


# Example usage
def example_server():
    """Example of how to run the DQN server"""
    
    # Create mock DQN model
    dqn_model = MockDQNModel(state_dim=19, num_actions=10)
    
    # Create server
    server = DQNCommServer(
        host="0.0.0.0",
        port=5000,
        dqn_model=dqn_model,
        max_connections=10
    )
    
    # Set callbacks
    def on_state_received(state, client_id):
        print(f"[Callback] State received from {client_id}: {state[:3]}... (showing first 3)")
    
    def on_reward_received(reward, done, client_id):
        print(f"[Callback] Reward from {client_id}: {reward:.4f}, Done: {done}")
    
    server.set_on_state_received_callback(on_state_received)
    server.set_on_reward_received_callback(on_reward_received)
    
    # Statistics printer thread
    def print_stats_periodically():
        while server.is_running:
            time.sleep(10)
            server.print_statistics()
    
    stats_thread = threading.Thread(target=print_stats_periodically, daemon=True)
    stats_thread.start()
    
    # Start server (blocking)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()


if __name__ == "__main__":
    print("Starting DQN Communication Server...")
    print("Press Ctrl+C to stop\n")
    example_server()
