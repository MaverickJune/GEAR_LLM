"""
Communication API for Jetson client to interact with server-side DQN
Sends state samples from OrinNaiveStateMonitor and receives actions
"""

import socket
import json
import struct
import numpy as np
from typing import List, Dict, Any, Optional
import time


class DQNCommClient:
    """
    Client for communicating with server-side DQN
    Sends state information and receives action decisions
    """
    
    def __init__(self, server_host: str = "localhost", server_port: int = 5000, 
                 timeout: float = 10.0):
        """
        Initialize DQN communication client
        
        Args:
            server_host: Server hostname or IP address
            server_port: Server port number
            timeout: Socket timeout in seconds
        """
        self.server_host = server_host
        self.server_port = server_port
        self.timeout = timeout
        self.socket = None
        self.is_connected = False
        
    def connect(self) -> bool:
        """
        Establish connection to DQN server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.server_host, self.server_port))
            self.is_connected = True
            print(f"Connected to DQN server at {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to DQN server: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Close connection to DQN server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            self.is_connected = False
            print("Disconnected from DQN server")
    
    def _send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send JSON message to server
        
        Args:
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
            self.socket.sendall(msg_length)
            
            # Send actual message
            self.socket.sendall(json_bytes)
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False
    
    def _receive_message(self) -> Optional[Dict[str, Any]]:
        """
        Receive JSON message from server
        
        Returns:
            Parsed JSON dictionary or None if failed
        """
        try:
            # Receive message length (4 bytes)
            length_data = self._recv_exact(4)
            if not length_data:
                return None
            
            msg_length = struct.unpack('>I', length_data)[0]
            
            # Receive actual message
            json_bytes = self._recv_exact(msg_length)
            if not json_bytes:
                return None
            
            # Decode and parse JSON
            json_data = json_bytes.decode('utf-8')
            message = json.loads(json_data)
            return message
        except Exception as e:
            print(f"Failed to receive message: {e}")
            return None
    
    def _recv_exact(self, num_bytes: int) -> Optional[bytes]:
        """
        Receive exact number of bytes from socket
        
        Args:
            num_bytes: Number of bytes to receive
            
        Returns:
            Received bytes or None if failed
        """
        data = b''
        while len(data) < num_bytes:
            try:
                chunk = self.socket.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                print("Socket receive timeout")
                return None
            except Exception as e:
                print(f"Socket receive error: {e}")
                return None
        return data
    
    def send_state_get_action(self, state: List[float], 
                             additional_info: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Send state to DQN server and receive action
        
        Args:
            state: State vector from OrinNaiveStateMonitor (19 dimensions)
            additional_info: Optional additional information to send
            
        Returns:
            Action index from DQN or None if failed
        """
        if not self.is_connected:
            print("Not connected to server. Call connect() first.")
            return None
        
        # Prepare message
        message = {
            'type': 'state',
            'timestamp': time.time(),
            'state': state,
        }
        
        if additional_info:
            message['additional_info'] = additional_info
        
        # Send state
        if not self._send_message(message):
            return None
        
        # Receive action
        response = self._receive_message()
        if response is None:
            return None
        
        # Validate response
        if response.get('type') != 'action':
            print(f"Unexpected response type: {response.get('type')}")
            return None
        
        action = response.get('action')
        if action is None:
            print("No action in response")
            return None
        
        return action
    
    def send_batch_states_get_actions(self, states: List[List[float]], 
                                      additional_info: Optional[Dict[str, Any]] = None) -> Optional[List[int]]:
        """
        Send batch of states to DQN server and receive batch of actions
        
        Args:
            states: List of state vectors
            additional_info: Optional additional information
            
        Returns:
            List of action indices or None if failed
        """
        if not self.is_connected:
            print("Not connected to server. Call connect() first.")
            return None
        
        # Prepare message
        message = {
            'type': 'batch_state',
            'timestamp': time.time(),
            'states': states,
            'batch_size': len(states)
        }
        
        if additional_info:
            message['additional_info'] = additional_info
        
        # Send states
        if not self._send_message(message):
            return None
        
        # Receive actions
        response = self._receive_message()
        if response is None:
            return None
        
        # Validate response
        if response.get('type') != 'batch_action':
            print(f"Unexpected response type: {response.get('type')}")
            return None
        
        actions = response.get('actions')
        if actions is None:
            print("No actions in response")
            return None
        
        return actions
    
    def send_reward(self, reward: float, done: bool = False, 
                   additional_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send reward feedback to DQN server (for online learning)
        
        Args:
            reward: Reward value
            done: Whether episode is done
            additional_info: Optional additional information
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            print("Not connected to server. Call connect() first.")
            return False
        
        # Prepare message
        message = {
            'type': 'reward',
            'timestamp': time.time(),
            'reward': reward,
            'done': done
        }
        
        if additional_info:
            message['additional_info'] = additional_info
        
        # Send reward
        return self._send_message(message)
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


class StateActionManager:
    """
    High-level manager for state monitoring and action requests
    Integrates OrinNaiveStateMonitor with DQNCommClient
    """
    
    def __init__(self, state_monitor, comm_client: DQNCommClient):
        """
        Initialize state-action manager
        
        Args:
            state_monitor: Instance of OrinNaiveStateMonitor
            comm_client: Instance of DQNCommClient
        """
        self.state_monitor = state_monitor
        self.comm_client = comm_client
        self.last_state = None
        self.last_action = None
        
    def get_state_and_request_action(self, additional_info: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Sample current state and request action from DQN server
        
        Args:
            additional_info: Optional additional information to send
            
        Returns:
            Tuple of (state, action) or (None, None) if failed
        """
        try:
            # Get current state from monitor
            state = self.state_monitor.get_state()
            self.last_state = state
            
            # Request action from DQN server
            action = self.comm_client.send_state_get_action(state, additional_info)
            
            if action is not None:
                self.last_action = action
                return state, action
            else:
                return None, None
                
        except Exception as e:
            print(f"Error in get_state_and_request_action: {e}")
            return None, None
    
    def send_reward_feedback(self, reward: float, done: bool = False) -> bool:
        """
        Send reward feedback based on last action
        
        Args:
            reward: Reward value
            done: Whether episode is done
            
        Returns:
            True if successful, False otherwise
        """
        return self.comm_client.send_reward(reward, done)


# Example usage function
def example_usage():
    """Example of how to use the communication API"""
    from GearLLM.jetson_utils.state_utils import OrinNaiveStateMonitor
    
    # Initialize state monitor
    print("Initializing state monitor...")
    state_monitor = OrinNaiveStateMonitor()
    
    # Initialize communication client
    print("Initializing DQN communication client...")
    comm_client = DQNCommClient(
        server_host="147.46.130.111",  # Replace with actual server IP
        server_port=61103,
        timeout=10.0
    )
    
    # Connect to server
    if not comm_client.connect():
        print("Failed to connect to server")
        return
    
    try:
        # Create state-action manager
        manager = StateActionManager(state_monitor, comm_client)
        
        # Main loop
        for i in range(10):
            print(f"\n--- Iteration {i+1} ---")
            
            # Get state and request action
            state, action = manager.get_state_and_request_action(
                additional_info={'iteration': i}
            )
            
            if state is None or action is None:
                print("Failed to get state or action")
                continue
            
            print(f"State (19-dim): {state}")
            print(f"Action received: {action}")
            
            # Execute action (user implementation)
            # ...
            
            # Calculate reward (user implementation)
            reward = 0.5  # Placeholder
            
            # Send reward feedback
            manager.send_reward_feedback(reward, done=False)
            
            time.sleep(1)  # Wait before next iteration
            
    finally:
        # Disconnect
        comm_client.disconnect()


if __name__ == "__main__":
    example_usage()
