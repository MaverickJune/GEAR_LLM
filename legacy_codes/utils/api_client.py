"""
API Client for communicating with remote Perf-Trainer server
Supports DQN-based DVFS control via HTTP requests
"""

import requests
import json
from typing import Dict, Any, Optional


class DVFSClient:
    """
    Client for interacting with remote Perf-Trainer server for DVFS control
    """
    
    def __init__(self, server_url: str = "http://192.168.137.1:5000", 
                 api_prefix: str = "/con", timeout: int = 30):
        """
        Initialize DVFS client
        
        Args:
            server_url: Base URL of the Perf-Trainer server
            api_prefix: API prefix ("/con" for context API or "/nn" for DQN API)
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.api_prefix = api_prefix
        self.timeout = timeout
        self.model_id = None
        
    def init_model(self, model_type: str = "dqn_nx", params: Optional[Dict] = None) -> bool:
        """
        Initialize a new model on the server
        
        Args:
            model_type: Type of model to initialize (e.g., "dqn_nx", "dqn_pro_nx")
            params: Model initialization parameters
            
        Returns:
            bool: True if initialization successful
        """
        url = f"{self.server_url}{self.api_prefix}/init_model"
        
        if params is None:
            params = {}
            
        payload = json.dumps({
            "m_type": model_type,
            "params": params
        })
        
        try:
            response = requests.post(
                url, 
                json=payload,
                timeout=self.timeout
            )
            result = response.json()
            
            if result.get('status', False):
                self.model_id = result.get('m_id')
                print(f"Model initialized successfully with ID: {self.model_id}")
                return True
            else:
                print(f"Failed to initialize model: {result}")
                return False
                
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False
    
    def get_action(self, state_data: Dict[str, Any], train: bool = True) -> Optional[Dict]:
        """
        Get action from the model based on current state
        
        Args:
            state_data: Dictionary containing state information (power, temp, pmu, etc.)
            train: If True, use training mode (with exploration), else use test mode
            
        Returns:
            Dict with 'action' and 'max_val', or None if failed
        """
        if self.model_id is None:
            print("Error: Model not initialized. Call init_model() first.")
            return None
        
        endpoint = "get_action" if train else "get_action_test"
        url = f"{self.server_url}{self.api_prefix}/{endpoint}"
        
        payload = json.dumps({
            "m_id": self.model_id,
            "data": state_data
        })
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            result = response.json()
            
            if result.get('status', False):
                return {
                    'action': result.get('action'),
                    'max_val': result.get('max_val')
                }
            else:
                print(f"Failed to get action: {result}")
                return None
                
        except Exception as e:
            print(f"Error getting action: {e}")
            return None
    
    def request_update(self) -> bool:
        """
        Request model to start training/update
        
        Returns:
            bool: True if request successful
        """
        if self.model_id is None:
            print("Error: Model not initialized.")
            return False
        
        url = f"{self.server_url}{self.api_prefix}/request_update"
        
        try:
            response = requests.get(
                url,
                params={"m_id": self.model_id},
                timeout=self.timeout
            )
            result = response.json()
            return result.get('status', False)
            
        except Exception as e:
            print(f"Error requesting update: {e}")
            return False
    
    def check_model_status(self) -> bool:
        """
        Check if model training/update is complete
        
        Returns:
            bool: True if model is ready (training complete)
        """
        if self.model_id is None:
            print("Error: Model not initialized.")
            return False
        
        url = f"{self.server_url}{self.api_prefix}/check_model_status"
        
        try:
            response = requests.get(
                url,
                params={"m_id": self.model_id},
                timeout=self.timeout
            )
            result = response.json()
            return result.get('status', False)
            
        except Exception as e:
            print(f"Error checking model status: {e}")
            return False
    
    def get_test_power(self) -> Optional[Dict]:
        """
        Get test results (power and time metrics) from server
        
        Returns:
            Dict with 'p' (power) and 't' (time), or None if failed
        """
        if self.model_id is None:
            print("Error: Model not initialized.")
            return None
        
        url = f"{self.server_url}{self.api_prefix}/get_test_power"
        
        payload = json.dumps({
            "m_id": self.model_id
        })
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            result = response.json()
            
            if result.get('status', False):
                return result.get('result')
            else:
                print(f"Failed to get test power: {result}")
                return None
                
        except Exception as e:
            print(f"Error getting test power: {e}")
            return None
    
    def train_context(self) -> bool:
        """
        Request context model training (for dqn_pro_nx)
        
        Returns:
            bool: True if training successful
        """
        if self.model_id is None:
            print("Error: Model not initialized.")
            return False
        
        url = f"{self.server_url}{self.api_prefix}/train_context"
        
        payload = json.dumps({
            "m_id": self.model_id
        })
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout * 10  # Context training takes longer
            )
            result = response.json()
            
            if result.get('status', False):
                print("Context model training complete!")
                return True
            else:
                print(f"Context training failed: {result}")
                return False
                
        except Exception as e:
            print(f"Error training context: {e}")
            return False
    
    def train_rl(self) -> bool:
        """
        Request RL controller training (for dqn_pro_nx)
        
        Returns:
            bool: True if training successful
        """
        if self.model_id is None:
            print("Error: Model not initialized.")
            return False
        
        url = f"{self.server_url}{self.api_prefix}/train_rl"
        
        payload = json.dumps({
            "m_id": self.model_id
        })
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout * 6  # RL training takes longer
            )
            result = response.json()
            
            if result.get('status', False):
                print("RL controller training complete!")
                return True
            else:
                print(f"RL training failed: {result}")
                return False
                
        except Exception as e:
            print(f"Error training RL: {e}")
            return False
    
    def remove_model(self) -> bool:
        """
        Remove model from server and clean up
        
        Returns:
            bool: True if removal successful
        """
        if self.model_id is None:
            print("Warning: No model to remove.")
            return True
        
        url = f"{self.server_url}{self.api_prefix}/rm_model"
        
        try:
            response = requests.get(
                url,
                params={"m_id": self.model_id},
                timeout=self.timeout
            )
            result = response.json()
            
            if result.get('status', False):
                print(f"Model {self.model_id} removed successfully")
                self.model_id = None
                return True
            else:
                print(f"Failed to remove model: {result}")
                return False
                
        except Exception as e:
            print(f"Error removing model: {e}")
            return False
    
    def __del__(self):
        """Cleanup: remove model when client is destroyed"""
        if self.model_id is not None:
            try:
                self.remove_model()
            except:
                pass
