import yaml
import os
from typing import Dict, Any

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Get the base directory (project root)
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load config
        config_path = os.path.join(self.base_path, "config", "settings.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.settings = yaml.safe_load(f)
                
            # Convert relative paths to absolute paths
            data_paths = self.settings.get('paths', {}).get('data', {})
            for key, path in data_paths.items():
                if path.startswith('./'):
                    data_paths[key] = os.path.join(self.base_path, path[2:])
                else:
                    data_paths[key] = os.path.join(self.base_path, path)
        else:
            self.settings = self._default_settings()

    def _default_settings(self) -> Dict[str, Any]:
        return {
            'model': {
                'provider': 'gemini',
                'model_name': 'gemini-2.5-flash',
                'timeout': 30
            },
            'security': {
                'encryption_enabled': True,
                'max_file_size_mb': 10,
                'allowed_file_types': ['png', 'jpg', 'jpeg', 'pdf']
            },
            'pharmacy': {
                'max_delivery_radius_km': 15,
                'base_delivery_fee': 25
            },
            'medical': {
                'min_confidence_threshold': 0.3,
                'severity_thresholds': {
                    'mild': 0.3,
                    'moderate': 0.6,
                    'severe': 0.8
                }
            }
        }