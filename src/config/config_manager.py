import yaml

class ConfigManager:
    def __init__(self, config_path='src/config/settings.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, key):
        return self.config.get(key, None)
