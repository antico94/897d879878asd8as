import yaml
from pathlib import Path
from typing import Any

class Config:
    def __init__(self, file_path: str = 'Configuration/Configuration.yaml') -> None:
        self._config = yaml.safe_load(Path(file_path).read_text())

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get_nested(self, *keys: str) -> Any:
        value = self._config
        for key in keys:
            value = value[key]
        return value
