import yaml
from pathlib import Path
import os
from typing import Any


class Config:
    def __init__(self, file_path: str = 'Configuration/Configuration.yaml') -> None:
        # Find the project root directory
        project_root = self._find_project_root()

        # Create an absolute path to the config file
        config_path = project_root / file_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        self._config = yaml.safe_load(config_path.read_text())

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get_nested(self, *keys: str) -> Any:
        value = self._config
        for key in keys:
            value = value[key]
        return value

    def _find_project_root(self) -> Path:
        """Find the project root directory containing the Configuration folder."""
        # Start from the current working directory
        current_dir = Path(os.getcwd())

        # Check if we're already at the project root (has Configuration directory)
        if (current_dir / 'Configuration').exists():
            return current_dir

        # Search parent directories for project root
        while True:
            # Check for Configuration directory at this level
            if (current_dir / 'Configuration').exists():
                return current_dir

            # Get parent directory
            parent_dir = current_dir.parent

            # If we're at the root of the filesystem and haven't found it
            if parent_dir == current_dir:
                # As a fallback, search in the directory where this script resides
                script_dir = Path(__file__).resolve().parent.parent
                if (script_dir / 'Configuration').exists():
                    return script_dir

                # If all fails, raise an error
                raise FileNotFoundError(
                    "Could not find project root containing Configuration directory. "
                    "Make sure the Configuration directory exists in the project."
                )

            # Move up to parent directory
            current_dir = parent_dir