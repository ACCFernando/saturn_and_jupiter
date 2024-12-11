# src/config.py
import json
import os

class Config:
    """
    A configuration loader that reads settings from a JSON file.

    The loaded configuration is accessible as object attributes. For example:
    `config_instance.trading_pair` might return a string like `"BTCUSDT"`.

    Attributes
    ----------
    config : dict
        A dictionary containing the loaded configuration values.
    """

    def __init__(self, config_path: str):
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def __getattr__(self, item: str):
        """
        Retrieve configuration parameters by attribute access.

        Parameters
        ----------
        item : str
            The name of the configuration parameter to retrieve.

        Returns
        -------
        Any
            The value of the requested configuration parameter, or None if not found.
        """
        return self.config.get(item, None)
