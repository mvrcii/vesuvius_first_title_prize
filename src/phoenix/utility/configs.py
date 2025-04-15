import importlib.util
import os.path
import pprint
import sys


class Config:
    """
    A class to manage configuration settings for the application.

    The configuration settings can be loaded from a Python file, and the current
    configuration can be saved back to a file. The class allows attribute-style
    access to configuration parameters.

    Attributes:
        Any configuration parameter loaded from the configuration file.

    Methods:
        load_from_file(config_path): Class method to load configuration from a file.
        save_to_file(file_path): Instance method to save current configuration to a file.
    """

    def __init__(self, dictionary=None, config_file_name=None, append_info="", prepend_info="", save=True):
        """
        Initialize the Config object with a dictionary of configuration parameters.

        If a dictionary is provided, the method sets each key-value pair in the
        dictionary as an attribute of the Config object.

        Args:
            dictionary (dict, optional): A dictionary of configuration parameters.
                                         Defaults to None.
        """
        if dictionary:
            for key, value in dictionary.items():
                setattr(self, key, value)
        self.save = save
        self.config_file_name = config_file_name
        self.append_info = append_info
        self.prepend_info = prepend_info

    @classmethod
    def load_local_cfg(cls):
        config = {}

        # Check for and apply local configuration overrides
        local_config_path = 'src/conf_local.py'
        if os.path.exists(local_config_path):
            local_config = cls.import_config_from_path(local_config_path)
            config.update({k: v for k, v in vars(local_config).items() if not k.startswith('__')})

        return cls(config, None, save=False)

    @classmethod
    def load_from_file(cls, config_path):
        ignore_keys = ['sys', 'A', 'os']
        model_config = cls.import_config_from_path(config_path)
        config = {}

        def filter_keys(items):
            return {k: v for k, v in items if not k.startswith('__') and k not in ignore_keys}

        # Update config with base and model configurations
        if hasattr(model_config, '_base_'):
            for base_path in model_config._base_:
                # Try to load base configs, if it doesn't work continue with warning
                try:
                    base_config = cls.import_config_from_path(base_path)
                    config.update(filter_keys(vars(base_config).items()))
                except Exception as e:
                    print(f"Warning: Could not load base config from {base_path}: {e}")

        config.update(filter_keys(vars(model_config).items()))

        append_info = ""
        # Read the source file
        with open(config_path, 'r') as file:
            prepend_info = ''.join(line for line in file if line.startswith('import'))
            prepend_info += ''.join(line for line in file if line.startswith('from'))
            prepend_info += ''.join(line for line in file if line.startswith('sys.path'))
            prepend_info += ''.join(line for line in file if line.startswith('sys'))
            prepend_info += ''.join(line for line in file if line.startswith('os'))
            prepend_info += ''.join(line for line in file if line.startswith('module'))

        # Check for and apply local configuration overrides
        local_config_path = 'src/conf_local.py'
        if os.path.exists(local_config_path):
            local_config = cls.import_config_from_path(local_config_path)
            config.update(filter_keys(vars(local_config).items()))
        else:
            print("Local config not found!")

        if config_path.__contains__('/'):
            config_file_name = config_path.split('/')[-1]
        elif config_path.__contains__('\\'):
            config_file_name = config_path.split('\\')[-1]
        else:
            print("Error: Config path incorrect.")
            sys.exit(1)
        return cls(config, config_file_name, append_info=append_info, prepend_info=prepend_info)

    @classmethod
    def load_from_dict(cls, dictionary):
        """
        Initialize a Config object from a dictionary.

        Similar to load_from_file, but takes a dictionary directly instead of loading from a file.
        Handles base configurations if specified in the dictionary and applies local configuration
        overrides if available.

        Args:
            dictionary (dict): The dictionary containing configuration parameters.
            save (bool, optional): Whether the configuration can be saved. Defaults to True.

        Returns:
            Config: A new Config instance with the specified configuration.
        """
        ignore_keys = ['sys', 'A', 'os']
        config = {}

        def filter_keys(items):
            return {k: v for k, v in items if not k.startswith('__') and k not in ignore_keys}

        # Handle base configurations if specified
        if '_base_' in dictionary:
            for base_path in dictionary['_base_']:
                try:
                    base_config = cls.import_config_from_path(base_path)
                    config.update(filter_keys(vars(base_config).items()))
                except Exception as e:
                    print(f"Warning: Could not load base config from {base_path}: {e}")

        # Update with the provided dictionary
        config.update(filter_keys(dictionary.items()))

        # Check for and apply local configuration overrides
        local_config_path = 'src/conf_local.py'
        if os.path.exists(local_config_path):
            local_config = cls.import_config_from_path(local_config_path)
            config.update(filter_keys(vars(local_config).items()))
        else:
            print("Local config not found!")

        return cls(config, None)

    def save_to_file(self, model_run_dir, file_path=None):
        """
        Save the current configuration to a Python file.

        If a file path is provided, the configuration is saved to that path. If no file path
        is provided, the method attempts to save the configuration to the original configuration
        file's path that was used to load this configuration (if available). If neither a file path
        is provided nor an original path is available, a ValueError is raised.

        Each configuration parameter is written to the file as a line in the format:
        'key = value', where 'key' is the name of the configuration parameter, and
        'value' is its value represented as a Python literal.

        Args:
            file_path (str, optional): The file path where the configuration will be saved.
                                       If None, tries to use the original configuration file's path.
                                       Defaults to None.

        Raises:
            ValueError: If both file_path is None and the original configuration file's path is unknown.
        """
        if not self.save:
            print("Only a local config has been loaded, which cannot be saved.")
            sys.exit(1)

        if file_path is None:
            file_path = self.config_file_name
            if file_path is None:
                raise ValueError("Target file path not specified and original config path unknown.")

        keys_to_ignore = [
            'config_file_name',
            'train_aug',
            'val_aug',
            'os',
            'A',
            'prepend_info',
            'append_info'
        ]

        file_path = os.path.join(model_run_dir, self.config_file_name)
        with open(file_path, 'w') as f:
            f.write(f'{self.prepend_info}')
            for key, value in self.__dict__.items():
                if key not in keys_to_ignore:
                    f.write(f'{key} = {repr(value)}\n')
            f.write(f'{self.append_info}')

    def to_clean_dict(self) -> dict:
        keys_to_ignore = [
            'config_file_name',
            'train_aug',
            'val_aug',
            'os',
            'A',
            'prepend_info',
            'append_info'
        ]
        return {k: v for k, v in self.__dict__.items() if k not in keys_to_ignore}

    def __str__(self):
        """
        Return a string representation of the configuration, in a pretty-printed format.

        Returns:
            str: A formatted string of the configuration dictionary.
        """
        return pprint.pformat(self.__dict__, indent=4, width=1)

    @staticmethod
    def import_config_from_path(path):
        module_name = os.path.basename(path).split('.')[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
