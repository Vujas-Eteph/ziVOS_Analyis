'''
Base configuration manager

by Stéphane Vujasinovic
'''

# - IMPORTS ---
from typing import Dict
import os
from abc import ABC
import yaml


# - BASE CLASS ---
class ConfigBase(ABC):
    '''
    Base class for basic functions and storing
    the basic configuration files locations
    '''

    def __init__(
        self
    ):
        """_summary_
        """
        # Init. variables
        self.image_directory = None
        self.gt_mask_directory = None
        self.pd_mask_directory = None
        self.logits_directory = None
        self.softmax_directory = None
        self.entropy_directory = None
        self.confusion_directory = None
        self.preprocessing_directory = None
        self.plot_hub_directory = None

        # Init. mapping
        self._mapping = {'datasets': './configuration/config_datasets.yaml',
                         'results': './configuration/config_results.yaml',
                         'dataset_name': None,
                         'method_name': None
                         }

    def __getitem__(
        self,
        key: str
    ) -> str:
        '''
        Check the location of the configuration files
        '''
        return self._mapping.get(key, None)

    def __setitem__(
        self,
        key,
        value
    ):
        '''
        Update the location of the configuration files
        '''
        self._mapping[key] = value

    @staticmethod
    def path_exists(
        path_to_file: str
    ) -> bool:
        """_summary_

        Args:
            path_to_file (str): _description_

        Returns:
            bool: _description_
        """
        return os.path.exists(path_to_file)

    @staticmethod
    def _read_yaml_configuration(
        path_to_config_file: str
    ) -> Dict:
        """_summary_

        Args:
            path_to_config_file (str): _description_

        Raises:
            FileNotFoundError: _description_
            RuntimeError: _description_

        Returns:
            Dict: _description_
        """
        if not ConfigBase.path_exists(path_to_config_file):
            raise FileNotFoundError(f"File not found: {path_to_config_file}")

        with open(path_to_config_file, 'r', encoding="utf-8") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise RuntimeError("Error while trying to read the following ",
                                   f"YAML file: {path_to_config_file}") from e

    def _read_all_datasets_available(self):
        return self._read_yaml_configuration(self._mapping.get('datasets'))

    def _read_dataset_configuration(
        self,
        dataset_name: str
    ) -> Dict:
        '''
        Return the datasets directory location
        '''
        yaml_data = self._read_all_datasets_available()
        global_cfg_dataset_dict = yaml_data.get('datasets')

        return global_cfg_dataset_dict.get(dataset_name)

    def _read_all_methods_available(self):
        return self._read_yaml_configuration(self._mapping.get('results'))

    def _read_results_configuration(
        self,
        method_name: str
    ) -> Dict:
        '''
        Return the location of the directory results of a particular model
        '''
        yaml_data = self._read_all_methods_available()
        global_cfg_results_dict = yaml_data.get('results')

        return global_cfg_results_dict.get(method_name)
