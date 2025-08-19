'''
Configuration manager

by Stéphane Vujasinovic
'''

# - IMPORTS ---
import os
from configuration.configuration_base_manager import ConfigBase


# - CLASS ---
class ConfigManager(ConfigBase):
    '''
    Class for handling all the configuration files
    '''
    def get_my_configuration(
        self
    ):
        '''
        Path generator based on the configuration files
        quick use with next():
        # Get the first set of values
        first_values = next(my_config.get_my_configurations())
        print("First set of values:", first_values)

        # Get the second set of values
        second_values = next(my_config.get_my_configurations())
        print("Second set of values:", second_values)
        '''
        self.dataset_name = self._mapping.get('dataset_name')
        self.method_name = self._mapping.get('method_name')

        # Read configurations
        self.cfg_dataset_dict = self._read_dataset_configuration(self.dataset_name)
        self.cfg_results_dict = self._read_results_configuration(self.method_name)

        # Return datasets informations (GT and imgmask_operations location)
        self.image_directory = self.cfg_dataset_dict.get('image_directory')
        self.gt_mask_directory = self.cfg_dataset_dict.get('mask_directory')
        yield self.image_directory, self.gt_mask_directory

        # Return predictions results
        self.prediction_directory = self.cfg_results_dict.get('prediction_directory')
        self.pd_mask_directory = os.path.join(self.prediction_directory,
                                              self.dataset_name, 'Annotations')
        self.logits_directory = os.path.join(self.prediction_directory,
                                             self.dataset_name, 'logits')
        self.softmax_directory = os.path.join(self.prediction_directory,
                                              self.dataset_name, 'softmax')
        yield self.pd_mask_directory, self.logits_directory, \
            self.softmax_directory

        # Return Intermediate results
        self.intermediate_directory = self.cfg_results_dict.get('intermediate_directory')
        self.entropy_directory = os.path.join(self.intermediate_directory,
                                              self.dataset_name, 'Entropy')
        self.confusion_directory = os.path.join(self.intermediate_directory,
                                                self.dataset_name, 'Confusion')
        # Add additional intermediate results here
        yield self.entropy_directory, self.confusion_directory

        # Return Pre-Processing results to change in the futur
        self.preprocessing_directory = os.path.join(self.intermediate_directory,
                                                    self.dataset_name,
                                                    'Preprocessing')
        
        # Small cheat to spare the generation of the nunmber of objects, since I already did it once
        self.preprocessing_directory = os.path.join("../intermediate/Single_Models/UXMem",
                                                    self.dataset_name,
                                                    'Preprocessing')
        
        # Add additional intermediate results here
        yield self.preprocessing_directory

    def get_iou_entropy_analysis_path(
        self
    ) -> str:
        loc = self.cfg_results_dict.get("iou_entropy_analysis_directory")
        return os.path.join(loc, self.method_name, self.dataset_name)

    def get_plot_config(
        self,
        family_plots: str
    ) -> str:
        """_summary_

        Args:
            family_plots (str): _description_

        Returns:
            str: _description_
        """
        dataset_name = self._mapping.get('dataset_name')
        method_name = self._mapping.get('method_name')
        cfg_results_dict = self._read_results_configuration(method_name)
        plot_hub_directory = cfg_results_dict.get('plot_hub')

        self.plot_hub_directory = os.path.join(plot_hub_directory,
                                               family_plots,
                                               dataset_name)
        return self.plot_hub_directory

    def get_prompts_history_dir_location(
        self
    ) -> str:
        """
        If available, get the json file which lists the pompts issued during
        the sVOS evaluation.
        """
        prompts_directory = os.path.join(self.prediction_directory,
                                         self.dataset_name, 'prompts')
        print(prompts_directory)
        print(os.path.isdir(prompts_directory))
        return prompts_directory if os.path.isdir(prompts_directory) else None

    def get_benchmark_dir_location(
        self
    ):
        benchmark_directory = \
            self.cfg_results_dict.get('svos_benchmark_directory')
        benchmark_results_dir = os.path.join(benchmark_directory,
                                             self.method_name)
        return benchmark_results_dir, f"{self.dataset_name}.csv"

    def get_all_available_datasets(self):
        r = super()._read_all_datasets_available()
        return list(r.get('datasets').keys())

    def get_all_available_methods(self):
        r = super()._read_all_methods_available()
        return list(r.get('results').keys())

    def QDMN_variant_loader(self):
        return self.cfg_results_dict.get('QAM_directory')
