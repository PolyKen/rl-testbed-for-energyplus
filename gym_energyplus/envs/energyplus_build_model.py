# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from re import match
import os

from gym_energyplus.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp import \
    EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp
from gym_energyplus.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp_Fan import \
    EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan
from gym_energyplus.envs.energyplus_model_test import EnergyPlusModelTest


def build_ep_model(model_file, log_dir, verbose=False):
    model_basename = os.path.splitext(os.path.basename(model_file))[0]

    if match('2ZoneDataCenterHVAC_wEconomizer_Temp_Fan.*', model_basename):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan(model_file=model_file, log_dir=log_dir,
                                                                        verbose=verbose)
    elif match('2ZoneDataCenterHVAC_wEconomizer_Temp.*', model_basename):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(model_file=model_file, log_dir=log_dir,
                                                                    verbose=verbose)
    elif match('2ZoneDataCenterHVAC_wEconomizer-baseline.*', model_basename):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(model_file=model_file, log_dir=log_dir,
                                                                    verbose=verbose)
    elif match('RefBldgLarge.*', model_basename):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(model_file=model_file, log_dir=log_dir,
                                                                    verbose=verbose)
    elif match('LargeOfficeGuangzhou.*', model_basename):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(model_file=model_file, log_dir=log_dir,
                                                                    verbose=verbose)
    elif match('Test.*', model_basename):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(model_file=model_file, log_dir=log_dir, verbose=verbose)
    
    elif match('Exercise2C-Solution-ems.*', model_basename):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(model_file=model_file, log_dir=log_dir, verbose=verbose)
    
    else:
        print('model basename = ' + model_basename)
        raise ValueError('Unsupported EnergyPlus model')

    return model
