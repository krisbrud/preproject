import matplotlib as plt
# matplotlib import fixes absurd bug where stable_baselines3 imports
# matplotlib in a way that causes an error

from utils.parsing import parse_experiment_info
from utils.training_schemes import train_end_to_end_path_follow

from config import get_cfg_defaults

def get_config(verbose=True):
    cfg = get_cfg_defaults()
    cfg.merge_from_file("first-experiment.yaml") # Needs to be updated here if using different experiment config
    cfg.freeze()

    if verbose:
        print("Config:\n", cfg)
    
    return cfg


def main():
    cfg = get_config()


    env_name = cfg.env.name
        
    train_end_to_end_path_follow(hyperparams=hyperparams,
                                 experiment_config=experiment_config,
                                 env_name=env_name)
        

if __name__ == '__main__':
    main()
