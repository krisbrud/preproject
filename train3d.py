import matplotlib as plt
# matplotlib import fixes absurd bug where stable_baselines3 imports
# matplotlib in a way that causes an error

from utils.parsing import parse_experiment_info
from utils.training_schemes import train_end_to_end_path_follow

def main():
    experiment_config = parse_experiment_info()

    n_repetitions = experiment_config.n_repetitions 


    env_name = "PathFollowAuv3D-v0"
    # env_name = "Track2d-v0"
    for i in range(n_repetitions):
        if n_repetitions > 1:
            print(f"Experiment repetition {i} started!")
        
        train_end_to_end_path_follow(hyperparams=hyperparams,
                                 experiment_config=experiment_config,
                                 env_name=env_name)
        

if __name__ == '__main__':
    main()
