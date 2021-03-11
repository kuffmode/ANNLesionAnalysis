# Importing libraries
import pickle
import neat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# importing necessary libraries
from collections import deque
import pickle
import gc
import warnings
import cv2
import gym
from joblib import Parallel, delayed
from keras import Model
import neat
from sklearn.preprocessing import StandardScaler
import os
import copy
import random
import time
import tensorflow as tf
import experiment_toolbox as exto
# tensorflow has some weird warnings, this will hide them
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


def load_model(filename, dense_number):
    # Loads autoencoder. Specify the bottleneck layer
    with open(filename, 'rb') as f:
        autoencoder = pickle.load(f)
    model = Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer(f'dense_{dense_number}').output
    )
    return model


def preprocess(observation):
    grayscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)  # converting colours to grayscale
    resize = cv2.resize(grayscale, (53, 70))  # resizing the screen
    cropped = resize[8:65, 3:50]  # cropping the score and ground part [from top : to bottom, from left : to right]
    _, thresh1 = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY)  # thresholding the pixel values to either 0 or 255
    normalized = thresh1 / 255  # binarizing the values
    return normalized.astype(np.float32).reshape(1, -1)


def reduce_dimensions(frame_space: np.ndarray, model):
    latent_space = model.predict(frame_space)  # feeding the screen to the autoencoder
    return latent_space.reshape(-1, 1)


def frame_to_state(observation, model):
    frame_space = preprocess(observation)  # preprocesses the screen
    latent_space = reduce_dimensions(frame_space, model)  # feeds the preprocessed screen to the autoencoder
    latent_space_scaled = StandardScaler().fit_transform(latent_space)
    del latent_space
    return latent_space_scaled


def init_states_stack(frames_per_state, nfeatures):
    # generates an empty placeholder to be updated by features incrementally.
    stacked_state = deque([], frames_per_state * nfeatures)
    for no_feature in range(stacked_state.maxlen):
        stacked_state.append(0.)
    return stacked_state


def append_states_stack(stack, latent_space):
    # adds the latent space to the empty placeholder
    for feature in latent_space:
        stack.append(feature)
    return stack


def play_game(environment_name: str = 'SpaceInvaders-v4',
              model_filename: str = "",
              genome: neat.genome.DefaultGenome = None,
              config: neat.Config = None,
              frames_per_state: int = 2,
              nfeatures: int = 6,
              noise_input=False):
    # environment != observation. observation is a state from the environment
    environment = gym.make(environment_name)
    model = load_model(model_filename, 4)

    # initiates a network from the trained genome and its config file
    network = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)

    stack = init_states_stack(frames_per_state, nfeatures)
    current_fitness = 0
    observation = environment.reset()  # the first frame is generated here
    done = False
    while not done:
        state = frame_to_state(observation, model)
        stack = append_states_stack(stack, state)

        if noise_input is False:
            # network assigns values to actions according to the given state
            actions = network.activate(stack)
        else:
            # feeds noise to the network instead of the latent space of AE
            actions = network.activate(np.random.random(len(stack)))
        # picks the action with the highest value
        max_action_idx = np.argmax(actions)

        # feeds it to the environment and collects the reward if there is any
        observation, reward, done, info = environment.step(max_action_idx)
        current_fitness += reward

    environment.close()
    stack.clear()
    del model, stack, state, environment
    tf.keras.backend.clear_session()
    # to prevent RAM build-up
    gc.collect()
    return current_fitness


def play_games(game_function, n_games: int, n_jobs: int = -2, **params):
    # parallelize agents (n_jobs) to play simultaneously (n_games)
    results = Parallel(n_jobs=n_jobs,
                       backend='multiprocessing',
                       prefer='processes')(
        delayed(game_function)(**params)
        for _ in range(n_games))
    print(f'The average score is: {np.mean(results)}')
    return results


def random_agent(n_trials):
    # randomly choosing actions
    dataset = []
    environment = gym.make('SpaceInvaders-v4')
    for trial in range(n_trials):
        done = False
        score = 0
        environment.reset()
        while not done:
            _, reward, done, _ = environment.step(environment.action_space.sample())
            score += reward
        dataset.append(score)
        print(f'trial:{trial}, score: {score}')
    environment.close()
    return dataset


def compute_shapley(genome,
                    scores_table,
                    shapley_table,
                    ae_file,
                    config,
                    n_samples=100):
    # get the links out
    links = list(genome.connections.keys())

    # have a backup to reset the network
    genome_backup = copy.deepcopy(genome)

    for sample in range(n_samples):
        # generates a permutation
        links_sequence = tuple(random.sample(links, len(links)))
        shapley_vals = []

        # goes through all of the corresponding combinations
        for i, _ in enumerate(links_sequence):
            seq1 = frozenset(links_sequence[: i + 1])
            seq2 = frozenset(links_sequence[: i])
            # checks if a performance for these combinations exist, if not, plays the game.
            if seq1 not in scores_table:
                genome = copy.deepcopy(genome_backup)
                # keeps the selected link active (the rest is deactivated)
                for link in seq1:
                    genome.connections[link].enabled = True
                params = dict(
                    environment_name='SpaceInvaders-v4',
                    model_filename=ae_file,
                    genome=genome,
                    config=config,
                    frames_per_state=2,
                    nfeatures=6,
                    noise_input=True)

                performance = play_games(play_game,
                                         n_games=16,
                                         n_jobs=-1, **params)
                scores_table[seq1] = np.mean(performance)
                del genome

            if seq2 not in scores_table:
                genome = copy.deepcopy(genome_backup)
                for link in seq2:
                    genome.connections[link].enabled = True
                params = dict(
                    environment_name='SpaceInvaders-v4',
                    model_filename=ae_file,
                    genome=genome,
                    config=config,
                    frames_per_state=2,
                    nfeatures=6,
                    noise_input=True
                )

                performance = play_games(play_game,
                                         n_games=16,
                                         n_jobs=-1, **params)
                scores_table[seq2] = np.mean(performance)
                del genome

            # calculates the shapley value for that permutation
            shapley_vals.append(scores_table[seq1]
                                - scores_table[seq2])
        # saves it to a permutations:performance pair dataset
        shapley_table[links_sequence] = shapley_vals
        print('Permutation number: ', sample)




# The files
config_file = '/home/kayson/PycharmProjects/' \
              'shapleyvalue/AEconfig-SI.txt'

genome_file = '/home/kayson/PycharmProjects/' \
              'shapleyvalue/network_3_fitness_1300_.pkl'

ae_file = '/home/kayson/PycharmProjects/shapleyvalue/' \
          'SI-AE-model.pkl'

config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,
                     neat.DefaultStagnation,
                     config_file)

with open(genome_file, 'rb') as f:
    optimized_genome = pickle.load(f)

# NEAT produces junk (disabled) connections, this line prunes them
exto.pruner(optimized_genome)

shapley_table = {}  # unsorted pair of shapley values and elements
scores_table = {}   # multi-perturbation dataset, load if one already exists
n_samples = 1000

# Calculating shapley values, depending on how many jobs you can run in parallel
# and how many trials (n_samples) you specified, this part might take days.
# 15 jobs, 1000 samples = around 1 week, 100 samples = a day

compute_shapley(optimized_genome,
                scores_table,
                shapley_table,
                ae_file,
                config,
                n_samples = n_samples)

# sorting shapley values and elements in a clean pandas table
shapley_sorted = pd.DataFrame([dict(zip(seq, vals))
                               for seq, vals in shapley_table.items()])


# reindex them according to averages from the smallest to largest shapley value, for visual clarity.
shapley_sorted = exto.preprocess_plot(shapley_sorted)

# saving datasets, in both csv and pkl format. multi-perturbation dataset is better to be just pkl
shapley_sorted.to_csv(f'SV_BL{n_samples}.csv', index=False)

with open(f'SV_BL{n_samples}.pkl', 'wb') as output:
    pickle.dump(shapley_sorted, output, 1)

with open(f'SV_BL{n_samples}_scores_table.pkl', 'wb') as output:
    pickle.dump(scores_table, output, 1)


# plotting the results
x_ticks = list(shapley_sorted.columns)

plt.figure(figsize=(16, 6), dpi=300)
sns.pointplot(data=shapley_sorted, ci=95, join=False, capsize=.3,
              n_boot=10000, color='k', errwidth=3)

plt.grid(color='k', linewidth=1, alpha = 0.15, axis= 'both')
plt.axhline(0, color='r', linewidth=2, alpha = 0.3)
plt.xlabel('connections (source → destination)', fontsize=14, fontweight='bold')
plt.xticks(np.arange(len(x_ticks)), labels=x_ticks, rotation=90, fontsize=12)
plt.yticks(range(-120, 141, 20), fontsize=12, fontweight='bold')
plt.ylabel('Shapley Values', fontsize=14, fontweight='bold')
plt.title(f'Estimated Shapley Value of all links (Blinded network), CI = %95, Number of permutations = {n_samples}',
          fontsize=16, fontweight='bold')
#plt.show()
plt.savefig(f'SV_BL_SD{n_samples}.pdf', bbox_inches='tight')

