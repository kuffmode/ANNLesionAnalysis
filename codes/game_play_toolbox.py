# importing necessary libraries
from collections import deque
import pickle
import gc
import warnings
import cv2
import gym
from joblib import Parallel, delayed
from keras import Model
import numpy as np
import neat
from sklearn.preprocessing import StandardScaler
import os
import copy
import random
import time
import pandas as pd
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
                # keeps the selected link active (the rest is deactive)
                for link in seq1:
                    genome.connections[link].enabled = True
                params = dict(
                    environment_name='SpaceInvaders-v4',
                    model_filename=ae_file,
                    genome=genome,
                    config=config,
                    frames_per_state=2,
                    nfeatures=6,
                    noise_input=False)

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
                    nfeatures=6
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


def play_sequences(genome,
                   scores_table,
                   pair,
                   ae_file,
                   config,
                   A_noB=False,
                   B_noA=False,
                   compound=False,
                   n_samples=100):
    performances = []
    genome_backup = copy.deepcopy(genome)
    links = list(genome.connections.keys())
    A = pair[0]
    B = pair[1]
    links_backup = copy.deepcopy(links)

    params = dict(environment_name='SpaceInvaders-v4',
                  model_filename=ae_file,
                  config=config,
                  frames_per_state=2,
                  nfeatures=6)

    for sample in range(n_samples):
        tic = time.perf_counter()
        print(f'trial number: {sample}')

        # generates a permutation with (i,j) as a compound element.
        if compound is True:
            AB = tuple((A, B))
            links.append(AB)  # adds the compound to the list
            links.pop(links.index(A))  # removes individual element from the list
            links.pop(links.index(B))  # removes individual element from the list
            links_sequence = list(random.sample(links, len(links)))  # shuffles to have a permutation
            links_sequence.insert(links_sequence.index(AB), A)  # puts back the elements on the new location
            links_sequence.insert(links_sequence.index(AB) + 1, B)  # puts back the elements on the new location
            links_sequence.pop(links_sequence.index(AB))  # removes the compound
            links_sequence = tuple(links_sequence)
            seq1 = frozenset(links_sequence[: links_sequence.index(B) + 1])
            seq2 = frozenset(links_sequence[: links_sequence.index(A)])

        else:
            links_sequence = tuple(random.sample(links, len(links)))

        if A_noB is True:  # takes the sequence out of the permutation wrt. which element is lesioned
            seq1 = frozenset(links_sequence[: links_sequence.index(A) + 1])
            seq2 = frozenset(links_sequence[: links_sequence.index(A)])
        elif B_noA is True:  # same
            seq1 = frozenset(links_sequence[: links_sequence.index(B) + 1])
            seq2 = frozenset(links_sequence[: links_sequence.index(B)])

        # checks if there is a score for that combination, if not, plays.
        if seq1 not in scores_table:
            print('playing the first sequence')
            genome = copy.deepcopy(genome_backup)
            for link in seq1:
                genome.connections[link].enabled = True
            params['genome'] = genome
            performance_seq1 = play_games(play_game,
                                          n_games=16,
                                          n_jobs=-1, **params)
            scores_table[seq1] = np.mean(performance_seq1)
            del genome

        if seq2 not in scores_table:
            print('playing the second sequence')
            genome = copy.deepcopy(genome_backup)
            for link in seq2:
                genome.connections[link].enabled = True
            params['genome'] = genome
            performance_seq2 = play_games(play_game,
                                          n_games=16,
                                          n_jobs=-1, **params)
            scores_table[seq2] = np.mean(performance_seq2)
            del genome
        performances.append(scores_table[seq1] - scores_table[seq2])
        links = copy.deepcopy(links_backup)
        toc = time.perf_counter()
        print(f'elapsed time: {toc - tic: 0.4f} seconds')
    return np.array(performances)


# this is for one complete round of calculating interactions
def compute_shapley_interaction(genome,
                                scores_table,
                                pair,
                                ae_file,
                                config,
                                n_samples):
    original_genome = copy.deepcopy(genome)

    print(f'pair: {pair}')
    A = pair[0]
    B = pair[1]
    # remove B and play with A, N times
    genome = copy.deepcopy(original_genome)
    for link in list(genome.connections.keys()):
        if genome.connections[link].key == B:
            genome.connections.pop(link)

    print('Playing with A without B')
    A_noB = play_sequences(genome,
                           scores_table,
                           pair,
                           ae_file,
                           config,
                           A_noB=True,
                           B_noA=False,
                           compound=False,
                           n_samples=n_samples)

    # remove A and play with B, N times
    genome = copy.deepcopy(original_genome)
    for link in list(genome.connections.keys()):
        if genome.connections[link].key == A:
            genome.connections.pop(link)

    print('Playing with B without A')
    B_noA = play_sequences(genome,
                           scores_table,
                           pair,
                           ae_file,
                           config,
                           A_noB=False,
                           B_noA=True,
                           compound=False,
                           n_samples=n_samples)

    # compound (A,B) and play with both
    genome = copy.deepcopy(original_genome)
    print('Playing the compound (A,B)')
    AB_compound = play_sequences(genome,
                                 scores_table,
                                 pair,
                                 ae_file,
                                 config,
                                 A_noB=False,
                                 B_noA=False,
                                 compound=True,
                                 n_samples=n_samples)

    interactions = AB_compound - A_noB - B_noA
    interactions_gamma_a = interactions + A_noB
    return interactions, interactions_gamma_a, A_noB


def compute_nodal_shapley(genome, all_nodes, links, n_trials, scores_table, ae_file, config):
    genome_backup = copy.deepcopy(genome)
    nodal_shapley_table = pd.DataFrame(columns=all_nodes, index=np.arange(n_trials))

    for node in all_nodes:
        print(f'Node:{node} ------------------------------\n')
        for trial in range(n_trials):
            print(f'trial number: {trial} ---------------------\n')
            # shuffling nodes
            permutation = list(random.sample(all_nodes, len(all_nodes)))

            # finding the first node sequence and translating it to links
            node_seq1 = frozenset(permutation[: permutation.index(node) + 1])
            translated_links_seq1 = []

            for element in node_seq1:
                exto.node_to_link(links, element, translated_links_seq1)

            # playing the first sequence, if necessary
            if frozenset(translated_links_seq1) not in scores_table:
                genome = copy.deepcopy(genome_backup)
                for link in translated_links_seq1:
                    genome.connections[link].enabled = True

                params = dict(
                    environment_name='SpaceInvaders-v4',
                    model_filename=ae_file,
                    genome=genome,
                    config=config,
                    frames_per_state=2,
                    nfeatures=6
                )
                performance_seq1 = play_games(play_game,
                                              n_games=16,
                                              n_jobs=-1,
                                              **params)

                scores_table[frozenset(translated_links_seq1)] = np.mean(performance_seq1)
                del genome
            # ------------------------------------------------------------
            # finding the second node sequence and translating it to links
            node_seq2 = frozenset(permutation[: permutation.index(node)])
            translated_links_seq2 = []

            for element in node_seq2:
                exto.node_to_link(links, element, translated_links_seq2)

            # playing the first sequence, if necessary
            if frozenset(translated_links_seq2) not in scores_table:
                genome = copy.deepcopy(genome_backup)
                for link in translated_links_seq2:
                    genome.connections[link].enabled = True

                params = dict(
                    environment_name='SpaceInvaders-v4',
                    model_filename=ae_file,
                    genome=genome,
                    config=config,
                    frames_per_state=2,
                    nfeatures=6
                )
                performance_seq2 = play_games(play_game,
                                              n_games=16,
                                              n_jobs=-1,
                                              **params)

                scores_table[frozenset(translated_links_seq2)] = np.mean(performance_seq2)
                del genome
            # ------------------------------------------------------------
            nodal_shapley_table[node][trial] = \
                scores_table[frozenset(translated_links_seq1)] - scores_table[frozenset(translated_links_seq2)]
    return nodal_shapley_table
