import time
import pickle
import gc
import warnings
import cv2
import gym
import numpy as np
import pandas as pd
import neat
import os
import copy
import random
import sys
from collections import deque
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from keras import Model
from datetime import datetime
import tensorflow as tf
from shutil import copyfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


def load_model(filename, dense_number):
    """Load autoencoder"""
    with open(filename, 'rb') as f:
        autoencoder = pickle.load(f)
    model = Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer(f'dense_{dense_number}').output
    )
    return model


def preprocess(observation):
    grayscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(grayscale, (53, 70))
    cropped = resize[8:65, 3:50]  # [from top : to bottom, from left : to right]
    _, thresh1 = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY)
    normalized = thresh1 / 255
    return normalized.astype(np.float32).reshape(1, -1)


def reduce_dimensions(frame_space: np.ndarray, model):
    latent_space = model.predict(frame_space)
    return latent_space.reshape(-1, 1)


def frame_to_state(observation, model):
    frame_space = preprocess(observation)
    latent_space = reduce_dimensions(frame_space, model)
    latent_space_scaled = StandardScaler().fit_transform(latent_space)
    del latent_space
    return latent_space_scaled


def init_states_stack(frames_per_state, nfeatures):
    stacked_state = deque([], frames_per_state * nfeatures)
    for no_feature in range(stacked_state.maxlen):
        stacked_state.append(0.)
    return stacked_state


def append_states_stack(stack, latent_space):
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
    environment = gym.make(environment_name)
    model = load_model(model_filename, 4)
    network = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)

    stack = init_states_stack(frames_per_state, nfeatures)
    current_fitness = 0

    observation = environment.reset()
    done = False
    while not done:
        state = frame_to_state(observation, model)
        stack = append_states_stack(stack, state)
        if noise_input is False:
            actions = network.activate(stack)
        else:
            actions = network.activate(np.random.random(len(stack)))

        max_action_idx = np.argmax(actions)
        observation, reward, done, info = environment.step(max_action_idx)
        current_fitness += reward

    environment.close()
    stack.clear()
    del model, stack, state, environment

    gc.collect()
    tf.keras.backend.clear_session()

    return current_fitness


def play_games(game_function, n_games: int, n_jobs: int = -2, **params):
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing', prefer='processes')(
        delayed(game_function)(**params)
        for _ in range(n_games))
    print(f'The average score is: {np.mean(results)}')
    return results


def pruner(genome):
    for link in list(genome.connections.keys()):
        if genome.connections[link].enabled == False:
            genome.connections.pop(link)


def play_sequences(genome,
                   scores_table,
                   pair,
                   ae_file_destination,
                   config,
                   A_noB=False,
                   B_noA=False,
                   compound=False):
    performances = []
    genome_backup = copy.deepcopy(genome)
    links = list(genome.connections.keys())
    A = pair[0]
    B = pair[1]
    links_backup = copy.deepcopy(links)

    tic = time.perf_counter()
    if compound is True:
        AB = tuple((A, B))
        links.append(AB)
        links.pop(links.index(A))
        links.pop(links.index(B))
        links_sequence = list(random.sample(links, len(links)))
        links_sequence.insert(links_sequence.index(AB), A)
        links_sequence.insert(links_sequence.index(AB) + 1, B)
        links_sequence.pop(links_sequence.index(AB))
        links_sequence = tuple(links_sequence)
        seq1 = frozenset(links_sequence[: links_sequence.index(B) + 1])
        seq2 = frozenset(links_sequence[: links_sequence.index(A)])

    else:
        links_sequence = tuple(random.sample(links, len(links)))

    if A_noB is True:
        seq1 = frozenset(links_sequence[: links_sequence.index(A) + 1])
        seq2 = frozenset(links_sequence[: links_sequence.index(A)])
    elif B_noA is True:
        seq1 = frozenset(links_sequence[: links_sequence.index(B) + 1])
        seq2 = frozenset(links_sequence[: links_sequence.index(B)])

    if seq1 not in scores_table:
        print('playing the first sequence')
        genome = copy.deepcopy(genome_backup)
        for link in seq1:
            genome.connections[link].enabled = True
        params = dict(
            environment_name='SpaceInvaders-v4',
            model_filename=ae_file_destination,
            genome=genome,
            config=config,
            frames_per_state=2,
            nfeatures=6
        )
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
        params = dict(
            environment_name='SpaceInvaders-v4',
            model_filename=ae_file_destination,
            genome=genome,
            config=config,
            frames_per_state=2,
            nfeatures=6
        )

        performance_seq2 = play_games(play_game,
                                      n_games=16,
                                      n_jobs=-1, **params)
        scores_table[seq2] = np.mean(performance_seq2)
        del genome
    performances.append(scores_table[seq1] - scores_table[seq2])
    links = copy.deepcopy(links_backup)
    toc = time.perf_counter()
    print(f'elapsed time of the sequence: {toc - tic: 0.4f} seconds')
    return np.array(performances)


def compute_shapley_interaction(genome,
                                scores_table,
                                pair,
                                ae_file_destination,
                                config):
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
                           ae_file_destination,
                           config,
                           A_noB=True,
                           B_noA=False,
                           compound=False)

    # remove A and play with B, N times
    genome = copy.deepcopy(original_genome)
    for link in list(genome.connections.keys()):
        if genome.connections[link].key == A:
            genome.connections.pop(link)

    print('Playing with B without A')
    B_noA = play_sequences(genome,
                           scores_table,
                           pair,
                           ae_file_destination,
                           config,
                           A_noB=False,
                           B_noA=True,
                           compound=False)

    # compound (A,B) and play with both
    genome = copy.deepcopy(original_genome)
    print('Playing the compound (A,B)')
    AB_compound = play_sequences(genome,
                                 scores_table,
                                 pair,
                                 ae_file_destination,
                                 config,
                                 A_noB=False,
                                 B_noA=False,
                                 compound=True)

    interactions = AB_compound - A_noB - B_noA
    interactions_gamma_a = interactions + A_noB
    return interactions, interactions_gamma_a, A_noB


def interaction_estimator(results_file, checkpoint, trial):
    counter = 0
    # pairs the links.
    for indexi, i in enumerate(results_file['interactions'].index):
        for indexj, j in enumerate(results_file['interactions'].columns):

            if i != j:  # to avoid pairing a link with itself
                if pd.isna(results_file['interactions'][i][j]) is True:  # to avoid re-playing

                    if counter <= checkpoint:
                        print(f'======================== pair number:{counter} ========================\n')
                        pair = tuple((i, j))
                        results_file['interactions'][i][j], \
                        results_file['interactions_gamma_a'][i][j], \
                        results_file['gamma_a'][i][j] = \
                            compute_shapley_interaction(optimized_genome,
                                                        scores_table,
                                                        pair,
                                                        ae_file_destination,
                                                        config)
                        counter += 1

                    else:  # counter is bigger than checkpoint
                        print('======================== reached checkpoint, saving the files ========================\n')
                        counter = 0
                        return


def placeholder_maker(genome):
    empty_template = pd.DataFrame(index=genome.connections.keys(),
                                  columns=genome.connections.keys())
    empty_placeholder = {'interactions': copy.deepcopy(empty_template),
                         'interactions_gamma_a': copy.deepcopy(empty_template),
                         'gamma_a': copy.deepcopy(empty_template)}
    return empty_placeholder


def clockino():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


################################################################################
###################### script itself ###########################################

# The files
config_file = '/work/bay5607/project_interaction/AEconfig-SI.txt'

genome_file = '/work/bay5607/project_interaction/network_3_fitness_1300_.pkl'

ae_file = '/work/bay5607/project_interaction/SI-AE-model.pkl'

scores_table_file = '/work/bay5607/project_interaction/Interactions_scores_table_updated.pkl'


ae_file_destination = '/dev/shm/SI-AE-model_copy.pkl'
copyfile(ae_file, ae_file_destination)



with open(scores_table_file, 'rb') as f:
    scores_table = pickle.load(f)
with open(genome_file, 'rb') as f:
    optimized_genome = pickle.load(f)



config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,
                     neat.DefaultStagnation,
                     config_file)



# NEAT produces junk (disabled) connections, this line prunes them
pruner(optimized_genome)

# disable all the links so the algorithm can enable the intended ones
for link in list(optimized_genome.connections.keys()):
    optimized_genome.connections[link].enabled = False

trial = sys.argv[1]
#print(trial)

results_file = f'/work/bay5607/project_interaction/results_{trial}.pkl'
with open(results_file, 'rb') as f:
    results = pickle.load(f)


clockino()
interaction_estimator(results, checkpoint=510, trial=trial)
with open(results_file, 'wb') as f:
    pickle.dump(results, f, 1)
clockino()

#with open(scores_table_file, 'wb') as f:
#   pickle.dump(scores_table, f, 1)
