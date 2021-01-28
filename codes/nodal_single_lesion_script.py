# -----------------------------------------------------------
# Importing libraries
# -----------------------------------------------------------

import pickle
import neat
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
# And the toolbox
import experiment_toolbox as exto
import game_play_toolbox as gpto

# -----------------------------------------------------------
# The files
# -----------------------------------------------------------

config_file = '/home/kayson/PycharmProjects/shapleyvalue/AEconfig-SI.txt'
genome_file = '/home/kayson/PycharmProjects/shapleyvalue/network_3_fitness_1300_.pkl'
ae_file = '/home/kayson/PycharmProjects/shapleyvalue/SI-AE-model.pkl'
config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,
                     neat.DefaultStagnation,
                     config_file)

scores_table_file = '/home/kayson/PycharmProjects/' \
                    'shapleyvalue/Interactions_scores_table_updated.pkl'

with open(scores_table_file, 'rb') as f:
    scores_table = pickle.load(f)
# Genome is basically the NEAT-optimized network
with open(genome_file, 'rb') as f:
    optimized_genome = pickle.load(f)

# NEAT produces junk (disabled) connections, this line prunes them
exto.pruner(optimized_genome)

input_nodes, hidden_nodes, output_nodes, links \
    , _, _, _, _, _ = exto.network_structure(config, optimized_genome)

all_nodes = input_nodes + hidden_nodes + output_nodes

for link in list(optimized_genome.connections.keys()):
    optimized_genome.connections[link].enabled = False

# -----------------------------------------------------------
# experiment itself
# -----------------------------------------------------------
n_trials = 512

for link in list(optimized_genome.connections.keys()):
    optimized_genome.connections[link].enabled = True
single_node_lesion = pd.DataFrame(columns=all_nodes,
                                  index=np.arange(n_trials))
params = dict(environment_name='SpaceInvaders-v4',
              model_filename=ae_file,
              config=config,
              frames_per_state=2,
              nfeatures=6,
              noise_input=False)

for node in single_node_lesion.columns:
    print(f'lesioning node: {node} -----------')
    translated_links = []
    genome = copy.deepcopy(optimized_genome)
    exto.node_to_link(links, node, translated_links)
    for link in translated_links:
        genome.connections[link].enabled = False
    for trial in single_node_lesion.index:
        print(f'trial number: {trial} -----------')
        performance = gpto.play_games(gpto.play_game,
                            n_games=16,
                            genome=genome,
                            n_jobs=-1, **params)
        single_node_lesion[node][trial] = np.mean(performance)

exto.preprocess_plot(single_node_lesion)
with open(f'single_node_lesion_{n_trials}.pkl', 'wb') as f:
    pickle.dump(single_node_lesion, f, 1)
