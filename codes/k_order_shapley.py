import pickle
import numpy as np
import pandas as pd
import itertools
import experiment_toolbox as exto
import game_play_toolbox as gpto
import neat
import os
import copy
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# The files
config_file = '/home/kayson/PycharmProjects/shapleyvalue/AEconfig-SI.txt'
genome_file = '/home/kayson/PycharmProjects/shapleyvalue/network_3_fitness_1300_.pkl'
ae_file = '/home/kayson/PycharmProjects/shapleyvalue/SI-AE-model.pkl'
intact_perf_file = '/home/kayson/ownCloud/shapleyvalue/clean_scripts/datasets/single_lesion_links_512.pkl'
nodal_shapley_file = '/home/kayson/ownCloud/shapleyvalue/clean_scripts/finalised material/datasets/shapley_nodes_1000.pkl'
single_node_file = '/home/kayson/ownCloud/shapleyvalue/clean_scripts/finalised material/datasets/single_lesion_nodes_512.pkl'
config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,
                     neat.DefaultStagnation,
                     config_file)
with open(nodal_shapley_file, 'rb') as f:
    nodal_shapley = pickle.load(f)
with open(single_node_file, 'rb') as f:
    single_node_lesion = pickle.load(f)
with open(intact_perf_file, 'rb') as f:
    single_site_performance = pickle.load(f)
intact_perf = single_site_performance['optimized_network']['intact'].mean()

single_node_lesion = single_node_lesion - intact_perf
# Genome is basically the NEAT-optimized network
with open(genome_file, 'rb') as f:
    optimized_genome = pickle.load(f)

# NEAT produces junk (disabled) connections, this line prunes them
exto.pruner(optimized_genome)
input_nodes, hidden_nodes, output_nodes, links \
    , _, _, _, _, _ = exto.network_structure(config, optimized_genome)
all_nodes = input_nodes + hidden_nodes + output_nodes
n_trials = 50
params = dict(environment_name='SpaceInvaders-v4',
              model_filename=ae_file,
              config=config,
              frames_per_state=2,
              nfeatures=6,
              noise_input=False)
n_k = range(1, 4)
k_order_scores = {}
k_order_shapleys = {}
for k in n_k:
    print(f'Order is: {k}')
    k_order_scores[k] = gpto.k_order_nodal_shapley(all_nodes,
                                                   optimized_genome,
                                                   links,
                                                   intact_perf,
                                                   n_trials=n_trials,
                                                   order=k,
                                                   **params)
    k_order_shapleys[k] = pd.DataFrame(columns=all_nodes)
    for node in all_nodes:
        temp = []
        for combination in k_order_scores[k]:
            if node in combination:
                temp.append(k_order_scores[k][combination])
        k_order_shapleys[k][node] = temp

with open(f'{n_k}_order_SV{n_trials}_scores_table.pkl', 'wb') as output:
    pickle.dump(k_order_scores, output, 1)
with open(f'{n_k}_order_SV{n_trials}_shapley_table.pkl', 'wb') as output:
    pickle.dump(k_order_shapleys, output, 1)


nodal_shapley_sorted = nodal_shapley.reindex(nodal_shapley.mean().sort_values().index, axis=1)
k_order_shapleys_sorted = {}
for k in k_order_shapleys.keys():
    k_order_shapleys_sorted[k] = k_order_shapleys[k].reindex(nodal_shapley_sorted.mean().sort_values().index, axis=1)


import matplotlib.cm as cm
cmap = cm.GnBu

fig = plt.figure(figsize=(8, 4), dpi=300)
sns.pointplot(data = nodal_shapley_sorted, color ='r', label='Original Shapley value')
for k in k_order_shapleys_sorted.keys():
    sns.pointplot(data = k_order_shapleys_sorted[k],color = cmap(k/3.),label=f'k = {k}')
plt.grid(color='k', linewidth=1, alpha = 0.15, axis= 'both')
plt.tight_layout()
plt.savefig('k_order_shapley_clean.pdf')

