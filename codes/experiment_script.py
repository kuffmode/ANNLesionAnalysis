# Importing libraries
import pickle
import neat
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
# And the toolbox
import experiment_toolbox as exto
import game_play_toolbox as gpto

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# The files
config_file = '/home/kayson/PycharmProjects/shapleyvalue/AEconfig-SI.txt'
genome_file = '/home/kayson/PycharmProjects/shapleyvalue/network_3_fitness_1300_.pkl'
ae_file = '/home/kayson/PycharmProjects/shapleyvalue/SI-AE-model.pkl'
config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,
                     neat.DefaultStagnation,
                     config_file)

# Genome is basically the NEAT-optimized network
with open(genome_file, 'rb') as f:
    optimized_genome = pickle.load(f)

# NEAT produces junk (disabled) connections, this line prunes them
exto.pruner(optimized_genome)

# Extracting information from the network
network_info = {}
network_info['input_nodes'], \
network_info['hidden_nodes'], \
network_info['output_nodes'], \
network_info['connections'], \
network_info['weights'], \
network_info['edges'], \
network_info['activation_functions'], \
network_info['aggregation_functions'], \
network_info['bias_list'] = exto.network_structure(config, optimized_genome)

# Swapping the weights
weight_swapped_genome = exto.weight_swapper(optimized_genome, network_info['weights'])
what_swapped = exto.weight_tracker(optimized_genome, weight_swapped_genome)

# Randomizing connectivity
topology_shuffled_genome = exto.connectivity_randomizer(optimized_genome, network_info, n_self_loops=4)

# Binarize the weights
binary_weights_genome = exto.weight_binarizer(optimized_genome)

# Getting the performances
# (this is RAM hungry depending on the n_games, blocks are there to chunk the procedure)
# go for a maximum of 250 games per round with 64GB of RAM (trials < 250)
params = dict(environment_name='SpaceInvaders-v4',
              model_filename=ae_file,
              config=config,
              frames_per_state=2,
              nfeatures=6,
              noise_input=False)

performances = {}
blocks = 8
n_games = 64
subjects = {'optimized_network': optimized_genome,
            'binary_weights_network': binary_weights_genome,
          # 'connectivity_randomized_network': topology_shuffled_genome, # the performance is negligible
            'weight_swapped_network': weight_swapped_genome,
            'noisy_network': optimized_genome}

for subject in subjects:
    performances[subject] = []
    print(f'testing subject {subject}')

    for block in range(blocks):
        print(f'Playing Block number: {block}, {blocks - block} left')

        if subject is 'noisy_network':
            params['noise_input'] = True
        else:
            params['noise_input'] = False

        if len(performances[subject]) == 0:
            performances[subject] = exto.single_lesion(subjects[subject], n_games=n_games, **params)
        else:
            block_performance = exto.single_lesion(subjects[subject], n_games=n_games, **params)
            for connection in performances[subject].keys():
                performances[subject][connection].extend(block_performance[connection])
            del block_performance
    if subject is 'optimized_network':  # or 'connectivity_randomized_network':
        performances[subject] = exto.preprocess_plot(performances[subject])
    else:
        performances[subject] = exto.preprocess_plot(performances[subject],
                                                     performances['optimized_network'])

performances['random_agent'] = gpto.random_agent(blocks * n_games)

with open('single_lesion_performances.pkl', 'wb') as output:
    pickle.dump(performances, output, 1)


with open('/home/kayson/PycharmProjects/shapleyvalue/clean_scripts/single_lesion_performances.pkl', 'rb') as f:
    performances = pickle.load(f)

color_codes = ['#00B1DB',  # light blue: optimized
               '#0729F5',  # dark blue: binary weights
             # '#F53A61',  # red: connectivity randomized
               '#F0864C',  # orange: weight swapped
               '#E6B213']  # yellow: noise as input
intact_location = performances['optimized_network'].columns.get_loc('intact')

significant_lesions = exto.bootstrap_hyp_test(performances['optimized_network'],
                                              p_value=0.05/len(performances['optimized_network'].columns),
                                              bootstrap_samples=10000,
                                              reference_mean=performances['optimized_network']['intact'])

# plotting the results
x_ticks = list(performances['optimized_network'].columns)
for index, label in enumerate(x_ticks):
    if label in significant_lesions.columns:
        x_ticks[index] = f'*{label}'

_, optimized_lower, optimized_upper = exto.mean_confidence_interval(
    performances['optimized_network']['intact'], confidence=0.95)

_, noisy_lower, noisy_upper = exto.mean_confidence_interval(
    performances['noisy_network']['intact'], confidence=0.95)

_, swapped_lower, swapped_upper = exto.mean_confidence_interval(
    performances['weight_swapped_network']['intact'], confidence=0.95)

_, random_lower, random_upper = exto.mean_confidence_interval(
    performances['random_agent'], confidence=0.95)

# plt.figure(figsize=(16, 6), dpi=300)
#
# for index, data in enumerate(performances):
#     if data is not 'random_agent':
#         sns.pointplot(data=performances[data], ci=95, join=False, capsize=.3,
#                       n_boot=10000, color=color_codes[index], errwidth=3)
#
# plt.grid(color='k', linewidth=1, alpha = 0.15, axis= 'both')
# plt.axhspan(optimized_lower, optimized_upper, color = '#00B1DB', alpha = 0.2)
# plt.axhspan(swapped_lower, swapped_upper, color = '#F0864C', alpha = 0.2)
# plt.axhspan(noisy_lower, noisy_upper, color = '#E6B213', alpha = 0.2)
#
# plt.xlabel('connections (source → destination)',fontsize=14, fontweight='bold')
# plt.xticks(np.arange(len(x_ticks)), labels=x_ticks, rotation=90, fontsize=12)
# plt.ylabel('score', fontsize=14, fontweight='bold')
# plt.yticks(range(0, 451, 50),fontsize=12, fontweight='bold')
# plt.axvline(intact_location, color='k', linestyle=':')
# plt.title(f'Single-lesion analysis of the optimized and null-networks, CI = %95, Number of trials = {blocks * n_games}',
#           fontsize=16, fontweight='bold')
# plt.savefig('single_lesion_performances.pdf', bbox_inches='tight')


# without binary weights to have a cleaner plot

color_codes = ['#00B1DB',  # light blue: optimized
               '#F0864C',  # orange: weight swapped
               '#E6B213']  # yellow: noise as input


without_binary = copy.deepcopy(performances)
del without_binary['binary_weights_network']
del without_binary['random_agent']

plt.figure(figsize=(16, 6), dpi=300)

for index, data in enumerate(without_binary):
    sns.pointplot(data=performances[data], ci=95, join=False, capsize=.3,
                  n_boot=10000, color=color_codes[index], errwidth=3)

plt.grid(color='k', linewidth=1, alpha = 0.15, axis= 'both')
plt.axhspan(optimized_lower, optimized_upper, color = '#00B1DB', alpha = 0.2)
plt.axhspan(swapped_lower, swapped_upper, color = '#F0864C', alpha = 0.2)
plt.axhspan(noisy_lower, noisy_upper, color = '#E6B213', alpha = 0.2)

plt.xlabel('connections (source → destination)',fontsize=14, fontweight='bold')
plt.xticks(np.arange(len(x_ticks)), labels=x_ticks, rotation=90, fontsize=12)
plt.ylabel('score', fontsize=14, fontweight='bold')
plt.yticks(range(0, 451, 50),fontsize=12, fontweight='bold')
plt.axvline(intact_location, color='k', linewidth = 2, alpha = 0.2)
plt.title(f'Single-lesion analysis of the optimized and null-networks, CI = %95, Number of trials = {blocks * n_games}',
          fontsize=16, fontweight='bold')
plt.savefig('single_lesion_performances_clean.pdf', bbox_inches='tight')
