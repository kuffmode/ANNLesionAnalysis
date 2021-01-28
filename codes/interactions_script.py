# Importing libraries
import pickle
import neat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# And the toolbox
import experiment_toolbox as exto
import game_play_toolbox as gpto

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

# disable all the links so the algorithm can enable the intended ones
for link in list(optimized_genome.connections.keys()):
    optimized_genome.connections[link].enabled = False

with open('/home/kayson/PycharmProjects/shapleyvalue/Interactions_scores_table.pkl', 'rb') as f:
    scores_table = pickle.load(f)

n_samples = 64
candidates = [(-4, 0), (0, 0), (-1, 4)]


def interaction_estimator(candidate, n_samples):

    pair = []
    interactions = {}
    interactions_gamma_a = {}
    gamma_a = {}

    # pairs the candidate with all other links
    for link in optimized_genome.connections.keys():
        if candidate != link:
            pair.append(tuple((candidate, link)))

    for pair_index, linklist in enumerate(pair):
        print(pair_index, linklist)
        interactions[linklist], \
        interactions_gamma_a[linklist], \
        gamma_a[linklist] = \
            gpto.compute_shapley_interaction(optimized_genome,
                                             scores_table,
                                             linklist,
                                             ae_file,
                                             config,
                                             n_samples)

        # in case you need to save the results every n step, uncomment this
        # if pair_index % 10 == 0:
        #     interactions_checkpoint = pd.DataFrame.from_dict(interactions)
        #     interactions_gamma_a_checkpoint = pd.DataFrame.from_dict(interactions_gamma_a)
        #     gamma_a_checkpoint = pd.DataFrame.from_dict(gamma_a)
        #
        #     interactions_checkpoint.to_csv(
        #         f'checkpoint_interactions_{candidate}_{pair_index}.csv',
        #         index=False)
        #     interactions_gamma_a_checkpoint.to_csv(
        #         f'checkpoint_interactions_gamma_a_{candidate}_{pair_index}.csv',
        #         index=False)
        #     gamma_a_checkpoint.to_csv(
        #         f'checkpoint_gamma_a_{candidate}_{pair_index}.csv',
        #         index=False)

    interactions = exto.preprocess_plot(interactions)
    interactions_gamma_a = exto.preprocess_plot(interactions_gamma_a, interactions)
    gamma_a = exto.preprocess_plot(gamma_a, interactions)

    interactions.to_csv(f'interactions_{candidate}.csv', index=False)
    interactions_gamma_a.to_csv(f'interactions_gamma_a_{candidate}.csv', index=False)
    gamma_a.to_csv(f'gamma_a_{candidate}.csv', index=False)

    return interactions, interactions_gamma_a, gamma_a


results = {}
for candidate in candidates:
    interactions, interactions_gamma_a, gamma_a = interaction_estimator(candidate, n_samples)
    results[candidate] = {'interaction': interactions,
                          'interactions_gamma_a': interactions_gamma_a,
                          'gamma_a': gamma_a}

with open('/home/kayson/PycharmProjects/shapleyvalue/Interactions_scores_table_updated.pkl', 'wb') as f:
    pickle.dump(scores_table, f, 1)


plt.figure(figsize=(16, 6), dpi=300)
sns.pointplot(data=results[(0, 0)]['interaction'], ci=95, join=False, capsize=.3,
              n_boot=1000, color='k', errwidth=3)
sns.pointplot(data=results[(0, 0)]['interactions_gamma_a'], ci=95, join=False, capsize=.3,
              n_boot=1000, color='r', errwidth=3)
sns.pointplot(data=results[(0, 0)]['gamma_a'], ci=95, join=False, capsize=.3,
              n_boot=1000, color='c', errwidth=3)
plt.grid(color='k', linewidth=1, alpha = 0.15, axis= 'both')
plt.axhline(0, color='r', linewidth=2, alpha = 0.3)
plt.xlabel('connections (source → destination)', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12, fontweight='bold')
plt.ylabel('Shapley Values', fontsize=14, fontweight='bold')
plt.title(f'Estimated Shapley Value of all links, CI = %95, Number of permutations = {n_samples}',
          fontsize=16, fontweight='bold')
plt.savefig(f'inttest.pdf', bbox_inches='tight')
