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

shapley_table = {}  # unsorted pair of shapley values and elements
scores_table = {}   # multi-perturbation dataset, load if one already exists
n_samples = 128

# Calculating shapley values, depending on how many jobs you can run in parallel
# and how many trials (n_samples) you specified, this part might take days.
# 15 jobs, 1000 samples = around 1 week, 100 samples = a day

gpto.compute_shapley(optimized_genome,
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
shapley_sorted.to_csv(f'SV{n_samples}.csv', index=False)

with open(f'SV{n_samples}.pkl', 'wb') as output:
    pickle.dump(shapley_sorted, output, 1)

with open(f'SV{n_samples}_scores_table.pkl', 'wb') as output:
    pickle.dump(scores_table, output, 1)

# Testing each distribution against a zero-mean one using a bootstrap hypothesis testing process
# P-value is corrected for multiple comparison using Bonferroni correction method: p-value/n_tests
significant_shapley_values = exto.bootstrap_hyp_test(shapley_sorted,
                                                     p_value=0.05/len(shapley_sorted.columns),
                                                     bootstrap_samples=10000,
                                                     reference_mean=None)

# plotting the results
x_ticks = list(shapley_sorted.columns)
for index, label in enumerate(x_ticks):
    if label in significant_shapley_values.columns:
        x_ticks[index] = f'*{label}'

plt.figure(figsize=(16, 6), dpi=300)
sns.pointplot(data=shapley_sorted, ci=95, join=False, capsize=.3,
              n_boot=10000, color='k', errwidth=3)

plt.grid(color='k', linewidth=1, alpha = 0.15, axis= 'both')
plt.axhline(0, color='r', linewidth=2, alpha = 0.3)
plt.xlabel('connections (source → destination)', fontsize=14, fontweight='bold')
plt.xticks(np.arange(len(x_ticks)), labels=x_ticks, rotation=90, fontsize=12)
plt.yticks(range(-120, 141, 20), fontsize=12, fontweight='bold')
plt.ylabel('Shapley Values', fontsize=14, fontweight='bold')
plt.title(f'Estimated Shapley Value of all links, CI = %95, Number of permutations = {n_samples}',
          fontsize=16, fontweight='bold')
plt.savefig(f'SV{n_samples}.pdf', bbox_inches='tight')

