# Importing libraries
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

# The files
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


n_trials = 1000

nodal_shapleys = gpto.compute_nodal_shapley(optimized_genome,
                                            all_nodes, links,
                                            n_trials, scores_table,
                                            ae_file, config)


with open('nodal_shapley_table_1000.pkl', 'wb') as f:
    pickle.dump(nodal_shapleys, f, 1)

with open('nodal_scores_table.pkl', 'wb') as f:
    pickle.dump(scores_table, f, 1)

nodal_shapleys = exto.preprocess_plot(nodal_shapleys)

significant_nodes = exto.bootstrap_hyp_test(nodal_shapleys,
                                            p_value=0.05/len(nodal_shapleys.columns),
                                            bootstrap_samples=10000,
                                            reference_mean=None)

# plotting the results
x_ticks = list(nodal_shapleys.columns)
for index, label in enumerate(x_ticks):
    if label in significant_nodes.columns:
        x_ticks[index] = f'*{label}'



plt.figure(figsize=(10, 6), dpi=300)

sns.pointplot(data=nodal_shapleys, ci=95, join=False, capsize=.3,
              n_boot=10000, color='k', errwidth=3)

plt.grid(color='k', linewidth=1, alpha = 0.15, axis= 'both')

plt.xlabel('Nodes',fontsize=14, fontweight='bold')
plt.xticks(np.arange(len(x_ticks)), labels=x_ticks, rotation=90, fontsize=12)
plt.ylabel('Shapley Value', fontsize=14, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.axhline(0, color='r', linewidth = 2, alpha = 0.2)
plt.title(f'Shapley values of the nodes, CI = %95, Number of trials = {n_trials}',
          fontsize=16, fontweight='bold')
plt.savefig('nodal_shapley_values_1000_bonfcorr.pdf')


# ---------------------------------------------------
for link in list(optimized_genome.connections.keys()):
    optimized_genome.connections[link].enabled = True
single_node_lesion = pd.DataFrame(columns=all_nodes,
                                  index=np.arange(512))
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

with open(f'single_node_{n_trials}.pkl', 'wb') as f:
    pickle.dump(single_node_lesion, f, 1)


exto.preprocess_plot(single_node_lesion)
single_node_lesion = single_node_lesion.reindex(single_node_lesion.mean().sort_values().index, axis=1)

plt.figure(figsize=(10, 6), dpi=300)

sns.pointplot(data=single_node_lesion, ci=95, join=False, capsize=.3,
              n_boot=10000, color='k', errwidth=3)

plt.grid(color='k', linewidth=1, alpha = 0.15, axis= 'both')

plt.xlabel('Nodes',fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=12)
plt.ylabel('Performance', fontsize=14, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
#plt.axhline(0, color='r', linewidth = 2, alpha = 0.2)
plt.title(f'Single lesion analysis of each node, CI = %95, Number of trials = {512}',
          fontsize=16, fontweight='bold')

plt.savefig('nodal_single_lesion.pdf')




