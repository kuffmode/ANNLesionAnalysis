import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import experiment_toolbox as exto
import HPC_placeholder_builder as plbl

genome_file = '/home/kayson/PycharmProjects/' \
              'shapleyvalue/network_3_fitness_1300_.pkl'
with open(genome_file, 'rb') as f:
    optimized_genome = pickle.load(f)
exto.pruner(optimized_genome)
#
# empty_template = pd.DataFrame(index=optimized_genome.connections.keys(),
#                               columns=optimized_genome.connections.keys())


dataset = plbl.placeholder_maker(optimized_genome)

counter = 0

for indexi, i in enumerate(dataset['interactions'].index):
    for indexj, j in enumerate(dataset['interactions'].columns):
        if i != j:
            if pd.isna(dataset['interactions'][i][j]) is True:
                temp_interactions = []
                temp_interactions_gamma_a = []
                temp_gamma_a = []
                for k in range(100):
                    with open(
                            f'/home/kayson/PycharmProjects/'
                            f'shapleyvalue/clean_scripts/'
                            f'results_files/Filled100/'
                            f'results_{k}.pkl',
                            'rb') as f:
                        results = pickle.load(f)
                        temp_interactions.append(results['interactions'][i][j])
                        temp_interactions_gamma_a.append(results['interactions_gamma_a'][i][j])
                        temp_gamma_a.append(results['gamma_a'][i][j])
                dataset['interactions'][i][j] = np.mean(temp_interactions)
                dataset['interactions_gamma_a'][i][j] = np.mean(temp_interactions_gamma_a)
                dataset['gamma_a'][i][j] = np.mean(temp_gamma_a)
                print(f'pair number: {counter}')
                counter += 1
for i in dataset:
    dataset[i].columns = [str(c) for c in dataset[i].columns]
    dataset[i].index = [str(c) for c in dataset[i].index]

threshold = np.std(dataset['interactions'].values)*2

dataset['thresholded_interactions'] = dataset['interactions'].where(
    (dataset['interactions'] < -threshold) | (dataset['interactions'] > threshold))
dataset['modulations'] = dataset['thresholded_interactions'].where(
    ((dataset['interactions_gamma_a'] > 0) & (dataset['gamma_a'] < 0)))

for i in dataset:
    dataset[i] = dataset[i].replace('NaN',0)

with open('interaction_dataset100.pkl', 'wb') as f:
    pickle.dump(dataset, f, 1)


with open('/home/kayson/PycharmProjects/shapleyvalue/clean_scripts/datasets/interactions_100.pkl', 'rb') as f:
    dataset = pickle.load(f)


import seaborn as sns

for i in dataset:
    plt.figure(figsize=(12, 12), dpi=300)
    sns.heatmap(dataset[f'{i}'],linewidths = 0.3,
                linecolor='k',
                cmap='Spectral_r',
                square = True,
                cbar_kws={"shrink": .5})
    plt.yticks(np.arange(51)+0.5,dataset[f'{i}'].columns)
    plt.xticks(np.arange(51)+0.5,dataset[f'{i}'].columns)
    plt.ylim(52,-1)
    plt.xlim(-1,52)
    plt.title(f'Pairwise {i} of all connections, Number of permutations = 100',fontweight='bold')
    plt.xlabel('connections (source, destination)',fontsize=10, fontweight='bold')
    plt.ylabel('connections (source, destination)',fontsize=10, fontweight='bold')
    plt.savefig(f'matrix_{i}_100.pdf')


import networkx as nx
from matplotlib import cm
import matplotlib.colors as mcolors


for i in dataset:
    dataset[i] = dataset[i].fillna(0)
G=nx.convert_matrix.from_pandas_adjacency(dataset['thresholded_interactions'])
nx.write_graphml(G,'thresholded_interactions_graph.graphml')

# nodes = G.nodes()
# degrees = G.degree()
# n_color = np.asarray([degrees[n] for n in nodes])
#
# edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
#
#
# fig, axes = plt.subplots(figsize=(15, 15), dpi=300)
# options = {
#     "font_size": 12,
#     "edgelist": edges,
#     "edge_color": weights,
#     "edge_cmap": cm.Spectral_r,
#     "node_color": n_color,
#     "cmap": cm.Spectral_r,
#     "node_size": n_color*800,
#     "edgecolors": "black",
#     "linewidths": 2,
#     "width": 2,
# }
# sc = nx.draw_circular(G,ax=axes,with_labels= True, **options)
# sc.set_norm(mcolors.LogNorm())
# fig.colorbar(sc)
# plt.show()

