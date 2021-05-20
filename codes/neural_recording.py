# -----------------------------------------------------------
# importing and preparing things
# -----------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
import networkx as nx
from game_play_toolbox import *
import os
import pickle
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config_file = '/home/kayson/PycharmProjects/shapleyvalue/AEconfig-SI.txt'
genome_file = '/home/kayson/PycharmProjects/shapleyvalue/network_3_fitness_1300_.pkl'
ae_file = '/home/kayson/PycharmProjects/shapleyvalue/SI-AE-model.pkl'
with open('/home/kayson/PycharmProjects/shapleyvalue/clean_scripts/datasets/shapley_nodes_1000.pkl', 'rb') as f:
    node_shapley = pickle.load(f)
with open('/home/kayson/ownCloud/shapleyvalue/clean_scripts/finalised material/datasets/shapley_links_1000.pkl',
          'rb') as f:
    link_shapley = pickle.load(f)

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
intact_genome = copy.deepcopy(optimized_genome)
input_nodes, hidden_nodes, output_nodes, links \
    , _, _, _, _, _ = exto.network_structure(config, optimized_genome)

params = dict(environment_name='SpaceInvaders-v4',
              model_filename=ae_file,
              config=config)

all_nodes = input_nodes + hidden_nodes + output_nodes

# -----------------------------------------------------------
# Calculating the impact of single-lesions on the functional connectivity (nodes)
# -----------------------------------------------------------
n_trials = range(50)
node_wise = {}
node_wise['fc_impact'], node_wise['scores'] = nodal_functional_connectivity(all_nodes,
                                                                            links,
                                                                            intact_genome,
                                                                            n_trials,
                                                                            **params)

# -----------------------------------------------------------
# Correlating the scores, shapley values, and FC impact
# -----------------------------------------------------------
node_wise['r_FC-score'], node_wise['p_FC-score'] = stats.pearsonr(
    node_wise['fc_impact'].mean(), node_wise['scores'].mean())

node_wise['r_FC-shapley'], node_wise['p_FC-shapley'] = stats.pearsonr(
    node_wise['fc_impact'].mean(), node_shapley.mean())

node_wise['r_score-shapley'], node_wise['p_score-shapley'] = stats.pearsonr(
    node_wise['scores'].mean(), node_shapley.mean())

with open(f'node_FC.pkl', 'wb') as f:
    pickle.dump(node_wise, f, 1)

# -----------------------------------------------------------
# Calculating the impact of single-lesions on the functional connectivity (links)
# -----------------------------------------------------------
link_wise = {}
link_wise['fc_impact'], link_wise['scores'] = link_functional_connectivity(links,
                                                                           intact_genome,
                                                                           n_trials,
                                                                           **params)

# -----------------------------------------------------------
# Correlating the scores, shapley values, and FC impact
# -----------------------------------------------------------
link_wise['r_FC-score'], link_wise['p_FC-score'] = stats.pearsonr(
    link_wise['fc_impact'].mean(), link_wise['scores'].mean())

link_wise['r_FC-shapley'], link_wise['p_FC-shapley'] = stats.pearsonr(
    link_wise['fc_impact'].mean(), link_shapley.mean())

link_wise['r_score-shapley'], link_wise['p_score-shapley'] = stats.pearsonr(
    link_wise['scores'].mean(), link_shapley.mean())

with open(f'link_FC.pkl', 'wb') as f:
    pickle.dump(link_wise, f, 1)

# -----------------------------------------------------------
# Plotting the relationships.
# -----------------------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), constrained_layout=True,dpi=300)
sns.regplot(x=link_wise['fc_impact'].mean(),
            y=link_wise['scores'].mean(),
            n_boot=10_000,
            truncate=False,
            ci=95,
            color='k',
            ax=axes[0][0])
axes[0][0].set_xlabel('Impact of lesioning links on FC')
axes[0][0].set_ylabel('Score')

sns.regplot(x=link_wise['fc_impact'].mean(),
            y=link_shapley.mean(),
            n_boot=10_000,
            truncate=False,
            ci=95,
            color='k',
            ax=axes[0][1])
axes[0][1].set_xlabel('Impact of lesioning links on FC')
axes[0][1].set_ylabel('Shapley Value')

sns.regplot(x=node_wise['fc_impact'].mean(),
            y=node_wise['scores'].mean(),
            n_boot=10_000,
            truncate=False,
            ci=95,
            color='k',
            ax=axes[1][0])
axes[1][0].set_xlabel('Impact of lesioning nodes on FC')
axes[1][0].set_ylabel('Score')

sns.regplot(x=node_wise['fc_impact'].mean(),
            y=node_shapley.mean(),
            n_boot=10_000,
            truncate=False,
            ci=95,
            color='k',
            ax=axes[1][1])
axes[1][1].set_xlabel('Impact of lesioning nodes on FC')
axes[1][1].set_ylabel('Shapley Value')
plt.tight_layout(pad=5.0)
plt.savefig('FC_correlation.pdf')

# -----------------------------------------------------------
# Quickly checking what is happening during a single trial (lesioned vs intact)
# -----------------------------------------------------------
lesioned_genome = copy.deepcopy(optimized_genome)
another_lesioned_genome = copy.deepcopy(optimized_genome)

lesioned_genome.connections[(0, 0)].enabled = False
another_lesioned_genome.connections[(-1, 4)].enabled = False

# -----------------------------------------------------------
# Intact network
# -----------------------------------------------------------
intact_score, intact_neural_data, intact_time_of_deaths = neural_recorder(genome=optimized_genome,env_seed=1, **params)

# -----------------------------------------------------------
# Lesioning (0, 0)
# -----------------------------------------------------------
lesioned_score, lesioned_neural_data, lesioned_time_of_deaths = neural_recorder(
    genome=lesioned_genome,env_seed=1, **params)

# -----------------------------------------------------------
# Lesioning (-1, 4)
# -----------------------------------------------------------
another_lesioned_score, another_lesioned_neural_data, another_lesioned_time_of_deaths = neural_recorder(
    genome=another_lesioned_genome, env_seed=1, **params)

# -----------------------------------------------------------
# Finding the chosen action at each time point
# -----------------------------------------------------------
intact_action_mask = action_argmax_mask(intact_neural_data, input_nodes, hidden_nodes)
lesioned_action_mask = action_argmax_mask(lesioned_neural_data, input_nodes, hidden_nodes)
another_lesioned_action_mask = action_argmax_mask(another_lesioned_neural_data, input_nodes, hidden_nodes)


# -----------------------------------------------------------
# Plotting a trial
# -----------------------------------------------------------
_, axs = plt.subplots(3, 1, figsize=(16, 14))
sns.heatmap(ax=axs[0], data=intact_neural_data.T, cmap='RdBu_r',linewidths=0, center=0)
sns.heatmap(ax=axs[1], data=lesioned_neural_data.T, cmap='RdBu_r',linewidths=0, center=0)
sns.heatmap(ax=axs[2], data=another_lesioned_neural_data.T, cmap='RdBu_r',linewidths=0, center=0)

for death in intact_time_of_deaths:
    axs[0].axvline(death, color='r')
for death in lesioned_time_of_deaths:
    axs[1].axvline(death, color='r')
for death in another_lesioned_time_of_deaths:
    axs[2].axvline(death, color='r')

axs[0].axhline(12, color='k')
axs[0].axhline(18, color='k')
axs[1].axhline(12, color='k')
axs[1].axhline(18, color='k')
axs[2].axhline(12, color='k')
axs[2].axhline(18, color='k')
sns.heatmap(ax=axs[0], data=intact_action_mask.T, cmap='binary', cbar=False, alpha=0.3)
sns.heatmap(ax=axs[1], data=lesioned_action_mask.T, cmap='binary', cbar=False, alpha=0.3)
sns.heatmap(ax=axs[2], data=another_lesioned_action_mask.T, cmap='binary', cbar=False, alpha=0.3)

plt.ylabel('neurons')
plt.xlabel('time')
axs[0].set_title(f'Intact network with a score of {intact_score}')
axs[1].set_title(f'(0, 0) Lesioned with a score of {lesioned_score}')
axs[2].set_title(f'(-1, 4) Lesioned with a score of {another_lesioned_score}')

axs[0].set_xticks(range(0, (len(intact_neural_data) + 1), 20))
axs[0].set_xticklabels(range(0, (len(intact_neural_data) + 1), 20))

axs[1].set_xticks(range(0, (len(lesioned_neural_data) + 2), 20))
axs[1].set_xticklabels(range(0, (len(lesioned_neural_data) + 2), 20))

axs[2].set_xticks(range(0, (len(another_lesioned_neural_data) + 2), 20))
axs[2].set_xticklabels(range(0, (len(another_lesioned_neural_data) + 2), 20))

axs[0].tick_params('y', labelrotation=0)
axs[1].tick_params('y', labelrotation=0)
axs[2].tick_params('y', labelrotation=0)

plt.tight_layout()
plt.show()
plt.savefig('neural recording of intact vs lesioned.pdf')

# -----------------------------------------------------------
# Calculating + plotting FC impact of a single trial
# -----------------------------------------------------------
intact_fc = intact_neural_data.corr(method='pearson').fillna(0)
lesion_fc = lesioned_neural_data.corr(method='pearson').fillna(0)
diff_fc = intact_fc - lesion_fc
impact = np.abs(diff_fc.sum().sum())
cbar_kws = {'label': 'Pearson correlation', 'orientation': 'horizontal'}
cbar_kws_diff = {'label': 'Difference', 'orientation': 'horizontal'}

_, axs = plt.subplots(1, 3, figsize=(14, 6))
sns.heatmap(ax=axs[0], data=intact_fc, cmap='RdBu_r', center=0,
            linecolor='k', linewidths=0.2, square=True, cbar_kws=cbar_kws)
axs[0].tick_params('x', labelrotation=90)
axs[1].set_title('FC of the intact and the lesioned network')
sns.heatmap(ax=axs[1], data=lesion_fc, cmap='RdBu_r', center=0,
            linecolor='k', linewidths=0.2, square=True, cbar_kws=cbar_kws)
axs[1].tick_params('x', labelrotation=90)

sns.heatmap(ax=axs[2], data=diff_fc, cmap='RdBu_r', center=0,
            linecolor='k', linewidths=0.2, square=True, cbar_kws=cbar_kws_diff)
axs[2].tick_params('x', labelrotation=90)

plt.tight_layout()
plt.savefig('FC intact vs 0_0 lesioned.pdf')

# -----------------------------------------------------------
# Some exploratory Graph analyses
# -----------------------------------------------------------
intact_graph = nx.from_pandas_adjacency(intact_fc)
lesioned_graph = nx.from_pandas_adjacency(lesion_fc)
nx.global_efficiency(intact_graph)  # 0.70
nx.global_efficiency(lesioned_graph)  # 0.79

nx.local_efficiency(intact_graph)  # 0.84
nx.local_efficiency(lesioned_graph)  # 0.89

nx.average_clustering(intact_graph)  # 0.84
nx.average_clustering(lesioned_graph)  # 0.89

intact_neural_data.to_csv('intact_data',index = False)
lesioned_neural_data.to_csv('lesioned_data',index = False)