# -----------------------------------------------------------
# Importing libraries
# -----------------------------------------------------------

import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
# And the toolbox
import experiment_toolbox as exto
import game_play_toolbox as gpto
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'GothamSSm'
# -----------------------------------------------------------
# preparing to plot everything (shapley values vs single lesions)
# -----------------------------------------------------------
with open('/home/kayson/PycharmProjects/shapleyvalue/clean_scripts/datasets/single_lesion_links_512.pkl', 'rb') as f:
    single_link_lesion = pickle.load(f)
with open('/home/kayson/PycharmProjects/shapleyvalue/clean_scripts/datasets/single_lesion_nodes_512.pkl', 'rb') as f:
    single_node_lesion = pickle.load(f)

with open('/home/kayson/PycharmProjects/shapleyvalue/clean_scripts/datasets/shapley_nodes_1000.pkl', 'rb') as f:
    node_shapley = pickle.load(f)
node_shapley = exto.preprocess_plot(node_shapley)
with open('/home/kayson/PycharmProjects/shapleyvalue/clean_scripts/datasets/shapley_links_1000.pkl', 'rb') as f:
    link_shapley = pickle.load(f)
link_shapley = exto.preprocess_plot(link_shapley)

# -----------------------------------------------------------
# we need these CI's to plot the vertical areas
# -----------------------------------------------------------
_, optimized_lower, optimized_upper = exto.mean_confidence_interval(
    single_link_lesion['optimized_network']['intact'], confidence=0.95)
_, noisy_lower, noisy_upper = exto.mean_confidence_interval(
    single_link_lesion['noisy_network']['intact'], confidence=0.95)
_, swapped_lower, swapped_upper = exto.mean_confidence_interval(
    single_link_lesion['weight_swapped_network']['intact'], confidence=0.95)
intact_location = single_link_lesion['optimized_network'].columns.get_loc('intact')

# -----------------------------------------------------------
# bootstrap hypothesis testing + bonferroni correction
# -----------------------------------------------------------
sig_link_lesions = exto.bootstrap_hyp_test(single_link_lesion['optimized_network'],
                                           p_value=0.05 / len(
                                               single_link_lesion['optimized_network'].columns),
                                           bootstrap_samples=10000,
                                           reference_mean=single_link_lesion['optimized_network']['intact'])
sig_node_lesions = exto.bootstrap_hyp_test(single_node_lesion,
                                           p_value=0.05 / len(single_node_lesion.columns),
                                           bootstrap_samples=10000,
                                           reference_mean=single_link_lesion['optimized_network']['intact'])

sig_node_shapley = exto.bootstrap_hyp_test(node_shapley,
                                           p_value=0.05 / len(node_shapley.columns),
                                           bootstrap_samples=10000,
                                           reference_mean=None)

sig_link_shapley = exto.bootstrap_hyp_test(link_shapley,
                                           p_value=0.05 / len(link_shapley.columns),
                                           bootstrap_samples=10000,
                                           reference_mean=None)

# -----------------------------------------------------------
# non parametric hypothesis testing + post-hoc
# -----------------------------------------------------------
random_agent_perf = gpto.random_agent(512)

sig_intact = stats.kruskal(single_link_lesion['optimized_network']['intact'],
                           single_link_lesion['weight_swapped_network']['intact'],
                           single_link_lesion['noisy_network']['intact'],
                           random_agent_perf)
# KruskalResult(statistic=577.5726772253214, pvalue=7.331284283898258e-125)

sig_intact_weight_sw = stats.mannwhitneyu(single_link_lesion['optimized_network']['intact'],
                                          single_link_lesion['weight_swapped_network']['intact'])
# MannwhitneyuResult(statistic=31157.0, pvalue=1.053846319605133e-99)

sig_intact_blind = stats.mannwhitneyu(single_link_lesion['optimized_network']['intact'],
                                      single_link_lesion['noisy_network']['intact'])
# MannwhitneyuResult(statistic=50919.5, pvalue=9.848849828142587e-65)

intact_performances = {
    'optimized': single_link_lesion['optimized_network']['intact'],
    'blind': single_link_lesion['noisy_network']['intact'],
    'weight_swapped': single_link_lesion['weight_swapped_network']['intact'],
    'random': random_agent_perf}
intact_performances = pd.DataFrame(intact_performances)

plt.figure(figsize=(8, 10), dpi=300)
sns.displot(data=intact_performances, kind='kde', rug=True)
plt.xlabel('Performance')
plt.savefig('intact_performances.svg')
sig_intact_rand = stats.mannwhitneyu(single_link_lesion['optimized_network']['intact'],
                                     random_agent_perf)
# MannwhitneyuResult(statistic=39542.0, pvalue=9.084810449045245e-84)
# -----------------------------------------------------------
# preparing labels (adding significance star)
# -----------------------------------------------------------
x_links = list(single_link_lesion['optimized_network'].columns)
for index, label in enumerate(x_links):
    if label in sig_link_lesions.columns:
        x_links[index] = f'*{label}'

x_nodes = list(single_node_lesion.columns)
for index, label in enumerate(x_nodes):
    if label in sig_node_lesions.columns:
        x_nodes[index] = f'*{label}'

x_nodes_sv = list(node_shapley.columns)
for index, label in enumerate(x_nodes_sv):
    if label in sig_node_shapley.columns:
        x_nodes_sv[index] = f'*{label}'

x_links_sv = list(link_shapley.columns)
for index, label in enumerate(x_links_sv):
    if label in sig_link_shapley.columns:
        x_links_sv[index] = f'*{label}'

# -----------------------------------------------------------
# the plot itself
# -----------------------------------------------------------
plt.style.use('bmh')
fig, axes = plt.subplots(nrows=2, ncols=2,
                         gridspec_kw={'width_ratios': [1, 4]},
                         figsize=(16, 8), dpi=300, sharey='row',
                         constrained_layout=True)
# -----------------------------------------------------------
sns.pointplot(data=single_node_lesion, ci=95, join=False, capsize=.4,
              n_boot=10_000, color='k', markers='_', errwidth=2, ax=axes[0][0])
sns.pointplot(data=single_link_lesion['optimized_network'], ci=95, join=False, capsize=.4,
              n_boot=10_000, color='k', markers='_', errwidth=2, ax=axes[0][1])
sns.pointplot(data=node_shapley, ci=95, join=False, capsize=.4,
              n_boot=10_000, color='k', markers='_', errwidth=2, ax=axes[1][0])
sns.pointplot(data=link_shapley, ci=95, join=False, capsize=.4,
              n_boot=10_000, color='k', markers='_', errwidth=2, ax=axes[1][1])
# -----------------------------------------------------------
axes[0][0].set_xticklabels(labels=x_nodes, rotation=90, fontsize=10, fontweight='bold')
# axes[0][0].set_yticklabels(labels = range(150,451,50) , fontsize=10, fontweight='bold')
axes[0][0].set_ylabel('Raw Performance', fontsize=10, fontweight='bold')
axes[0][0].grid(color='k', linewidth=1, alpha=0.15, axis='both')
axes[0][0].axhspan(optimized_lower, optimized_upper, color='#00B1DB', alpha=0.2)
axes[0][0].axhspan(swapped_lower, swapped_upper, color='#F0864C', alpha=0.2)
axes[0][0].axhspan(noisy_lower, noisy_upper, color='#E6B213', alpha=0.2)
# -----------------------------------------------------------
# axes[0][1].set_yticklabels(labels = range(150,451,50) , fontsize=10, fontweight='bold')
axes[0][1].axhspan(optimized_lower, optimized_upper, color='#00B1DB', alpha=0.2)
axes[0][1].axhspan(swapped_lower, swapped_upper, color='#F0864C', alpha=0.2)
axes[0][1].axhspan(noisy_lower, noisy_upper, color='#E6B213', alpha=0.2)
axes[0][1].set_xticklabels(labels=x_links, rotation=90, fontsize=10, fontweight='bold')
axes[0][1].axvline(intact_location, color='k', linewidth=2, alpha=0.2)
axes[0][1].grid(color='k', linewidth=1, alpha=0.15, axis='both')
# -----------------------------------------------------------
axes[1][0].set_xticklabels(labels=x_nodes_sv, rotation=90, fontsize=10, fontweight='bold')
axes[1][0].grid(color='k', linewidth=1, alpha=0.15, axis='both')
axes[1][0].axhline(0, color='r', linewidth=2, alpha=0.2)
axes[1][0].set_ylabel('Shapley Value', fontsize=10, fontweight='bold')
axes[1][0].set_xlabel('Nodes', fontsize=10, fontweight='bold')
# -----------------------------------------------------------
axes[1][1].set_xticklabels(labels=x_links_sv, rotation=90, fontsize=10, fontweight='bold')
axes[1][1].axhline(0, color='r', linewidth=2, alpha=0.2)
axes[1][1].grid(color='k', linewidth=1, alpha=0.15, axis='both')
axes[1][1].set_xlabel('Links (source → destination)', fontsize=10, fontweight='bold')
# -----------------------------------------------------------
plt.savefig('single_lesion_vs_shapley.pdf')

# -----------------------------------------------------------
# a quick scatterplot of single lesions vs shapley values
# -----------------------------------------------------------
x = {}
y = {}
fig, axes = plt.subplots(nrows=1, ncols=3,
                         figsize=(9, 3), dpi=300,
                         constrained_layout=True)
for index, i in enumerate([-4, 0, -10]):
    x[f'{i}'] = np.array(single_node_lesion[f'{i}'], dtype=np.float)
    y[f'{i}'] = np.array(node_shapley[f'{i}'][:512], dtype=np.float)

    sns.scatterplot(x=x[f'{i}'], y=y[f'{i}'], color='k', markers='.', ax=axes[index], s=12)
    axes[index].set_xlabel('Single-lesion Analysis')
    axes[index].set_ylabel('Shapley Value')
    axes[index].set_title(f'Node ({i})')
    axes[index].grid(color='k', linewidth=0.5, alpha=0.15, axis='both')

plt.savefig('scatterplots_single_lesion_vs_shapley.pdf')
