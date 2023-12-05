# ANNLesionAnalysis

If you are here to use MSA, please visit our Python library instead of this repository:
https://github.com/kuffmode/msa
-
Codes and datasets that used or produced during the research project: **Systematic Lesion Analysis of an Artificial Neural Network: A Step Towards Quantifying Causation in the Brain.**

Firstly, I'm not a good coder yet so I appreciate any feedback that can make the codes more efficient. When I started this project I wasn't aware of many Python/programming conventions and I learned a lot of them after I was almost done with this project so I already know a few things that I need to work on later:

    1. Docstring
    2. Type hints
    3. Better dependency management
    4. Better project structure
    5. Small adjustments here and there to make the code easier to read and faster.

I have the functions in two toolboxes ([game_play_toolbox.py](https://github.com/kuffmode/ANNLesionAnalysis/blob/main/codes/game_play_toolbox.py) and [experiment_toolbox.py](https://github.com/kuffmode/ANNLesionAnalysis/blob/main/codes/experiment_toolbox.py)) I tried to comment things as much as I could but let me know if things are not clear. Generally, game_play contains the game itself and Shapley related functions while the experiment_toolbox is more about single-site lesion analysis.

The experiments themselves are the files ending with *script.py*, for example [experiment_script.py](https://github.com/kuffmode/ANNLesionAnalysis/blob/main/codes/experiment_script.py) I still need to figure out a nice and proper way to paths so for now unfortunately you might need to tinker those yourself before running the scripts. Will fix it as soon as I learn how to do it!

There are a few HPC related codes that calculates PCIA on the HPC by distributing elements across n nodes of a high-performance cluster. Our cluster didn't support direct jupyter access or DASK Parallel jobs so I had to do things manually by creating jobs and submitting them all using tiny scripts. These codes are the least reusable ones since they are made for our HPC system.
