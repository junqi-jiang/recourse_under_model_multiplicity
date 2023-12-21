# Argumentative Ensembling

This repository contains codes to reproduce all results in the paper "Recourse under Model Multiplicity via Argumentative Ensembling". It has been accepted as a full paper in [AAMAS 2024 conference](https://www.aamas2024-conference.auckland.ac.nz/).

The Python libraries needed are straightforward in this project with Python version 3.7.13. The main packages are numpy, pandas, sklearn, carla, and clingo. 
To install carla (for loading preprocessed compas and heloc datasets) and clingo, use

    pip install git+https://github.com/carla-recourse/carla.git#egg=carla
    conda install -c potassco clingo

The code requires running clingo solvers to compute extensions for argumentation frameworks, this is manually executed between every:

    exp_x.eval()

and

    exp_x.eval_ours()

lines in the jupyter notebook. If on a Windows machine, this can be done by executing the commands in commands.txt. When executing commands, please make sure the current working directory is ./experiments of this folder.
