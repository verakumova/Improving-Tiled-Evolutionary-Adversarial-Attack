# Improving Tiled Evolutionary Adversarial Attack


## Overview
This is the repository for the paper "Improving Tiled Evolutionary Adversarial Attack".

## Description of Files

### Evolutionary Attack
The evolutionary attack used in the paper is described by the `EvolutionaryAttack` class in the `evo_attack.py` file.

### Detection and Utils Functions
The `detect.py` file contains functions for detecting statistics that we use for testing the ability of an adversarial image to evade the chosen defense.

The `utils.py` file contains other auxiliary functions.

Both of these files are adopted from the authors original implementation: https://github.com/s-huu/TurningWeaknessIntoStrength.

### Image Classes
The `image.py` file contains classes for adversarial and benign images.

### Transferability
The `transferability.py` is used for testing the transferability between models.
