# Multi-Instance Security Degradation of Code-Based KEMs

This repository contains the additional material of the paper "Multi-Instance Security Degradation of Code-Based KEMs" by Alexander May and Gabriel Sá Diogo.

- Cryptology ePrint Archive: <https://eprint.iacr.org/2026/517>
- SCN 2026: (to be added)

## Structure of This Repository

### ``appendix.pdf``

Contains details on the used DOOM algorithms.
In particular, it explains how we calculate the bit complexities of these algorithms.

Not necessary for the understanding of the paper.

### ``.py``-Scripts

These are the python scripts to produce the results of sections 3, 4 and 5.
More specifically:

- ``doom``: Functions that compute the (expected) minimal runtime of DS-DOOM and MMT-DOOM.
They also output the corresponding space complexity and the algorithm parameters to achieve this runtime.
(Note: Computations for MMT-DOOM can take a while to complete.)
- ``hqcdoom``, ``hqcCommonCode``, ``bikedoom``, ``mceliecedoom``: Compute the bit complexities of the 1-out-of-$M$ session/secret key recovery attacks.
- ``istarmap``: Patch for the ``multiprocessing`` module defining a ``starmap`` function that correctly works with the ``tqdm`` module to produce a nice progress bar.

### ``Figures/``

Contains all plots produced by the aforementioned scripts.
Always one plot for runtime complexity, space complexity and $\log_M$ speedups each.

Naming convention virtually the same as for the scripts, and subfolders for each parameter set.

### ``Out/``

Contains the outputs of the aforementioned scripts in structured tables.
Naming convention same as in ``Figures/``.
