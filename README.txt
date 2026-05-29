RSMI Improved Operators
=======================

Code companion to the paper

  "Improving CFT Operators Using Machine Learning"
  Lior Oppenheim, Snir Gazit, Zohar Ringel
  arXiv:2605.28929, https://arxiv.org/abs/2605.28929

Trains symmetry-projected neural lattice operators with the RSMI-NE
algorithm and uses them to extract scaling dimensions of primary
operators in 2D critical lattice models (Ising and the q=3 Potts model).

Setup
-----
  pip install -r requirements.txt

  The pinned versions in requirements.txt match those used for the paper
  (Python 3.9, TensorFlow 2.12). On Linux, TensorFlow 2.12 uses the GPU
  automatically given a matching system CUDA 11.8 / cuDNN 8.6 install.

Project layout
--------------
  RSMI/             Core package: training (RSMI-NE), V/E geometry, symmetry
                    projections, connected-correlator estimation, plotting.
  mcmc/             Monte Carlo samplers (Wolff cluster updates) for the
                    Ising and q=3 Potts models in arbitrary dimension.
  RSMI_Ising.ipynb  End-to-end workflow for the 2D Ising model.
  RSMI_Potts.ipynb  End-to-end workflow for the 2D q=3 Potts model.

Step 1 -- Generate Monte Carlo samples
--------------------------------------
The training notebooks consume TFRecord shards produced by the `mcmc/`
samplers. Run, e.g., for the 2D Ising model at L=40:

  python -m mcmc.ising  --L 40 --num-thermal 10000 --num-cluster-moves 10 \
                        --num-measure 10000000 --output-dir Data

and for the 2D q=3 Potts model:

  python -m mcmc.potts  --L 40 --num-thermal 10000 --num-cluster-moves 10 \
                        --num-measure 10000000 --output-dir Data

Both scripts use Numba-jitted Wolff-style cluster updates and write
int8 TFRecord shards. Output layout (the notebooks expect this exact
form):

  <output-dir>/<model>/L{L}/
      config.json
      0.tfrecord(.gz)
      1.tfrecord(.gz)
      ...

`config.json` records L, dim, J (defaults to the critical coupling),
records-per-file, compression, etc. Pass `--gzip` for GZIP-compressed
shards. `--dim 3` switches to a cubic lattice (still single-J).

Step 2 -- Train an RSMI-NE neural operator
------------------------------------------
Open `RSMI_Ising.ipynb` or `RSMI_Potts.ipynb` and run all cells. Each
notebook:

  1. Loads TFRecord samples for a single linear size L via
     `build_tfrecord_dataset` (split into train/test by file).
  2. Extracts circular visible (V) and environment-shell (E) regions
     via `samples_to_ve_dataset`.
  3. Builds a `CoarseGrainer` (a fully-connected network into a single
     scalar, followed by BatchNorm with a hard MaxNorm constraint on
     gamma and a Relaxed-Bernoulli stochastic head) and a
     `SeparableCritic`, and trains them jointly to maximize the InfoNCE
     lower bound on I(H;E).
  4. Wraps the trained encoder in a symmetry projection
     (Z_2 x D_4 for Ising, Z_3 x D_4 for Potts) to obtain the final
     neural operator in the chosen symmetry sector.

Trained models and TensorBoard logs are written to `logs/<timestamp>/`.

Step 3 -- Measure scaling dimensions
------------------------------------
The scaling-dimension measurement uses Sandvik's L vs. 2L finite-size
scaling, so it needs samples at several linear sizes L (each paired
with its double 2L), not just the single L used for training. Generate
these additional datasets with the same samplers before running this
step, e.g. for Ising:

  for L in 12 24 48; do \
    python -m mcmc.ising --L $L --num-thermal 10000 --num-cluster-moves 10 \
                         --num-measure 10000000 --output-dir Data; \
  done

and analogously with `python -m mcmc.potts` for the Potts model. This
populates `Data/<model>/L{L}/` for each size. Increase number of 
measurements to improve accuracy.

After training, the second half of each notebook loads the saved
encoder and runs `run_connected_correlation` over these linear sizes L,
then calls `scaling_dimensions_from_correlations` to extract scaling
dimensions from each L, 2L pair.

Data, logs, and ignored artifacts
---------------------------------
By default the notebooks read from `Data/` and write training logs to
`logs/`. Both are listed in `.gitignore`.

Citing
------
If you use this code, please cite the accompanying paper:

  L. Oppenheim, S. Gazit, and Z. Ringel,
  "Improving CFT Operators Using Machine Learning",
  arXiv:2605.28929. https://arxiv.org/abs/2605.28929
