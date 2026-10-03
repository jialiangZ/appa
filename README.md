# 🦬 Appa: Bending Weather Dynamics with Latent Diffusion Models for Global Data Assimilation

[[paper](https://arxiv.org/abs/2504.18720)] [[website](https://montefiore-sail.github.io/appa)] [[data & models](https://huggingface.co/datasets/montefiore-sail/appa)] [[live forecasts](https://appa.montefiore.uliege.be/)]

<p align="center">
        <img src="./.github/banner.webp" alt="Three forecasts sampled from Appa against ground truth. Visualization of specific humidity at 850hPa."/>
</p>

This repository contains the code and data for the paper *Appa: Bending Weather Dynamics with Latent Diffusion Models for Global Data Assimilation* by Gérôme Andry, Sacha Lewin, François Rozet, Omer Rochman, Victor Mangeleer, Matthias Pirlet, Elise Faulx, Marilaure Grégoire, and Gilles Louppe. The paper was published in 2025 in the *Machine Learning and the Physical Sciences Workshop* at NeurIPS.

> Deep learning has advanced weather forecasting, but accurate prediction requires identifying the current state of the atmosphere from observational data. We introduce Appa, a score-based data assimilation model generating global atmospheric trajectories at 0.25° resolution and 1-hour intervals. Powered by a 565M-parameter latent diffusion model trained on ERA5, Appa can be conditioned on arbitrary observations to infer posterior distributions of plausible states without retraining. Our probabilistic framework handles reanalysis, filtering, and forecasting, within a single model, producing physically consistent reconstructions from various inputs. Results establish latent score-based data assimilation as a promising foundation for future global atmospheric modeling systems.

### Code

Most of our code should be well-documented. The repository follows a `src/` layout with code and artifacts strictly separated:

```
├── src/                  # all source code
│   ├── appa/             # main module, organized by domain
│   │   ├── config/       #   Hydra helpers and data-path registry (switch resolution with APPA_RESOLUTION)
│   │   ├── data/         #   ERA5 datasets, loaders, transforms, and download tools
│   │   ├── models/       #   Diffusion process, samplers, losses, and neural architectures (models/nn)
│   │   ├── assimilation/ #   Observation operators for data assimilation
│   │   ├── training/     #   Lightning modules, datamodules, callbacks, loggers, and optimizers (SOAP, etc.)
│   │   ├── diagnostics/  #   Plotting and spectral/reconstruction diagnostics
│   │   └── utils/        #   Dates, grids, math, and I/O helpers
│   ├── experiments/      # experiment scripts (importable package; one folder per domain)
│   └── scripts/          # data download/processing scripts and plotting scripts (plot_*.py)
├── tests/                # unit tests (test_<domain>_<module>.py)
├── tools/                # one-off debugging and verification scripts
├── artifacts/            # gitignored outputs and data (autoencoders/, data/)
└── run_*.py              # local, no-Slurm launchers (torchrun entry points)
```

### Usage

All the steps to run your own model or reproduce our results are detailed in the [Wiki](https://github.com/montefiore-sail/appa/wiki). Feel free to contact us through the issues or by mail if you have any further question.

Quick start with the provided Makefile:

```bash
make sync        # recreate the environment exactly from uv.lock
make test        # run the test suite
make train-ae    # train the autoencoder (GPUS=1 ARGS="train.update_steps=2000")
make train-den   # train the denoiser (ARGS="ae_run=<id>/0 latent_dump=default")
```

Equivalently, launchers can be invoked directly from any directory (training is
powered by [PyTorch Lightning](https://lightning.ai); DDP processes are spawned
automatically, so no `torchrun` is needed — set `APPA_DEVICES` for multi-GPU):

```bash
python -m appa.launchers ae-train train.update_steps=2000
```

Experiments are tracked with [SwanLab](https://swanlab.cn) or
[Weights & Biases], selected through the `tracker` section of the training
configuration (`tracker.name`: `swanlab`, `wandb`, or `none`). Use
`SWANLAB_MODE=disabled` / `WANDB_MODE=disabled` for untracked local runs.

Training runs are seeded by default (`train.seed=42`); set `train.seed=null` to disable. To switch the active dataset resolution (240x121 ↔ 1440x721), set `APPA_RESOLUTION` in the environment.

### Data

Our work is based on the ERA5 reanalysis dataset, which is provided by Google through [WeatherBench2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html). We provide scripts to download and process the data in the `scripts/data` folder. A latent dump of ERA5 encoded by our autoencoder is available on [HuggingFace](https://huggingface.co/datasets/montefiore-sail/appa). This HuggingFace repository also contains our trained models and ERA5 statistics.

### Citation

If you find this work useful in your research, please consider citing:
```bibtex
@article{andry2025appa,
  title={Appa: Bending Weather Dynamics with Latent Diffusion Models for Global Data Assimilation},
  author={Gérôme Andry and Sacha Lewin and François Rozet and Omer Rochman and Victor Mangeleer and Matthias Pirlet and Elise Faulx and Marilaure Grégoire and Gilles Louppe},
  booktitle={Machine Learning and the Physical Sciences Workshop (NeurIPS)},
  year={2025},
  url={https://arxiv.org/abs/2504.18720},
}
```
