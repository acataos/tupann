# Precipitation nowcasting of satellite data using physically-aligned neural networks

Transferable and Universal Physics-Aligned Nowcasting Network (TUPANN) is a satellite-only deep learning model for short-term precipitation nowcasting, designed to operate effectively in regions lacking dense weather-radar networks. It is trained on GOES-16 RRQPE and IMERG data and evaluated across diverse climates.

![alt text](https://github.com/acataos/tupann/blob/main/tupann_diagram.png)

## Abstract

Accurate short-term precipitation forecasts predominantly rely on dense weather-radar networks, limiting operational value in places most exposed to climate extremes. We present TUPANN (Transferable and Universal Physics-Aligned Nowcasting Network), a satellite-only model trained on GOES-16 RRQPE. Unlike most deep learning models for nowcasting, TUPANN decomposes the forecast into physically meaningful components: a variational encoder–decoder infers motion and intensity fields from recent imagery under optical-flow supervision, a lead-time-conditioned MaxViT evolves the latent state, and a differentiable advection operator reconstructs future frames. We evaluate TUPANN on both GOES-16 and IMERG data, in up to four distinct climates (Rio de Janeiro, Manaus, Miami, La Paz) at 10–180-min lead times using the CSI and HSS metrics over 4–64 mm/h thresholds. Comparisons against optical-flow, deep learning and hybrid baselines show that TUPANN achieves the best or second-best skill in most settings, with pronounced gains at higher thresholds. Training on multiple cities further improves performance, while cross-city experiments show modest degradation and occasional gains for rare heavy-rain regimes. The model produces smooth, interpretable motion fields aligned with numerical optical flow and runs in near real time due to the low latency of GOES‑16. These results indicate that physically aligned learning can provide nowcasts that are skillful, transferable and global.

## Usage

This repository provides the code used to process data and train TUPANN. First, clone the repository:

```bash
git clone https://github.com/acataos/tupann.git
cd tupann
```

### Installation

TUPANN requires Python 3.10 or newer and a CUDA-capable GPU. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Set the project root on the Python path so that all scripts can locate the ``src`` package:

```bash
export PYTHONPATH=$PWD
```


### Preparing the Data

In order to prepare the data, first download the GOES-16 RRQPE dataset using the script ``download_goes16_rrqpe.sh``, which requires a file describing the datetimes to be downloaded. We provide a full list of datetimes for rain events in the city of Rio de Janeiro used in the paper in ``configs/data/datetimes-goes16_rrqpe-rj.txt``. A smaller sample is present in ``configs/data/datetimes-goes16_rrqpe-rj_sample.txt`` for testing purposes. To download the data, run:

```bash
bash src/data/download_goes16_rrqpe.sh configs/data/datetimes-goes16_rrqpe-rj.txt
```

Once the data is successfully downloaded, you can process it by running

```bash
python src/data/process/process_data.py data/rain_events/goes16_rrqpe-rj.yaml --location rio_de_janeiro
```

### Generating splits

The provided datetime files in ``configs/data/`` already contain pre-computed train, validation and test splits for Rio de Janeiro. To generate splits for a different set of rain events, run:

```bash
python src/data/process/generate_split.py \
    --yaml_file data/rain_events/goes16_rrqpe-rj.yaml \
    --exclude_first 10 --exclude_last 1
```

This produces ``configs/data/{split}_datetimes[10:-1]-goes16_rrqpe-rj.txt`` files for each of the three splits. The ``--exclude_first`` and ``--exclude_last`` arguments control how many timesteps are dropped from the start and end of each rain event to avoid boundary effects.

### Training

Training requires two stages: first the variational autoencoder, then the full TUPANN model. Both stages use the ``train.py`` script, which takes a data configuration, an evaluation configuration and a model hyperparameter configuration as arguments.

To train the autoencoder, run:

```bash
python src/models/train.py \
    --data_config goes16_rj_autoenc \
    --eval_config goes16_eval_rio \
    --hparams_config autoenc_tupann_image_loss
```

Checkpoint files are saved to ``models/tupann_autoenc/<model_hash>/train/<data_config>/``. Once training is complete, set ``autoenc_ckpt`` and ``autoenc_hash`` in ``configs/models/tupann_image_loss.yaml`` to the checkpoint filename and hash from that directory. Then train TUPANN by running:

```bash
python src/models/train.py \
    --data_config goes16_tuppann_rj_rain_events_darts_paper \
    --eval_config goes16_eval_rio \
    --hparams_config tupann_image_loss
```

Training can be resumed from an existing checkpoint by passing ``--ckpt_file <filename>`` to ``train.py``, where ``<filename>`` is the name of a ``.ckpt`` file in the model output directory. Multiple GPU training is supported via the ``--devices`` argument.

### Evaluation

To run prediction and compute evaluation metrics on the test set, run:

```bash
python src/models/predict.py \
    --data_config goes16_tuppann_rj_rain_events_darts_paper \
    --hparams_config tupann_image_loss \
    --fold test \
    --ckpt_file <checkpoint_file> \
    --devices 0 \
    --save_prediction_options 2
```

where ``<checkpoint_file>`` is the name of a ``.ckpt`` file in the model output directory. Predictions are saved as HDF files under ``models/<model_hash>/predictions/``. Evaluation metrics (CSI and HSS across multiple rainfall thresholds and lead times) are saved as CSV files to ``eval/<model_hash>/<data_config>/``. To evaluate on the validation set instead, pass ``--fold val``.
