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

Install the required libraries:

```bash
pip install -r requirements.txt
```


### Preparing the Data

In order to prepare the data, first download the GOES-16 RRQPE dataset using the script ``download_goes16_rrqpe.sh``, which requires a file describing the datetimes to be downloaded. We provide a full list of datetimes for rain events in the city of Rio de Janeiro used in the paper in ``configs/data/datetimes-goes16_rrqpe-rj.txt``. A smaller sample is present in ``configs/data/datetimes-goes16_rrqpe-rj-sample.txt`` for testing purposes. To download the data, run:

```bash
sh download_goes16_rrqpe.sh configs/data/datetimes-goes16_rrqpe-rj.txt
```

Once the data is successfully downloaded, you can process it by running

```bash
python src/data/process/process_satellite.py data/rain_events/goes16_rrqpe-rj.yaml --location rio_de_janeiro
```
