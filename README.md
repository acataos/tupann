# Precipitation nowcasting of satellite data using physically-aligned neural networks

Transferable and Universal Physics-Aligned Nowcasting Network (TUPANN) is a satellite-only deep learning model for short-term precipitation nowcasting, designed to operate effectively in regions lacking dense weather-radar networks. It is trained on GOES-16 RRQPE and IMERG data and evaluated across diverse climates.

![alt text](https://github.com/acataos/tupann/blob/main/tupann_diagram.png)

## Abstract

Accurate short-term precipitation forecasts predominantly rely on dense weather-radar networks, limiting operational value in places most exposed to climate extremes. We present TUPANN (Transferable and Universal Physics-Aligned Nowcasting Network), a satellite-only model trained on GOES-16 RRQPE. Unlike most deep learning models for nowcasting, TUPANN decomposes the forecast into physically meaningful components: a variational encoder–decoder infers motion and intensity fields from recent imagery under optical-flow supervision, a lead-time-conditioned MaxViT evolves the latent state, and a differentiable advection operator reconstructs future frames. We evaluate TUPANN on both GOES-16 and IMERG data, in up to four distinct climates (Rio de Janeiro, Manaus, Miami, La Paz) at 10–180-min lead times using the CSI and HSS metrics over 4–64 mm/h thresholds. Comparisons against optical-flow, deep learning and hybrid baselines show that TUPANN achieves the best or second-best skill in most settings, with pronounced gains at higher thresholds. Training on multiple cities further improves performance, while cross-city experiments show modest degradation and occasional gains for rare heavy-rain regimes. The model produces smooth, interpretable motion fields aligned with numerical optical flow and runs in near real time due to the low latency of GOES‑16. These results indicate that physically aligned learning can provide nowcasts that are skillful, transferable and global.

