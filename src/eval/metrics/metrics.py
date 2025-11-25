import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


@torch.no_grad()
def cal_SSIM(gt, pred, is_img=True, collapse_lag=False):
    """
    iter_cal=True, gt.shape=pred.shape=[nb b t c h w]
    iter_cal=False, gt.shape=pred.shape=[n t c h w]
    """
    cal_ssim = StructuralSimilarityIndexMeasure(data_range=int(torch.max(gt) - torch.min(gt)), reduction=None).to(
        gt.device
    )
    if is_img:
        pred = torch.maximum(pred, torch.min(gt))
        pred = torch.minimum(pred, torch.max(gt))
    n = pred.shape[0]
    pred = einops.rearrange(pred, "n t c h w -> (n t) c h w")
    gt = einops.rearrange(gt, "n t c h w -> (n t) c h w")
    ssim = cal_ssim(pred, gt).cpu()
    ssim = einops.rearrange(ssim, "(n t) -> n t", n=n)

    if collapse_lag:
        ssim = torch.sum(ssim)
    else:
        ssim = torch.sum(ssim, axis=0)
    if torch.isnan(ssim).any():
        ssim = torch.zeros_like(ssim)
    return ssim, n


@torch.no_grad()
def cal_PSNR(gt, pred, is_img=True, collapse_lag=False):
    """
    gt.shape=pred.shape=[n t c h w]
    """
    cal_psnr = PeakSignalNoiseRatio().to(gt.device)
    if is_img:
        pred = torch.maximum(pred, torch.min(gt))
        pred = torch.minimum(pred, torch.max(gt))
    pred = einops.rearrange(pred, "n t c h w -> (n t) c h w")
    gt = einops.rearrange(gt, "n t c h w -> (n t) c h w")
    psnr = 0
    for n in range(pred.shape[0]):
        psnr += cal_psnr(pred[n], gt[n]).cpu()
    return (psnr / pred.shape[0]).item()


@torch.no_grad()
def cal_CRPS(gt, pred, type="avg", scale=4, mode="mean", eps=1e-10):
    """
    gt: (b, t, c, h, w)
    pred: (b, n, t, c, h, w)
    """
    assert mode in ["mean", "raw"], "CRPS mode should be mean or raw"
    _normal_dist = torch.distributions.Normal(0, 1)
    _frac_sqrt_pi = 1 / np.sqrt(np.pi)

    b, n, t, _, _, _ = pred.shape
    gt = rearrange(gt, "b t c h w -> (b t) c h w")
    pred = rearrange(pred, "b n t c h w -> (b n t) c h w")
    if type == "avg":
        pred = F.avg_pool2d(pred, scale, stride=scale)
        gt = F.avg_pool2d(gt, scale, stride=scale)
    elif type == "max":
        pred = F.max_pool2d(pred, scale, stride=scale)
        gt = F.max_pool2d(gt, scale, stride=scale)
    else:
        gt = gt
        pred = pred
    gt = rearrange(gt, "(b t) c h w -> b t c h w", b=b)
    pred = rearrange(pred, "(b n t) c h w -> b n t c h w", b=b, n=n)

    pred_mean = torch.mean(pred, dim=1)
    pred_std = torch.std(pred, dim=1) if n > 1 else torch.zeros_like(pred_mean)
    normed_diff = (pred_mean - gt + eps) / (pred_std + eps)
    cdf = _normal_dist.cdf(normed_diff)
    pdf = _normal_dist.log_prob(normed_diff).exp()

    crps = (pred_std + eps) * (normed_diff *
                               (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)
    if mode == "mean":
        return torch.mean(crps).item()
    return crps.item()


def _threshold(target, pred, T):
    t = (target >= T).float()
    p = (pred >= T).float()
    is_nan = torch.logical_or(torch.isnan(target), torch.isnan(pred))
    t[is_nan] = 0
    p[is_nan] = 0
    return t, p


class SEVIRSkillScore(object):
    def __init__(
        self,
        layout="NHWT",
        mode="1",
        seq_len=None,
        preprocess_type="identity",
        threshold_list=[1, 2, 4, 8, 16, 32, 64],
        metrics_list=[
            "csi",
            "csi-4-avg",
            "csi-16-avg",
            "csi-4-max",
            "csi-16-max",
            "bias",
            "sucr",
            "pod",
            "hss",
        ],  # ['csi', 'bias', 'sucr', 'pod'],
        dist_eval=False,
        #  device='cuda',
        eps=1e-4,
    ):
        self.layout = layout
        self.preprocess_type = preprocess_type
        self.threshold_list = threshold_list
        self.metrics_list = metrics_list
        self.eps = eps
        self.mode = mode
        self.seq_len = seq_len

        self.dist_eval = dist_eval
        # self.device = device

        if mode in ("0",):
            self.keep_seq_len_dim = False
            state_shape = (len(self.threshold_list),)
        elif mode in ("1", "2"):
            self.keep_seq_len_dim = True
            assert isinstance(
                self.seq_len, int), "seq_len must be provided when we need to keep seq_len dim."
            state_shape = (len(self.threshold_list), self.seq_len)
        else:
            raise NotImplementedError(f"mode {mode} not supported!")

        self.hits = torch.zeros(state_shape)
        self.misses = torch.zeros(state_shape)
        self.fas = torch.zeros(state_shape)
        self.cor = torch.zeros(state_shape)
        self.error_list = []

        ## pooling csi ##
        self.hits_avg_pool_4 = torch.zeros(state_shape)
        self.misses_avg_pool_4 = torch.zeros(state_shape)
        self.fas_avg_pool_4 = torch.zeros(state_shape)

        self.hits_max_pool_4 = torch.zeros(state_shape)
        self.misses_max_pool_4 = torch.zeros(state_shape)
        self.fas_max_pool_4 = torch.zeros(state_shape)

        self.hits_avg_pool_16 = torch.zeros(state_shape)
        self.misses_avg_pool_16 = torch.zeros(state_shape)
        self.fas_avg_pool_16 = torch.zeros(state_shape)

        self.hits_max_pool_16 = torch.zeros(state_shape)
        self.misses_max_pool_16 = torch.zeros(state_shape)
        self.fas_max_pool_16 = torch.zeros(state_shape)

        ### for image losses ###
        self.sum_ssim = torch.zeros(
            self.seq_len) if self.keep_seq_len_dim else 0
        self.sum_psnr = torch.zeros(
            self.seq_len) if self.keep_seq_len_dim else 0
        self.num_samples = 0

    def pod(self, hits, misses, fas, eps):
        return hits / (hits + misses + eps)

    def sucr(self, hits, misses, fas, eps):
        return hits / (hits + fas + eps)

    def csi(self, hits, misses, fas, eps):
        return hits / (hits + misses + fas + eps)

    def bias(self, hits, misses, fas, eps):
        bias = (hits + fas) / (hits + misses + eps)
        logbias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
        return logbias

    def hss(self, hits, misses, fas, cor, eps):
        hss = 2 * (hits * cor - misses * fas) / ((hits + misses) *
                                                 (misses + cor) + (hits + fas) * (fas + cor) + eps)
        return hss

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.hits)
        dist.all_reduce(self.misses)
        dist.all_reduce(self.fas)
        ### avg 4 ###
        dist.all_reduce(self.hits_avg_pool_4)
        dist.all_reduce(self.misses_avg_pool_4)
        dist.all_reduce(self.fas_avg_pool_4)
        ### max 4 ###
        dist.all_reduce(self.hits_max_pool_4)
        dist.all_reduce(self.misses_max_pool_4)
        dist.all_reduce(self.fas_max_pool_4)
        ### avg 16 ###
        dist.all_reduce(self.hits_avg_pool_16)
        dist.all_reduce(self.misses_avg_pool_16)
        dist.all_reduce(self.fas_avg_pool_16)
        ### max 16 ###
        dist.all_reduce(self.hits_max_pool_16)
        dist.all_reduce(self.misses_max_pool_16)
        dist.all_reduce(self.fas_max_pool_16)

    @property
    def hits_misses_fas_reduce_dims(self):
        if not hasattr(self, "_hits_misses_fas_reduce_dims"):
            seq_dim = self.layout.find("T")
            self._hits_misses_fas_reduce_dims = list(range(len(self.layout)))
            if self.keep_seq_len_dim:
                self._hits_misses_fas_reduce_dims.pop(seq_dim)
        return self._hits_misses_fas_reduce_dims

    def preprocess(self, pred, target):
        if self.preprocess_type == "sevir":
            pred = pred.detach() / (1.0 / 255.0)
            target = target.detach() / (1.0 / 255.0)
        elif self.preprocess_type == "meteonet":
            pred = pred.detach() / (1.0 / 70.0)
            target = target.detach() / (1.0 / 70.0)
        elif self.preprocess_type == "identity":
            pred = pred.detach()
            target = target.detach()
        else:
            raise NotImplementedError
        return pred, target

    def preprocess_pool(self, pred, target, pool_size=4, type="avg"):
        if self.preprocess_type == "sevir":
            pred = pred.detach() / (1.0 / 255.0)
            target = target.detach() / (1.0 / 255.0)
        elif self.preprocess_type == "meteonet":
            pred = pred.detach() / (1.0 / 70.0)
            target = target.detach() / (1.0 / 70.0)
        elif self.preprocess_type == "identity":
            pred = pred.detach()
            target = target.detach()
        b, t, _, _, _ = pred.shape
        pred = rearrange(pred, "b t c h w -> (b t) c h w")
        target = rearrange(target, "b t c h w -> (b t) c h w")
        if type == "avg":
            pred = F.avg_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.avg_pool2d(
                target, kernel_size=pool_size, stride=pool_size)
        elif type == "max":
            pred = F.max_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.max_pool2d(
                target, kernel_size=pool_size, stride=pool_size)
        pred = rearrange(pred, "(b t) c h w -> b t c h w", b=b)
        target = rearrange(target, "(b t) c h w -> b t c h w", b=b)
        return pred, target

    def calc_seq_hits_misses_fas(self, pred, target, threshold):
        with torch.no_grad():
            t, p = _threshold(target, pred, threshold)
            hits = torch.sum(t * p, dim=self.hits_misses_fas_reduce_dims).int()
            misses = torch.sum(
                t * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
            fas = torch.sum(
                (1 - t) * p, dim=self.hits_misses_fas_reduce_dims).int()
            cor = torch.sum((1 - t) * (1 - p),
                            dim=self.hits_misses_fas_reduce_dims).int()
        return hits, misses, fas, cor

    @torch.no_grad()
    def update(self, pred, target, metadata=None):
        ## pool 1 ##
        self.hits = self.hits.to(pred.device)
        self.misses = self.misses.to(pred.device)
        self.fas = self.fas.to(pred.device)
        self.cor = self.cor.to(pred.device)
        _pred, _target = self.preprocess(pred, target)
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(
                _pred, _target, threshold)
            batch_hits, batch_misses, batch_fas, batch_cor = zip(
                *[
                    self.calc_seq_hits_misses_fas(_pred[j].unsqueeze(
                        0), _target[j].unsqueeze(0), threshold)
                    for j in range(_pred.shape[0])
                ]
            )
            batch_hits = torch.stack(batch_hits)
            batch_misses = torch.stack(batch_misses)
            batch_fas = torch.stack(batch_fas)
            batch_cor = torch.stack(batch_cor)
            if metadata is not None:
                for j in range(batch_hits.shape[0]):
                    for k in range(batch_hits.shape[1]):
                        self.error_list.append(
                            {
                                **dict(
                                    [
                                        (key, v[j]) if key != "index" else (
                                            key, v[j].cpu().item())
                                        for key, v in metadata.items()
                                    ]
                                ),
                                "threshold": threshold,
                                "hits": batch_hits[j][k].cpu().numpy(),
                                "misses": batch_misses[j][k].cpu().numpy(),
                                "fas": batch_fas[j][k].cpu().numpy(),
                                "cor": batch_cor[j][k].cpu().numpy(),
                                "lag": k,
                            }
                        )
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas
            self.cor[i] += cor
        ## max pool 4 ##
        self.hits_max_pool_4 = self.hits_max_pool_4.to(pred.device)
        self.misses_max_pool_4 = self.misses_max_pool_4.to(pred.device)
        self.fas_max_pool_4 = self.fas_max_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(
            pred, target, pool_size=4, type="max")
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(
                _pred, _target, threshold)
            self.hits_max_pool_4[i] += hits
            self.misses_max_pool_4[i] += misses
            self.fas_max_pool_4[i] += fas
        ## max pool 16 ##
        self.hits_max_pool_16 = self.hits_max_pool_16.to(pred.device)
        self.misses_max_pool_16 = self.misses_max_pool_16.to(pred.device)
        self.fas_max_pool_16 = self.fas_max_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(
            pred, target, pool_size=16, type="max")
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(
                _pred, _target, threshold)
            self.hits_max_pool_16[i] += hits
            self.misses_max_pool_16[i] += misses
            self.fas_max_pool_16[i] += fas
        ## avg pool 4 ##
        self.hits_avg_pool_4 = self.hits_avg_pool_4.to(pred.device)
        self.misses_avg_pool_4 = self.misses_avg_pool_4.to(pred.device)
        self.fas_avg_pool_4 = self.fas_avg_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(
            pred, target, pool_size=4, type="avg")
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(
                _pred, _target, threshold)
            self.hits_avg_pool_4[i] += hits
            self.misses_avg_pool_4[i] += misses
            self.fas_avg_pool_4[i] += fas
        ## avg pool 16 ##
        self.hits_avg_pool_16 = self.hits_avg_pool_16.to(pred.device)
        self.misses_avg_pool_16 = self.misses_avg_pool_16.to(pred.device)
        self.fas_avg_pool_16 = self.fas_avg_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(
            pred, target, pool_size=16, type="avg")
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(
                _pred, _target, threshold)
            self.hits_avg_pool_16[i] += hits
            self.misses_avg_pool_16[i] += misses
            self.fas_avg_pool_16[i] += fas

        ### for ssim losses ###
        metrics_dict = self.get_single_frame_metrics(target, pred)
        self.sum_ssim += metrics_dict["ssim"][0]
        self.num_samples += metrics_dict["ssim"][1]
        self.sum_psnr += metrics_dict["psnr"]

    def _get_hits_misses_fas(self, metric_name):
        if metric_name.endswith("-4-avg"):
            hits = self.hits_avg_pool_4
            misses = self.misses_avg_pool_4
            fas = self.fas_avg_pool_4
        elif metric_name.endswith("-16-avg"):
            hits = self.hits_avg_pool_16
            misses = self.misses_avg_pool_16
            fas = self.fas_avg_pool_16
        elif metric_name.endswith("-4-max"):
            hits = self.hits_max_pool_4
            misses = self.misses_max_pool_4
            fas = self.fas_max_pool_4
        elif metric_name.endswith("-16-max"):
            hits = self.hits_max_pool_16
            misses = self.misses_max_pool_16
            fas = self.fas_max_pool_16
        else:
            hits = self.hits
            misses = self.misses
            fas = self.fas
        return [hits, misses, fas]

    def _get_correct_negtives(self):
        return self.cor

    @torch.no_grad()
    def compute(self):
        if self.dist_eval:
            self.synchronize_between_processes()

        metrics_dict = {
            "pod": self.pod,
            "csi": self.csi,
            "csi-4-avg": self.csi,
            "csi-16-avg": self.csi,
            "csi-4-max": self.csi,
            "csi-16-max": self.csi,
            "sucr": self.sucr,
            "bias": self.bias,
            "hss": self.hss,
        }
        ret = {}
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}

        for metrics in self.metrics_list:
            if self.keep_seq_len_dim:
                score_avg = np.zeros((self.seq_len,))
            else:
                score_avg = 0
            hits, misses, fas = self._get_hits_misses_fas(metrics)
            # scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            if metrics != "hss":
                scores = metrics_dict[metrics](hits, misses, fas, self.eps)
            else:
                cor = self._get_correct_negtives()
                scores = metrics_dict[metrics](
                    hits, misses, fas, cor, self.eps)
            scores = scores.detach().cpu().numpy()
            for i, threshold in enumerate(self.threshold_list):
                if self.keep_seq_len_dim:
                    score = scores[i]  # shape = (seq_len, )
                else:
                    score = scores[i].item()  # shape = (1, )
                if self.mode in ("0", "1"):
                    ret[threshold][metrics] = score
                elif self.mode in ("2",):
                    ret[threshold][metrics] = np.mean(score).item()
                else:
                    raise NotImplementedError
                score_avg += score
            score_avg /= len(self.threshold_list)
            if self.mode in ("0", "1"):
                ret["avg"][metrics] = score_avg
            elif self.mode in ("2",):
                ret["avg"][metrics] = np.mean(score_avg).item()
            else:
                raise NotImplementedError

        ### for image losses ###
        ssim = (
            self.sum_ssim / self.num_samples
            if self.keep_seq_len_dim
            else self.sum_ssim / (self.num_samples * self.seq_len)
        )
        return ret, ssim

    @torch.no_grad()
    def save_metrics_time_series(self, filepath):
        import pandas as pd

        df = pd.DataFrame(self.error_list)
        df.to_csv(filepath, index=False)

    @torch.no_grad()
    def get_single_frame_metrics(
        self,
        target,
        pred,
        metrics=[
            "ssim",
            "psnr",
        ],
    ):  # 'cspr', 'cspr-4-avg', 'cspr-16-avg', 'cspr-4-max', 'cspr-16-max'
        metric_funcs = {"ssim": cal_SSIM, "psnr": cal_PSNR}
        metrics_dict = {}
        for metric in metrics:
            metric_fun = metric_funcs[metric]
            metrics_dict[metric] = metric_fun(
                gt=target, pred=pred, is_img=False, collapse_lag=not self.keep_seq_len_dim
            )
        return metrics_dict

    def reset(self):
        self.hits = self.hits * 0
        self.misses = self.misses * 0
        self.fas = self.fas * 0

        self.hits_avg_pool_4 *= 0
        self.hits_avg_pool_16 *= 0
        self.hits_max_pool_4 *= 0
        self.hits_max_pool_16 *= 0

        self.misses_avg_pool_4 *= 0
        self.misses_avg_pool_16 *= 0
        self.misses_max_pool_4 *= 0
        self.misses_max_pool_16 *= 0

        self.fas_avg_pool_4 *= 0
        self.fas_avg_pool_16 *= 0
        self.fas_max_pool_4 *= 0
        self.fas_max_pool_16 *= 0
