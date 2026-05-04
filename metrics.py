import numpy as np
import pandas as pd
import properscoring as ps
from pysteps.verification.spatialscores import fss
from scipy.stats import norm
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import torch
from torch.nn import functional as F


class Evaluator:
    def __init__(
        self,
        model_name: str,
        dataset: str,
        output_seq_len: int,
        metric_dir: str,
        pixel_thresholds: list = None,
        pixel_balancing_weights: list = None,
        dataset_max_value: float = None,
    ):
        self.pixel_thresholds = pixel_thresholds
        self.pixel_balancing_weights = pixel_balancing_weights
        self.dataset_max_value = dataset_max_value
        self.model_name = model_name
        self.dataset = dataset
        self.output_seq_len = output_seq_len
        self.metric_dir = metric_dir
        self.scales = [1, 2, 4, 8]

    def MAE(self, pred, target):
        return np.mean(np.abs(target - pred), dtype=np.float64)

    def MSE(self, pred, target):
        return np.mean((target - pred) ** 2, dtype=np.float64)

    def RMSE(self, pred, target):
        return np.sqrt(np.mean((target - pred) ** 2, dtype=np.float64))

    def B_MAE(self, pred, target, weights):
        return np.mean(weights * (np.abs(target - pred)), dtype=np.float64)

    def B_MSE(self, pred, target, weights):
        return np.mean(weights * ((target - pred) ** 2), dtype=np.float64)

    def cal_TP(self, pred=None, target=None, th=None):
        return (
            torch.where(torch.logical_and(pred >= th, target >= th), 1, 0).sum(
                dim=(-1, -2)
            )
        ).sum(dim=0)

    def cal_TN(self, pred=None, target=None, th=None):
        return (
            torch.where(torch.logical_and(pred < th, target < th), 1, 0).sum(
                dim=(-1, -2)
            )
        ).sum(dim=0)

    def cal_FP(self, pred=None, target=None, th=None):
        return (
            torch.where(torch.logical_and(pred >= th, target < th), 1, 0).sum(
                dim=(-1, -2)
            )
        ).sum(dim=0)

    def cal_FN(self, pred=None, target=None, th=None):
        return (
            torch.where(torch.logical_and(pred < th, target >= th), 1, 0).sum(
                dim=(-1, -2)
            )
        ).sum(dim=0)

    def cal_TP_per_seq(self, pred=None, target=None, th=None):
        return torch.where(torch.logical_and(pred >= th, target >= th), 1, 0).sum(
            dim=(-1, -2)
        )

    def cal_TN_per_seq(self, pred=None, target=None, th=None):
        return torch.where(torch.logical_and(pred < th, target < th), 1, 0).sum(
            dim=(-1, -2)
        )

    def cal_FP_per_seq(self, pred=None, target=None, th=None):
        return torch.where(torch.logical_and(pred >= th, target < th), 1, 0).sum(
            dim=(-1, -2)
        )

    def cal_FN_per_seq(self, pred=None, target=None, th=None):
        return torch.where(torch.logical_and(pred < th, target >= th), 1, 0).sum(
            dim=(-1, -2)
        )

    def cal_Dr(
        self, pred=None, target=None, th=None, TP=None, TN=None, FP=None, FN=None
    ):
        """
        Dr = (TP + FP)*(TP + FN) / (TP + TN + FP + FN)
        """
        if TP is None and TN is None and FP is None and FN is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            TP = self.cal_TN(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)
        return (TP + FP) * (TP + FN) / (TP + FN + FP + TN)

    def cal_POD(self, pred=None, target=None, th=None, TP=None, FN=None):
        """
        Probability of Detection = TP / (TP + FN)
        """
        if TP is None and FN is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)

        POD = TP / (TP + FN)
        POD.nan_to_num_(nan=0.0)
        return POD

    def cal_FAR(self, pred=None, target=None, th=None, FP=None, TP=None):
        """
        False Alarm Rate = FP / (FP + TP)
        """
        if FP is None and TP is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)

        FAR = FP / (FP + TP)
        FAR.nan_to_num_(nan=0.0)
        return FAR

    def cal_CSI(self, pred=None, target=None, th=None, TP=None, FP=None, FN=None):
        """
        Critical Success Index = TP / (TP + FP + FN)
        """
        if TP is None and FP is None and FN is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)

        CSI = TP / (TP + FP + FN)
        CSI.nan_to_num_(nan=0.0)
        return CSI

    def cal_HSS(
        self, pred=None, target=None, th=None, TP=None, TN=None, FP=None, FN=None
    ):
        """
        Heidke Skill Score = 2 * (TP*TN-FN*FP) / ((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN))
        """
        if TP is None and TN is None and FP is None and FN is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            TN = self.cal_TN(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)

        HSS = 2 * (TP * TN - FN * FP) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
        HSS.nan_to_num_(nan=0.0)
        return HSS

    def cal_ETS(
        self,
        pred=None,
        target=None,
        th=None,
        TP=None,
        TN=None,
        FP=None,
        FN=None,
        Dr=None,
    ):
        """
        ETS = (TP - Dr)/ (TP + TN + FP - Dr)
        """
        if TP is None and TN is None and FP is None and FN is None and Dr is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            TN = self.cal_TN(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)
            Dr = self.cal_Dr(
                pred=pred, target=target, th=th, TP=TP, TN=TN, FP=FP, FN=FN
            )

        ETS = (TP - Dr) / (TP + FN + FP - Dr)
        ETS.nan_to_num_(nan=0.0)
        return ETS

    def cal_CRPS(self, pred=None, target=None, scale=None):
        target_cdf = norm.cdf(x=target.cpu().detach().numpy(), loc=0, scale=scale)
        pred_cdf = norm.cdf(x=pred.cpu().detach().numpy(), loc=0, scale=scale)
        forecast_score = ps.crps_ensemble(target_cdf, pred_cdf).mean(axis=(0, -1, -2))
        return forecast_score

    def cal_FSS(self, pred=None, target=None, threshold=None, scale=None):
        fss_score = []
        for frame in range(target.shape[0]):
            fra_score = 0.0
            count = 0
            for j in range(target.shape[1]):
                if np.any(target[frame, j] > threshold):
                    count += 1
                    fra_score += fss(pred[frame, j], target[frame, j], threshold, scale)
            if count != 0:
                fra_score = fra_score / count
            fss_score.append(fra_score)
        return np.array(fss_score)

    def cal_image_metrics(self, target_imgs, pred_images):
        target_imgs = target_imgs.cpu().detach().numpy()
        pred_images = pred_images.cpu().detach().numpy()

        def get_pixels_balancing_weights(target_imgs):
            def get_pixel_weight(x):
                for idx, threshold in enumerate(self.pixel_thresholds):
                    if x < threshold:
                        return self.pixel_balancing_weights[idx]
                return self.pixel_balancing_weights[-1]

            return np.vectorize(get_pixel_weight)(target_imgs)

        num_sequences = target_imgs.shape[0]
        sequence_len = target_imgs.shape[1]

        sum_mae = 0.0
        sum_mse = 0.0
        sum_rmse = 0.0
        sum_b_mae = 0.0
        sum_b_mse = 0.0
        weights = get_pixels_balancing_weights(target_imgs)

        for i in range(num_sequences):
            sum_mae += self.MAE(pred_images[i], target_imgs[i])
            sum_mse += self.MSE(pred_images[i], target_imgs[i])
            sum_rmse += self.RMSE(pred_images[i], target_imgs[i])
            sum_b_mae += self.B_MAE(pred_images[i], target_imgs[i], weights[i])
            sum_b_mse += self.B_MSE(pred_images[i], target_imgs[i], weights[i])

        mae_s = sum_mae / num_sequences
        mse_s = sum_mse / num_sequences
        rmse_s = sum_rmse / num_sequences
        b_mae_s = sum_b_mae / num_sequences
        b_mse_s = sum_b_mse / num_sequences

        aux = 0.0
        ssim_s = 0.0
        for i in range(num_sequences):
            for j in range(sequence_len):
                aux += SSIM(
                    np.squeeze(pred_images)[i, j],
                    np.squeeze(target_imgs)[i, j],
                    data_range=self.dataset_max_value,
                )
            ssim_s += aux / sequence_len
            aux = 0.0
        ssim_s = ssim_s / num_sequences

        aux = 0.0
        psnr_s = 0.0
        for i in range(num_sequences):
            for j in range(sequence_len):
                aux += PSNR(pred_images[i, j], target_imgs[i, j], data_range=self.dataset_max_value)
            psnr_s += aux / sequence_len
            aux = 0.0
        psnr_s = psnr_s / num_sequences

        return mae_s, mse_s, rmse_s, b_mae_s, b_mse_s, ssim_s, psnr_s

    def cal_score(self, pred, target):
        if torch.is_tensor(pred):
            pred = pred.detach()
        if torch.is_tensor(target):
            target = target.detach()

        def check_shape(x):
            return x.unsqueeze(0) if len(x.shape) == 3 else x

        pred = check_shape(pred)
        target = check_shape(target)

        if isinstance(pred, torch.Tensor):
            pred = torch.nan_to_num(pred, nan=0)
        if isinstance(target, torch.Tensor):
            target = torch.nan_to_num(target, nan=0)

        pooled_crps = []
        for scale in self.scales:
            pooled_crps.append(
                self.cal_CRPS(
                    F.avg_pool2d(pred, kernel_size=scale),
                    F.avg_pool2d(target, kernel_size=scale),
                    scale=scale,
                )
            )

        ets = []
        pod = []
        far = []
        csi = []
        hss = []

        num_sequences = target.shape[0]
        sequence_len = target.shape[1]
        ets_s = torch.zeros(sequence_len).cuda()
        pod_s = torch.zeros(sequence_len).cuda()
        far_s = torch.zeros(sequence_len).cuda()
        csi_s = torch.zeros(sequence_len).cuda()
        hss_s = torch.zeros(sequence_len).cuda()

        for th in self.pixel_thresholds:
            for i in range(num_sequences):
                TP = self.cal_TP_per_seq(pred=pred[i], target=target[i], th=th)
                TN = self.cal_TN_per_seq(pred=pred[i], target=target[i], th=th)
                FP = self.cal_FP_per_seq(pred=pred[i], target=target[i], th=th)
                FN = self.cal_FN_per_seq(pred=pred[i], target=target[i], th=th)
                Dr = self.cal_Dr(th=th, TP=TP, TN=TN, FP=FP, FN=FN)

                ets_s += self.cal_ETS(TP=TP, TN=TN, FP=FP, FN=FN, Dr=Dr)
                pod_s += self.cal_POD(TP=TP, FN=FN)
                far_s += self.cal_FAR(TP=TP, FP=FP)
                csi_s += self.cal_CSI(TP=TP, FP=FP, FN=FN)
                hss_s += self.cal_HSS(TP=TP, TN=TN, FP=FP, FN=FN)

            ets.append(ets_s / num_sequences)
            pod.append(pod_s / num_sequences)
            far.append(far_s / num_sequences)
            csi.append(csi_s / num_sequences)
            hss.append(hss_s / num_sequences)

            ets_s.zero_()
            pod_s.zero_()
            far_s.zero_()
            csi_s.zero_()
            hss_s.zero_()

        ets = torch.stack(ets)
        pod = torch.stack(pod)
        far = torch.stack(far)
        csi = torch.stack(csi)
        hss = torch.stack(hss)
        pooled_crps = np.array(pooled_crps)

        target = target.permute(1, 0, 2, 3)
        pred = pred.permute(1, 0, 2, 3)
        target = target.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        sum_fss = []
        for th in self.pixel_thresholds:
            scale_fss = [
                self.cal_FSS(pred, target, th, scale) for scale in [1, 4, 16, 32]
            ]
            sum_fss.append(np.array(scale_fss))
        sum_fss = np.array(sum_fss)

        return ets, pod, far, csi, hss, pooled_crps, sum_fss

    def save_image_metrics(self, test_images, pred_images):
        mae_s, mse_s, rmse_s, b_mae_s, b_mse_s, ssim_s, psnr_s = self.cal_image_metrics(
            test_images, pred_images
        )

        image_metrics = [
            {
                "model_name": self.model_name,
                "dataset": self.dataset,
                "output_seq_len": self.output_seq_len,
                "MAE": mae_s,
                "MSE": mse_s,
                "RMSE": rmse_s,
                "B-MAE": b_mae_s,
                "B-MSE": b_mse_s,
                "SSIM": ssim_s,
                "PSNR": psnr_s,
            }
        ]

        df_image_metrics = pd.DataFrame(image_metrics)
        df_image_metrics.to_csv(
            f"{self.metric_dir}/{self.model_name}_{self.dataset}_image_metrics.csv",
            index=False,
        )

    def save_score_txt(self, test_images, pred_images, save_file):
        test_images = test_images[:, :, 0]
        pred_images = pred_images[:, :, 0]
        ets, pod, far, csi, hss, pooled_crps, sum_fss = self.cal_score(
            pred_images, test_images
        )  # bias
        with open(save_file, "w") as f:
            for i, threshold in enumerate(self.pixel_thresholds):
                f.write("Threshold = %g:\n" % threshold)
                f.write("   ETS=%s\n" % str(list(ets[i, :].tolist())))
                f.write("   POD=%s\n" % str(list(pod[i, :].tolist())))
                f.write("   FAR=%s\n" % str(list(far[i, :].tolist())))
                f.write("   CSI=%s\n" % str(list(csi[i, :].tolist())))
                f.write("   HSS=%s\n" % str(list(hss[i, :].tolist())))
                f.write(
                    "   ETS stat: avg %g/final %g\n"
                    % (torch.mean(ets[i, :]), ets[i, -1].item())
                )
                f.write(
                    "   POD stat: avg %g/final %g\n"
                    % (torch.mean(pod[i, :]), pod[i, -1].item())
                )
                f.write(
                    "   FAR stat: avg %g/final %g\n"
                    % (torch.mean(far[i, :]), far[i, -1].item())
                )
                f.write(
                    "   CSI stat: avg %g/final %g\n"
                    % (torch.mean(csi[i, :]), csi[i, -1].item())
                )
                f.write(
                    "   HSS stat: avg %g/final %g\n"
                    % (torch.mean(hss[i, :]), hss[i, -1].item())
                )

            for i, threshold in enumerate([1, 2, 4, 8]):
                f.write("Scale = %g:\n" % threshold)
                f.write("   Pooled_CRPS=%s\n" % str(list(pooled_crps[i, :].tolist())))

            for i, threshold in enumerate(self.pixel_thresholds):
                for j, scale in enumerate([1, 4, 16, 32]):
                    f.write("Threshold = %g:\n" % threshold)
                    f.write("   Scale = %g:\n" % scale)
                    f.write("       Fss=%s\n" % str(list(sum_fss[i, j].tolist())))

    def save_score_csv(self, test_images, pred_images):
        test_images = test_images[:, :, 0]
        pred_images = pred_images[:, :, 0]
        ets, pod, far, csi, hss, pooled_crps, sum_fss = self.cal_score(
            pred_images, test_images
        )

        thresholds_metrics = []
        for i, threshold in enumerate(self.pixel_thresholds):
            thresholds_metrics.append(
                {
                    "model_name": self.model_name,
                    "dataset": self.dataset,
                    "output_seq_len": self.output_seq_len,
                    "Threshold": threshold,
                    "ETS": ets[i, -1].item(),
                    "POD": pod[i, -1].item(),
                    "FAR": far[i, -1].item(),
                    "CSI": csi[i, -1].item(),
                    "HSS": hss[i, -1].item(),
                    "ETS_avg": torch.mean(ets[i, :]).item(),
                    "POD_avg": torch.mean(pod[i, :]).item(),
                    "FAR_avg": torch.mean(far[i, :]).item(),
                    "CSI_avg": torch.mean(csi[i, :]).item(),
                    "HSS_avg": torch.mean(hss[i, :]).item(),
                    "ETS_all": str(list(ets[i, :].tolist())),
                    "POD_all": str(list(pod[i, :].tolist())),
                    "FAR_all": str(list(far[i, :].tolist())),
                    "CSI_all": str(list(csi[i, :].tolist())),
                    "HSS_all": str(list(hss[i, :].tolist())),
                }
            )

        df_thresholds_metrics = pd.DataFrame(thresholds_metrics)
        df_thresholds_metrics.to_csv(
            f"{self.metric_dir}/{self.model_name}_{self.dataset}_threshold_metrics.csv",
            index=False,
        )

        pooled_crps_metrics = []
        for i, scale in enumerate([1, 2, 4, 8]):
            for j, pcrps in enumerate(list(pooled_crps[i, :].tolist())):
                pooled_crps_metrics.append(
                    {
                        "model_name": self.model_name,
                        "dataset": self.dataset,
                        "output_seq_len": self.output_seq_len,
                        "Scale": scale,
                        "Frame": j + 1,
                        "Pooled_CRPS": pcrps,
                    }
                )
        df_pooled_crps_metrics = pd.DataFrame(pooled_crps_metrics)
        df_pooled_crps_metrics.to_csv(
            f"{self.metric_dir}/{self.model_name}_{self.dataset}_pooled_crps.csv",
            index=False,
        )

        fss_metrics = []
        for i, threshold in enumerate(self.pixel_thresholds):
            for j, scale in enumerate([1, 4, 16, 32]):
                for k, fss_score in enumerate(list(sum_fss[i, j].tolist())):
                    fss_metrics.append(
                        {
                            "model_name": self.model_name,
                            "dataset": self.dataset,
                            "output_seq_len": self.output_seq_len,
                            "Threshold": threshold,
                            "Scale": scale,
                            "Frame": k + 1,
                            "FSS": fss_score,
                        }
                    )
        df_fss_metrics = pd.DataFrame(fss_metrics)
        df_fss_metrics.to_csv(
            f"{self.metric_dir}/{self.model_name}_{self.dataset}_fss.csv", index=False
        )
