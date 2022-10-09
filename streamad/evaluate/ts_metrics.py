# This script is from https://github.com/KurochkinAlexey/Time-series-precision-recall, Thanks!

import numpy as np


class TSMetric:
    def __init__(
        self,
        metric_option="classic",
        beta=1.0,
        alpha_r=0.0,
        cardinality="one",
        bias_p="flat",
        bias_r="flat",
    ):

        assert (alpha_r >= 0) & (alpha_r <= 1)
        assert metric_option in ["classic", "time-series", "numenta"]
        assert beta > 0
        assert cardinality in ["one", "reciprocal", "udf_gamma"]
        assert bias_p in ["flat", "front", "middle", "back"]
        assert bias_r in ["flat", "front", "middle", "back"]

        self.metric_option = metric_option
        self.beta = beta
        self.alpha_r = alpha_r
        self.alpha_p = 0
        self.cardinality = cardinality
        self.bias_p = bias_p
        self.bias_r = bias_r

    def _udf_gamma(self, overlap, task_type):
        """
        user defined gamma
        """
        return 1.0

    def _gamma_select(self, gamma, overlap, task_type):
        if gamma == "one":
            return 1.0
        elif gamma == "reciprocal":
            if overlap > 1:
                return 1.0 / overlap
            else:
                return 1.0
        elif gamma == "udf_gamma_def":
            if overlap > 1:
                return 1.0 / self._udf_gamma(overlap, task_type)
            else:
                return 1.0

    def _gamma_function(self, overlap_count, task_type):
        overlap = overlap_count[0]
        if task_type == 0:
            return self._gamma_select(self.cardinality, overlap, task_type)
        elif task_type == 1:
            return self._gamma_select(self.cardinality, overlap, task_type)
        else:
            raise Exception("invalid argument in gamma function")

    def _compute_omega_reward(self, r1, r2, overlap_count, task_type):
        if r1[1] < r2[0] or r1[0] > r2[1]:
            return 0
        else:
            overlap_count[0] += 1
            overlap = np.zeros(r1.shape)
            overlap[0] = max(r1[0], r2[0])
            overlap[1] = min(r1[1], r2[1])
            return self._omega_function(r1, overlap, task_type)

    def _omega_function(self, rrange, overlap, task_type):
        anomaly_length = rrange[1] - rrange[0] + 1
        my_positional_bias = 0
        max_positional_bias = 0
        temp_bias = 0
        for i in range(1, anomaly_length + 1):
            temp_bias = self._delta_function(i, anomaly_length, task_type)
            max_positional_bias += temp_bias
            j = rrange[0] + i - 1
            if j >= overlap[0] and j <= overlap[1]:
                my_positional_bias += temp_bias
        if max_positional_bias > 0:
            res = my_positional_bias / max_positional_bias
            return res
        else:
            return 0

    def _delta_function(self, t, anomaly_length, task_type):
        if task_type == 0:
            return self._delta_select(self.bias_p, t, anomaly_length, task_type)
        elif task_type == 1:
            return self._delta_select(self.bias_r, t, anomaly_length, task_type)
        else:
            raise Exception("Invalid task type in delta function")

    def _delta_select(self, delta, t, anomaly_length, task_type):
        if delta == "flat":
            return 1.0
        elif delta == "front":
            return float(anomaly_length - t + 1.0)
        elif delta == "middle":
            if t <= anomaly_length / 2.0:
                return float(t)
            else:
                return float(anomaly_length - t + 1.0)
        elif delta == "back":
            return float(t)
        elif delta == "udf_delta":
            return self._udf_delta(t, anomaly_length, task_type)
        else:
            raise Exception("Invalid positional bias value")

    def _udf_delta(self, t, anomaly_length, task_type):
        """
        user defined delta function
        """
        return 1.0

    def _update_precision(self, real_anomalies, predicted_anomalies):
        precision = 0
        if len(predicted_anomalies) == 0:
            return 0
        for i in range(len(predicted_anomalies)):
            range_p = predicted_anomalies[i, :]
            omega_reward = 0
            overlap_count = [0]
            for j in range(len(real_anomalies)):
                range_r = real_anomalies[j, :]
                omega_reward += self._compute_omega_reward(
                    range_p, range_r, overlap_count, 0
                )
            overlap_reward = (
                self._gamma_function(overlap_count, 0) * omega_reward
            )
            if overlap_count[0] > 0:
                existence_reward = 1
            else:
                existence_reward = 0

            precision += (
                self.alpha_p * existence_reward
                + (1 - self.alpha_p) * overlap_reward
            )
        precision /= len(predicted_anomalies)
        return precision

    def _update_recall(self, real_anomalies, predicted_anomalies):
        recall = 0
        if len(real_anomalies) == 0:
            return 0
        for i in range(len(real_anomalies)):
            omega_reward = 0
            overlap_count = [0]
            range_r = real_anomalies[i, :]
            for j in range(len(predicted_anomalies)):
                range_p = predicted_anomalies[j, :]
                omega_reward += self._compute_omega_reward(
                    range_r, range_p, overlap_count, 1
                )
            overlap_reward = (
                self._gamma_function(overlap_count, 1) * omega_reward
            )

            if overlap_count[0] > 0:
                existence_reward = 1
            else:
                existence_reward = 0

            recall += (
                self.alpha_r * existence_reward
                + (1 - self.alpha_r) * overlap_reward
            )
        recall /= len(real_anomalies)
        return recall

    def _shift(self, arr, num, fill_value=np.nan):
        arr = np.roll(arr, num)
        if num < 0:
            arr[num:] = fill_value
        elif num > 0:
            arr[:num] = fill_value
        return arr

    def _prepare_data(self, values_real, values_pred):

        assert len(values_real) == len(values_pred)

        if self.metric_option == "classic":
            real_anomalies = np.argwhere(values_real == 1).repeat(2, axis=1)
            predicted_anomalies = np.argwhere(values_pred == 1).repeat(
                2, axis=1
            )

        elif self.metric_option == "time-series":
            predicted_anomalies_ = np.argwhere(values_pred == 1).ravel()
            predicted_anomalies_shift_forward = self._shift(
                predicted_anomalies_, 1, fill_value=predicted_anomalies_[0]
            )
            predicted_anomalies_shift_backward = self._shift(
                predicted_anomalies_, -1, fill_value=predicted_anomalies_[-1]
            )
            predicted_anomalies_start = np.argwhere(
                (predicted_anomalies_shift_forward - predicted_anomalies_) != -1
            ).ravel()
            predicted_anomalies_finish = np.argwhere(
                (predicted_anomalies_ - predicted_anomalies_shift_backward)
                != -1
            ).ravel()
            predicted_anomalies = np.hstack(
                [
                    predicted_anomalies_[predicted_anomalies_start].reshape(
                        -1, 1
                    ),
                    predicted_anomalies_[predicted_anomalies_finish].reshape(
                        -1, 1
                    ),
                ]
            )

            real_anomalies_ = np.argwhere(values_real == 1).ravel()
            real_anomalies_shift_forward = self._shift(
                real_anomalies_,
                1,
                fill_value=real_anomalies_[0] if len(real_anomalies_) else 0,
            )
            real_anomalies_shift_backward = self._shift(
                real_anomalies_,
                -1,
                fill_value=real_anomalies_[-1] if len(real_anomalies_) else 0,
            )
            real_anomalies_start = np.argwhere(
                (real_anomalies_shift_forward - real_anomalies_) != -1
            ).ravel()
            real_anomalies_finish = np.argwhere(
                (real_anomalies_ - real_anomalies_shift_backward) != -1
            ).ravel()
            real_anomalies = np.hstack(
                [
                    real_anomalies_[real_anomalies_start].reshape(-1, 1),
                    real_anomalies_[real_anomalies_finish].reshape(-1, 1),
                ]
            )

        elif self.metric_option == "numenta":
            predicted_anomalies = np.argwhere(values_pred == 1).repeat(
                2, axis=1
            )
            real_anomalies_ = np.argwhere(values_real == 1).ravel()
            real_anomalies_shift_forward = self._shift(
                real_anomalies_,
                1,
                fill_value=real_anomalies_[0] if len(real_anomalies_) else 0,
            )
            real_anomalies_shift_backward = self._shift(
                real_anomalies_,
                -1,
                fill_value=real_anomalies_[-1] if len(real_anomalies_) else 0,
            )
            real_anomalies_start = np.argwhere(
                (real_anomalies_shift_forward - real_anomalies_) != -1
            ).ravel()
            real_anomalies_finish = np.argwhere(
                (real_anomalies_ - real_anomalies_shift_backward) != -1
            ).ravel()
            real_anomalies = np.hstack(
                [
                    real_anomalies_[real_anomalies_start].reshape(-1, 1),
                    real_anomalies_[real_anomalies_finish].reshape(-1, 1),
                ]
            )
        return real_anomalies, predicted_anomalies

    def score(self, values_real, values_predicted):
        assert isinstance(values_real, np.ndarray)
        assert isinstance(values_predicted, np.ndarray)

        if not values_predicted.any():
            if not values_real.any():
                return 1.0, 1.0, 1.0
            else:
                return 0.0, 0.0, 0.0

        real_anomalies, predicted_anomalies = self._prepare_data(
            values_real, values_predicted
        )
        precision = self._update_precision(real_anomalies, predicted_anomalies)
        recall = self._update_recall(real_anomalies, predicted_anomalies)
        if precision + recall != 0:
            Fbeta = (
                (1 + self.beta**2)
                * precision
                * recall
                / (self.beta**2 * precision + recall)
            )
        else:
            Fbeta = 0

        return precision, recall, Fbeta
