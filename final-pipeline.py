import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score, average_precision_score
from scipy.optimize import minimize
from xgboost import XGBRegressor
from tqdm import tqdm
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class Seasonal:
    def __init__(self):
        self.daily_weight = None
        self.weekly_weight = None

    def fit(self, train_data, val_data):
        train_arr = train_data.values if hasattr(train_data, 'values') else train_data
        val_arr = val_data.values if hasattr(val_data, 'values') else val_data

        def loss(weights):
            daily_w, weekly_w = weights[0], weights[1]
            preds = []
            for t in range(len(val_arr)):
                daily_idx = len(train_arr) - 24 + (t % 24)
                weekly_idx = len(train_arr) - 168 + (t % 168)
                daily_val = train_arr[daily_idx] if daily_idx >= 0 else train_arr[-1]
                weekly_val = train_arr[weekly_idx] if weekly_idx >= 0 else train_arr[-1]
                pred = daily_w * daily_val + weekly_w * weekly_val
                preds.append(pred)
            preds = np.array(preds)
            return mean_absolute_error(val_arr.flatten(), preds.flatten())

        result = minimize(loss, [0.6, 0.4], bounds=[(0, 1), (0, 1)], method='L-BFGS-B')
        self.daily_weight, self.weekly_weight = result.x
        print(f"Optimized weights: daily={self.daily_weight:.3f}, weekly={self.weekly_weight:.3f}")
        return self

    def predict(self, train_data, steps=168):
        train_arr = train_data.values if hasattr(train_data, 'values') else train_data
        preds = []
        for t in range(steps):
            daily_idx = len(train_arr) - 24 + (t % 24)
            weekly_idx = len(train_arr) - 168 + (t % 168)
            daily_val = train_arr[daily_idx] if daily_idx >= 0 else train_arr[-1]
            weekly_val = train_arr[weekly_idx] if weekly_idx >= 0 else train_arr[-1]
            pred = self.daily_weight * daily_val + self.weekly_weight * weekly_val
            preds.append(pred)
        return np.array(preds)


class Ensemble:
    def __init__(self):
        self.seasonal_model = None
        self.lr_model = None
        self.xgb_models = None
        self.daily_pattern = None
        self.train_data = None
        self.seasonal_weights = None
        self.ensemble_weights = None
        self.threshold = 0.5
        self.validation_performance = {}
        self.metadata = {}

    def predict(self, train_data, steps=168):
        seasonal_pred = self.seasonal_model.predict(train_data, steps)
        lr_pred = self._predict_lr(train_data, steps)
        xgb_pred = self._predict_xgb(train_data, steps)
        w_s, w_l, w_x = self.ensemble_weights
        ensemble_pred = (w_s * seasonal_pred + w_l * lr_pred + w_x * xgb_pred)
        if self.daily_pattern is not None:
            for t in range(steps):
                hour = t % 24
                ensemble_pred[t] = 0.85 * ensemble_pred[t] + 0.15 * self.daily_pattern[hour]
        return np.maximum(ensemble_pred, 0)

    def _predict_lr(self, train_data, steps=168):
        train_arr = train_data.values if hasattr(train_data, 'values') else train_data
        lr_pred = np.zeros((steps, train_arr.shape[1]))
        last_window = train_arr[-24:].copy()
        for t in range(steps):
            features = last_window.flatten()
            features = np.append(features, last_window.mean(axis=0))
            features = np.append(features, last_window.std(axis=0))
            pred = self.lr_model.predict(features.reshape(1, -1))[0]
            lr_pred[t] = pred
            last_window = np.vstack([last_window[1:], pred])
        return np.maximum(lr_pred, 0)

    def _predict_xgb(self, train_data, steps=168):
        train_arr = train_data.values if hasattr(train_data, 'values') else train_data
        xgb_pred = np.zeros((steps, train_arr.shape[1]))
        for beam_idx, model in self.xgb_models.items():
            beam_data = train_arr[:, beam_idx]
            beam_log = np.log1p(beam_data)
            last_168 = beam_log[-168:]
            preds_log = []
            current = last_168.copy()
            for _ in range(steps):
                features = current.tolist()
                features.append(current[-24:].mean())
                features.append(current.mean())
                features.append(current[-24:].std())
                pred_log = model.predict(np.array(features).reshape(1, -1))[0]
                preds_log.append(pred_log)
                current = np.roll(current, -1)
                current[-1] = pred_log
            xgb_pred[:, beam_idx] = np.expm1(preds_log)
        return np.maximum(xgb_pred, 0)

    def calculate_metrics(self, y_true, y_pred, threshold=0.5):
        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        y_true_flat = y_true_binary.flatten()
        y_pred_flat = y_pred_binary.flatten()
        metrics = {
            'precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
            'recall': recall_score(y_true_flat, y_pred_flat, zero_division=0),
            'f1_score': f1_score(y_true_flat, y_pred_flat, zero_division=0),
            'average_precision': average_precision_score(y_true_flat, y_pred_flat),
            'threshold': threshold,
            'actual_active_rate': y_true_binary.mean(),
            'pred_active_rate': y_pred_binary.mean()
        }
        per_beam_precision = []
        per_beam_recall = []
        per_beam_f1 = []
        for beam_idx in range(y_true.shape[1]):
            p = precision_score(y_true_binary[:, beam_idx], y_pred_binary[:, beam_idx], zero_division=0)
            r = recall_score(y_true_binary[:, beam_idx], y_pred_binary[:, beam_idx], zero_division=0)
            f = f1_score(y_true_binary[:, beam_idx], y_pred_binary[:, beam_idx], zero_division=0)
            per_beam_precision.append(p)
            per_beam_recall.append(r)
            per_beam_f1.append(f)
        metrics['per_beam_precision_mean'] = np.mean(per_beam_precision)
        metrics['per_beam_precision_std'] = np.std(per_beam_precision)
        metrics['per_beam_recall_mean'] = np.mean(per_beam_recall)
        metrics['per_beam_recall_std'] = np.std(per_beam_recall)
        metrics['per_beam_f1_mean'] = np.mean(per_beam_f1)
        metrics['per_beam_f1_std'] = np.std(per_beam_f1)
        return metrics

    def save(self, filepath='ensemble_model.joblib', compress=3):
        print(f"Saving ensemble to {filepath}...")
        joblib.dump(self, filepath, compress=compress)
        size = os.path.getsize(filepath) / 1024 / 1024
        print(f"Saved. File size: {size:.2f} MB")
        return self

    @classmethod
    def load(cls, filepath='ensemble_model.joblib'):
        print(f"Loading ensemble from {filepath}...")
        ensemble = joblib.load(filepath)
        print("Loaded successfully!")
        return ensemble


class Trainer:
    def __init__(self, train_df, val_df=None):
        self.train_df = train_df
        self.val_df = val_df
        self.train_data = train_df.values.astype(np.float32)
        if val_df is not None:
            self.val_data = val_df.values.astype(np.float32)
        else:
            # Используем последние 168 семплов для валидации
            self.val_data = self.train_data[-168:].copy()
            self.train_data = self.train_data[:-168].copy()
            self.train_df = self.train_df.iloc[:-168]
        # Удаляем лишние колонки если есть
        if self.train_data.shape[1] == 2881:
            self.train_data = self.train_data[:, 1:]
            self.val_data = self.val_data[:, 1:]
            self.train_df = self.train_df.iloc[:, 1:]
            if self.val_df is not None:
                self.val_df = self.val_df.iloc[:, 1:]
        self.n_beams = self.train_data.shape[1]
        self.n_timesteps = self.train_data.shape[0]
        self.seasonal_model = None
        self.lr_model = None
        self.xgb_models = {}
        self.daily_pattern = None
        self.ensemble_weights = None
        self.seasonal_pred = None
        self.lr_pred = None
        self.xgb_pred = None
        self.ensemble_pred = None

    def train_seasonal(self):
        print("Training Seasonal Model...")
        self.seasonal_model = Seasonal()
        self.seasonal_model.fit(self.train_df, self.val_data)
        self.seasonal_pred = self.seasonal_model.predict(self.train_df, steps=len(self.val_data))
        return self

    def train_linear_regression(self):
        print("Training Linear Regression...")

        def prepare_lr_features(data, window=24):
            X, y = [], []
            for t in range(window, len(data)):
                features = data[t-window:t].flatten()
                features = np.append(features, data[t-window:t].mean(axis=0))
                features = np.append(features, data[t-window:t].std(axis=0))
                X.append(features)
                y.append(data[t])
            return np.array(X), np.array(y)

        X_train, y_train = prepare_lr_features(self.train_data, window=24)
        self.lr_model = Ridge(alpha=0.1, random_state=42)
        self.lr_model.fit(X_train, y_train)
        self.lr_pred = np.zeros((len(self.val_data), self.n_beams))
        last_window = self.train_data[-24:].copy()
        for t in range(len(self.val_data)):
            features = last_window.flatten()
            features = np.append(features, last_window.mean(axis=0))
            features = np.append(features, last_window.std(axis=0))
            pred = self.lr_model.predict(features.reshape(1, -1))[0]
            self.lr_pred[t] = pred
            last_window = np.vstack([last_window[1:], pred])
        self.lr_pred = np.maximum(self.lr_pred, 0)
        return self

    def train_xgboost(self):
        print("Training XGBoost Models...")
        self.xgb_pred = np.zeros((len(self.val_data), self.n_beams))
        self.xgb_models = {}
        for beam_idx in tqdm(range(self.n_beams), desc="Training XGBoost"):
            beam_data = self.train_data[:, beam_idx]
            if beam_data.std() < 1e-6:
                print(f"Skipping beam {beam_idx} (constant/zero values)")
                continue
            beam_log = np.log1p(beam_data)
            X_beam, y_beam = [], []
            for t in range(168, len(beam_log)):
                features = beam_log[t-168:t].tolist()
                features.append(beam_log[t-24:t].mean())
                features.append(beam_log[t-168:t].mean())
                features.append(beam_log[t-24:t].std())
                X_beam.append(features)
                y_beam.append(beam_log[t])
            if len(X_beam) < 100:
                print(f"Skipping beam {beam_idx} (insufficient samples: {len(X_beam)})")
                continue
            X_beam = np.array(X_beam)
            y_beam = np.array(y_beam)
            model = XGBRegressor(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                n_jobs=1,
                verbosity=0
            )
            model.fit(X_beam, y_beam)
            self.xgb_models[beam_idx] = model
            last_168 = beam_log[-168:]
            preds_log = []
            current = last_168.copy()
            for _ in range(len(self.val_data)):
                features = current.tolist()
                features.append(current[-24:].mean())
                features.append(current.mean())
                features.append(current[-24:].std())
                pred_log = model.predict(np.array(features).reshape(1, -1))[0]
                preds_log.append(pred_log)
                current = np.roll(current, -1)
                current[-1] = pred_log
            self.xgb_pred[:, beam_idx] = np.expm1(preds_log)
        print(f"Successfully trained XGBoost for {len(self.xgb_models)} out of {self.n_beams} beams")
        return self

    def compute_daily_pattern(self):
        print("Computing daily pattern...")
        self.daily_pattern = np.zeros((24, self.n_beams))
        for hour in range(24):
            hour_indices = np.arange(hour, len(self.train_data), 24)
            if len(hour_indices) > 0:
                self.daily_pattern[hour] = self.train_data[hour_indices].mean(axis=0)
        return self

    def optimize_ensemble_weights(self):
        print("Optimizing ensemble weights...")
        val_actual = self.val_data

        def ensemble_loss(weights):
            w_s, w_l, w_x = weights
            w_sum = w_s + w_l + w_x
            if w_sum == 0:
                return 1e6
            w_s, w_l, w_x = w_s/w_sum, w_l/w_sum, w_x/w_sum
            pred = (w_s * self.seasonal_pred + w_l * self.lr_pred + w_x * self.xgb_pred)
            return mean_absolute_error(val_actual.flatten(), pred.flatten())

        result = minimize(ensemble_loss, [0.2, 0.3, 0.5],
                          bounds=[(0, 1), (0, 1), (0, 1)],
                          method='L-BFGS-B')
        w_s, w_l, w_x = result.x
        w_sum = w_s + w_l + w_x
        self.ensemble_weights = [w_s/w_sum, w_l/w_sum, w_x/w_sum]
        print(f"Optimal weights: Seasonal={self.ensemble_weights[0]:.3f}, "
              f"LR={self.ensemble_weights[1]:.3f}, XGB={self.ensemble_weights[2]:.3f}")
        return self

    def create_ensemble_predictions(self):
        print("Creating final ensemble predictions...")
        w_s, w_l, w_x = self.ensemble_weights
        self.ensemble_pred = (w_s * self.seasonal_pred +
                              w_l * self.lr_pred +
                              w_x * self.xgb_pred)

        for t in range(len(self.ensemble_pred)):
            hour = t % 24
            self.ensemble_pred[t] = 0.85 * self.ensemble_pred[t] + 0.15 * self.daily_pattern[hour]
        self.ensemble_pred = np.maximum(self.ensemble_pred, 0)
        return self

    def build_ensemble_object(self):
        ensemble = Ensemble()
        ensemble.seasonal_model = self.seasonal_model
        ensemble.lr_model = self.lr_model
        ensemble.xgb_models = self.xgb_models
        ensemble.daily_pattern = self.daily_pattern
        ensemble.ensemble_weights = self.ensemble_weights
        ensemble.threshold = 0.5
        metrics = ensemble.calculate_metrics(self.val_data, self.ensemble_pred, threshold=0.5)
        metrics['mae'] = mean_absolute_error(self.val_data.flatten(), self.ensemble_pred.flatten())
        ensemble.validation_performance = metrics
        ensemble.metadata = {
            'n_beams': self.n_beams,
            'n_timesteps': self.n_timesteps,
            'model_version': 'v1.0',
            'xgb_models_trained': len(self.xgb_models),
            'ensemble_weights': self.ensemble_weights,
            'validation_size': len(self.val_data)
        }
        return ensemble

    def train_complete(self):
        self.train_seasonal()
        self.train_linear_regression()
        self.train_xgboost()
        self.compute_daily_pattern()
        self.optimize_ensemble_weights()
        self.create_ensemble_predictions()
        return self.build_ensemble_object()


def evaluate_on_test(ensemble, test_data, train_data_for_prediction):
    print(f"Generating predictions for {len(test_data)} timesteps...")
    test_predictions = ensemble.predict(train_data_for_prediction, steps=len(test_data))
    test_metrics = ensemble.calculate_metrics(test_data, test_predictions, threshold=0.5)
    test_metrics['mae'] = mean_absolute_error(test_data.flatten(), test_predictions.flatten())
    print("Test Set Metrics:")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Average Precision: {test_metrics['average_precision']:.4f}")
    print(f"Actual Active Rate: {test_metrics['actual_active_rate']:.4f}")
    print(f"Predicted Active Rate: {test_metrics['pred_active_rate']:.4f}")
    return test_metrics, test_predictions


if __name__ == "__main__":
    print("Loading data...")
    train_df = pd.read_csv('data/MR_number_train_0w-5w.csv.zip', index_col=0)
    test_df = pd.read_csv('data/MR_number_test_5w-6w.csv.zip', index_col=0)
    val_size = 168  # 1 week of validation
    train_size = len(train_df) - val_size
    train_split_df = train_df.iloc[:train_size]
    val_split_df = train_df.iloc[train_size:]
    print("Data split summary:")
    print(f"Training period: {len(train_split_df)} timesteps")
    print(f"Validation period: {len(val_split_df)} timesteps")
    print(f"Test period: {len(test_df)} timesteps")
    print(f"Number of beams: {train_split_df.shape[1]}")
    trainer = Trainer(
        train_df=train_split_df,
        val_df=val_split_df  # Валидацию берем с конца обучающей выборки
    )
    ensemble = trainer.train_complete()
    print("Validation performance (on held-out validation set)")
    print(f"MAE: {ensemble.validation_performance['mae']:.4f}")
    print(f"Precision: {ensemble.validation_performance['precision']:.4f}")
    print(f"Recall: {ensemble.validation_performance['recall']:.4f}")
    print(f"F1 Score: {ensemble.validation_performance['f1_score']:.4f}")
    print(f"Average Precision: {ensemble.validation_performance['average_precision']:.4f}")
    ensemble.save('models/ensemble_model.joblib', compress=9)
    print("Evaluate in test data (post - training)")
    test_metrics, test_predictions = evaluate_on_test(
        ensemble=ensemble,
        test_data=test_df.values.astype(np.float32),
        train_data_for_prediction=train_df  # Нужно передавать все данные на которых было обучение в контекст.
    )
    test_predictions_df = pd.DataFrame(
        test_predictions,
        index=test_df.index,
        columns=test_df.columns
    )
    test_predictions_df.to_csv('models/test_predictions.csv')