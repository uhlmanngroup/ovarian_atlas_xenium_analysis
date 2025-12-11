import os
from typing import Any, Dict
import pickle

import matplotlib.pyplot as plt
import numpy
import optuna
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


class RunClassifier:
    def __init__(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        kfolds: int = 5,
        random_state: int = 42,
        test_size: float = 0.2,
        val_size: float = 0.2,
    ):
        self.random_state = random_state
        self.kfolds = kfolds

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        self.scaler = StandardScaler()

        # First split: separate test set
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print("Data split completed:")
        print(f"Training set: {self.x_train.shape[0]} samples")
        print(f"Test set: {self.x_test.shape[0]} samples")

    def tune_hyperparameters(self, n_trials: int = 100) -> Dict[str, Any]:
        def objective(trial):
            # Define hyperparameter search space
            params = {
                "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 2000),
                "random_state": self.random_state,
            }

            # Create pipeline
            pipeline = Pipeline(
                [
                    ("scaler", self.scaler),
                    ("classifier", LogisticRegression(**params)),
                ]
            )

            # Perform cross-validation
            kfold = KFold(
                n_splits=self.kfolds, shuffle=True, random_state=self.random_state
            )
            scores = cross_val_score(
                pipeline, self.x_train, self.y_train, cv=kfold, scoring="f1_weighted"
            )

            return scores.mean()

        # Run optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params

        print("\nBest hyperparameters found:")
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        print(f"Best CV F1 Score: {study.best_value:.4f}")

        return self.best_params

    def train_model(self, **model_params) -> None:
        """
        Train the logistic regression model with given parameters.

        Args:
            **model_params: Parameters for LogisticRegression
        """
        # Set default parameters if none provided
        default_params = {"random_state": self.random_state, "max_iter": 1000}
        default_params.update(model_params)

        # Create pipeline with scaling and logistic regression
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**default_params)),
            ]
        )

        # Fit the model
        self.pipeline.fit(self.x_train, self.y_train)
        self.model = self.pipeline.named_steps["classifier"]

        # Evaluate on training and test sets
        self._evaluate_model()

    def train(self) -> None:
        """Train model with the best parameters found during tuning."""
        if self.best_params is None:
            raise ValueError(
                "No best parameters found. Run hyperparameter tuning first."
            )

        self.train_model(**self.best_params)
        print("Model trained with best hyperparameters.")

    def _evaluate_model(self) -> None:
        """Evaluate model performance on train and test sets."""
        # Training scores
        y_train_pred = self.pipeline.predict(self.x_train)
        y_train_proba = self.pipeline.predict_proba(self.x_train)[:, 1]
        self.train_scores = self._calculate_metrics(
            self.y_train, y_train_pred, y_train_proba
        )

        y_test_pred = self.pipeline.predict(self.x_test)
        y_test_proba = self.pipeline.predict_proba(self.x_test)[:, 1]
        self.test_scores = self._calculate_metrics(
            self.y_test, y_test_pred, y_test_proba
        )

    def _calculate_metrics(
        self, y_true: numpy.ndarray, y_pred: numpy.ndarray, y_proba: numpy.ndarray
    ) -> Dict[str, float]:
        """Calculate various classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted"),
            "roc_auc": (
                roc_auc_score(y_true, y_proba) if len(numpy.unique(y_true)) == 2 else None
            ),
        }

    def save_model(
        self,
        save_dir: str = ".",
    ):
        os.makedirs(save_dir, exist_ok=True)
        model_dict = {"model": self.model, "scaler": self.scaler, "best_params": self.best_params}
        pickle.dump(model_dict, open(os.path.join(save_dir, "model_dict.pkl"), "wb"))

    def visualize_results(self, save_dir: str = "./visualizations") -> None:
        """
        Create and save visualizations of model performance.

        Args:
            save_dir (str): Directory to save the generated plots.
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        os.makedirs(save_dir, exist_ok=True)

        # 1. Performance Metrics Comparison
        plt.figure(figsize=(8, 6))
        metrics = ["accuracy", "precision", "recall", "f1"]
        train_vals = [self.train_scores[m] for m in metrics]
        test_vals = [self.test_scores[m] for m in metrics]
        x = numpy.arange(len(metrics))
        width = 0.25

        plt.bar(x - width, train_vals, width, label="Train", alpha=0.8)
        plt.bar(x + width, test_vals, width, label="Test", alpha=0.8)
        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title("Model Performance Comparison")
        plt.xticks(x, [m.capitalize() for m in metrics])
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "performance_comparison.png"))
        plt.close()

        # 2. Confusion Matrix (Validation Set)
        y_test_pred = self.pipeline.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_test_pred, normalize="true")
        cm = numpy.round(cm, 2)
        labels = self.label_encoder.inverse_transform(numpy.arange(cm.shape[0]))
        n_labels = len(labels)

        # Dynamic sizing logic
        cell_size = 0.6  # inches per cell
        min_size, max_size = 6, 20
        fig_width = min(max(n_labels * cell_size, min_size), max_size)
        fig_height = fig_width  # make square

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=True, yticklabels=True, ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix (Test set)")

        ax.xaxis.set_ticklabels(
            self.label_encoder.inverse_transform(numpy.arange(cm.shape[1]))
        )
        ax.yaxis.set_ticklabels(
            self.label_encoder.inverse_transform(numpy.arange(cm.shape[0]))
        )

        # label_fontsize = 10 if n_labels <= 10 else max(6, 14 - n_labels // 5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()

        # 3. ROC Curve (if binary classification)
        if len(numpy.unique(self.y_test)) == 2:
            y_test_proba = self.pipeline.predict_proba(self.x_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_test_proba)
            auc_score = roc_auc_score(self.y_test, y_test_proba)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "roc_curve.png"))
            plt.close()

        # 4. Feature Importance (Coefficients)
        if hasattr(self.model, "coef_"):
            coef = (
                self.model.coef_[0]
                if len(self.model.coef_) == 1
                else self.model.coef_.ravel()
            )
            feature_names = [f"Feature_{i}" for i in range(len(coef))]
            coef_abs = numpy.abs(coef)
            sorted_idx = numpy.argsort(coef_abs)[-10:]

            plt.figure(figsize=(8, 6))
            plt.barh(range(len(sorted_idx)), coef[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.xlabel("Coefficient Value")
            plt.title("Top 10 Feature Coefficients")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "feature_importance.png"))
            plt.close()
            
        elif hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_names = getattr(self, "feature_names", [f"Feature_{i}" for i in range(len(importances))])

            top_n = 20

            # Get top N features
            sorted_idx = numpy.argsort(importances)[-top_n:]
            top_features = [feature_names[i] for i in sorted_idx]
            top_importances = importances[sorted_idx]

            fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))  # Dynamic height
            ax.barh(range(top_n), top_importances, align="center")
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_features)
            ax.set_xlabel("Feature Importance")
            ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "rf_feature_importance.png"))
            plt.close(fig)

        # 5. Learning Curve
        train_sizes = numpy.linspace(0.1, 1.0, 10)
        train_scores_mean = []
        test_scores_mean = []

        for size in train_sizes:
            n_samples = int(size * len(self.x_train))
            x_subset = self.x_train[:n_samples]
            y_subset = self.y_train[:n_samples]

            model = clone(self.model)
            temp_pipeline = Pipeline([
                ("scaler", self.scaler),
                ("classifier", model),
            ])
            temp_pipeline.fit(x_subset, y_subset)

            train_pred = temp_pipeline.predict(x_subset)
            test_pred = temp_pipeline.predict(self.x_test)

            train_scores_mean.append(accuracy_score(y_subset, train_pred))
            test_scores_mean.append(accuracy_score(self.y_test, test_pred))

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores_mean, "o-", label="Training Score")
        plt.plot(train_sizes, test_scores_mean, "o-", label="Test Score")
        plt.xlabel("Training Set Size (Proportion)")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "learning_curve.png"))
        plt.close()


class LRClassifier(RunClassifier):
    def __init__(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        kfolds: int = 5,
        random_state: int = 42,
        test_size: float = 0.2,
        val_size: float = 0.2,
    ):
        super().__init__(
            x,
            y,
            kfolds,
            random_state,
            test_size,
            val_size,
        )

    def tune_hyperparameters(self, n_trials: int = 100) -> Dict[str, Any]:
        def objective(trial):
            # Define hyperparameter search space
            params = {
                "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 2000),
                "random_state": self.random_state,
            }

            # Create pipeline
            pipeline = Pipeline(
                [
                    ("scaler", self.scaler),
                    ("classifier", LogisticRegression(**params)),
                ]
            )

            # Perform cross-validation
            kfold = KFold(
                n_splits=self.kfolds, shuffle=True, random_state=self.random_state
            )
            scores = cross_val_score(
                pipeline, self.x_train, self.y_train, cv=kfold, scoring="f1_weighted"
            )

            return scores.mean()

        # Run optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params

        print("\nBest hyperparameters found:")
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        print(f"Best CV F1 Score: {study.best_value:.4f}")

        return self.best_params

    def train_model(self, **model_params) -> None:
        """
        Train the logistic regression model with given parameters.

        Args:
            **model_params: Parameters for LogisticRegression
        """
        # Set default parameters if none provided
        default_params = {"random_state": self.random_state, "max_iter": 1000}
        default_params.update(model_params)

        # Create pipeline with scaling and logistic regression
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**default_params)),
            ]
        )

        # Fit the model
        self.pipeline.fit(self.x_train, self.y_train)
        self.model = self.pipeline.named_steps["classifier"]

        # Evaluate on training and test sets
        self._evaluate_model()


class RFClassifier(RunClassifier):
    def __init__(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        kfolds: int = 5,
        random_state: int = 42,
        test_size: float = 0.2,
        val_size: float = 0.2,
    ):
        super().__init__(
            x,
            y,
            kfolds,
            random_state,
            test_size,
            val_size,
        )

    def tune_hyperparameters(self, n_trials: int = 100) -> Dict[str, Any]:
        def objective(trial):
            # Define hyperparameter search space for RandomForest
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state
            }
            
            # Add max_samples parameter only if bootstrap is True
            if params['bootstrap']:
                params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
            
            # Suggest criterion
            params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            
            # Class weight handling for imbalanced datasets
            params['class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])
            
            # Create pipeline (RandomForest doesn't typically need scaling, but keeping for consistency)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(**params))
            ])
            
            # Perform cross-validation
            kfold = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(pipeline, self.x_train, self.y_train, cv=kfold, scoring='f1_weighted')
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize', 
                                sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        
        print(f"\nBest hyperparameters found:")
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        print(f"Best CV F1 Score: {study.best_value:.4f}")
        
        return self.best_params

    def train_model(self, **model_params) -> None:
        """
        Train the logistic regression model with given parameters.

        Args:
            **model_params: Parameters for LogisticRegression
        """
        # Set default parameters if none provided
        default_params = {"random_state": self.random_state}
        default_params.update(model_params)

        # Create pipeline with scaling and logistic regression
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(**default_params)),
            ]
        )

        # Fit the model
        self.pipeline.fit(self.x_train, self.y_train)
        self.model = self.pipeline.named_steps["classifier"]

        # Evaluate on training and test sets
        self._evaluate_model()