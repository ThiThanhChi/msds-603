from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class TrainingFlow(FlowSpec):
    random_seed = Parameter("seed", default=42, type=int)

    @step
    def start(self):
        """
        Load and prepare the heart dataset for training.
        """
        data = pd.read_csv("/Users/chinguyen/Desktop/USF-MSDS/msds-603/mlops/data/heart_dataset.csv")

        # Separate features and target label
        X = data.drop(columns=['HeartDisease']).copy()
        y = data['HeartDisease'].copy()

        # Split data into train/test sets using the provided random seed
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )

        # Proceed to training step
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train a Random Forest classifier and evaluate its accuracy.
        """
        # Initialize and train the RandomForest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_seed)
        self.model.fit(self.X_train, self.y_train)

        # Evaluate model performance on the test set
        self.accuracy = accuracy_score(self.y_test, self.model.predict(self.X_test))
        print(f"Model accuracy: {self.accuracy:.4f}")

        # Proceed to model logging step
        self.next(self.log_model)

    @step
    def log_model(self):
        """
        Log and register the trained model to MLflow.
        """
        # Set the MLflow tracking URI  (local)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        # Set or create an MLflow experiment for tracking
        mlflow.set_experiment("mlflow-metaflow-lab-6")

        # Start MLflow run context and log model + metadata
        with mlflow.start_run():
            mlflow.log_param("seed", self.random_seed)
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name="BestModel-MetaflowLab"  # Register under this name
            )

        # Proceed to final step
        self.next(self.end)

    @step
    def end(self):
        """
        Final step: confirm training and logging is complete.
        """
        print("Training complete, model registered.")


# Entrypoint for running the flow
if __name__ == '__main__':
    TrainingFlow()


