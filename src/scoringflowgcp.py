from metaflow import FlowSpec, step, Parameter, conda, kubernetes, timeout, retry, catch
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

@conda(libraries={"pandas": "1.5.3", "scikit-learn": "1.2.2", "mlflow": "2.10.0"})
class ScoringFlow(FlowSpec):
    model_name = Parameter("model_name", default="BestModel-MetaflowLab")
    model_stage = Parameter("model_stage", default="Production")

    @kubernetes(cpu=1, memory=2000)
    @timeout(seconds=600)
    @retry(times=2)
    @catch(var="error")
    @step
    def start(self):
        """
        Load input data and MLflow model.
        """
        self.input_data = pd.read_csv("gs://storage-lab8chi-metaflow-default/heart_dataset.csv")
        self.features = self.input_data.drop(columns=['HeartDisease'])
        self.true_labels = self.input_data['HeartDisease']

        mlflow.set_tracking_uri("https://mlflow-tracking-910812384196.us-west2.run.app")
        model_uri = f"models:/{self.model_name}/{self.model_stage}"
        self.model = mlflow.sklearn.load_model(model_uri)

        print(f"Loaded model from: {model_uri}")
        self.next(self.predict)

    @kubernetes(cpu=1, memory=2000)
    @timeout(seconds=300)
    @retry(times=2)
    @catch(var="error")
    @step
    def predict(self):
        """
        Predict using the loaded model.
        """
        self.predictions = self.model.predict(self.features)
        self.results = pd.DataFrame({
            "Actual": self.true_labels,
            "Predicted": self.predictions
        })
        print("Sample predictions:")
        print(self.results.head())
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """
        Evaluate prediction accuracy.
        """
        self.accuracy = accuracy_score(self.true_labels, self.predictions)
        print(f"Scoring Accuracy: {self.accuracy:.4f}")
        self.next(self.end)

    @step
    def end(self):
        """
        Final step: confirm flow completion.
        """
        print("Scoring flow complete.")

if __name__ == '__main__':
    ScoringFlow()
