from metaflow import FlowSpec, step, Parameter, conda_base, resources, kubernetes, timeout, retry, catch

@conda_base(
    python="3.10",
    libraries={
        "pandas": "2.1.4",
        "scikit-learn": "1.3.2",
        "mlflow": "2.10.1",
        "gcsfs": "2023.6.0"
    }
    # Remove the disabled=True line
)
class TrainingFlowGCP(FlowSpec):
    random_seed = Parameter("seed", default=42, type=int)
    
    @step
    def start(self):
        import pandas as pd
        data = pd.read_csv("gs://storage-lab8chi-metaflow-default/heart_dataset.csv")
        X = data.drop(columns=['HeartDisease']).copy()
        y = data['HeartDisease'].copy()
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed)
        self.next(self.train_model)
    
    @resources(cpu=2, memory=4096)
    @kubernetes(
        image="registry.hub.docker.com/yourusername/metaflow-mlflow:latest",
        cpu=2,
        memory=4096
    )
    @timeout(seconds=600)
    @retry(times=2)
    @catch(var="error")
    @step
    def train_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_seed)
        self.model.fit(self.X_train, self.y_train)
        self.accuracy = accuracy_score(self.y_test, self.model.predict(self.X_test))
        print(f"Model accuracy: {self.accuracy:.4f}")
        self.next(self.log_model)
    
    @step
    def log_model(self):
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri("https://mlflow-tracking-910812384196.us-west2.run.app")
        mlflow.set_experiment("mlflow-metaflow-lab-6")
        with mlflow.start_run():
            mlflow.log_param("seed", self.random_seed)
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name="BestModel-MetaflowLab"
            )
        self.next(self.end)
    
    @step
    def end(self):
        print("Training complete, model registered.")

if __name__ == "__main__":
    TrainingFlowGCP()
