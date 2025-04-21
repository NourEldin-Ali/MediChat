FROM python:3.9-slim

# Install MLflow and dependencies
RUN pip install mlflow>=2.4.0 psycopg2-binary

# Create directory for MLflow artifacts
RUN mkdir -p /mlflow/artifacts

# Expose port for MLflow UI
EXPOSE 5000

# Start MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]