# Deployment Guide

This document describes how to deploy, test, monitor, and rollback the customer churn prediction FastAPI service.

---

## 1. Local Container Deployment (Docker)

To build and run the serving endpoint locally:

### Step 1: Build the Docker Image
Navigate to the root directory containing the `Dockerfile` and execute:
```bash
docker build -t churn-serving-api:v1.0 -f src/serving/Dockerfile .
```

### Step 2: Run the Docker Container
Run the container, binding port 8000:
```bash
docker run -d -p 8000:8000 --name churn-api-container \
  -e OPENAI_API_KEY="your-openai-api-key" \
  churn-serving-api:v1.0
```

### Step 3: Verify the Running API
Check container status and logs:
```bash
docker ps
docker logs churn-api-container
```

---

## 2. Production Cloud Deployment (Azure ML Endpoint)

To deploy the service as an online endpoint in Azure Machine Learning:

### Step 1: Create Endpoint Configuration
Create a YAML definition `endpoint.yml`:
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: customer-churn-endpoint
auth_mode: key
```
Deploy the endpoint:
```bash
az ml online-endpoint create --file endpoint.yml
```

### Step 2: Create Deployment Configuration
Create a YAML deployment definition `deployment.yml`:
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: churn-predictor-v1
endpoint_name: customer-churn-endpoint
model: azureml:churn_predictor:1
code_configuration:
  code: ./src/serving
  entry_script: main.py
environment: azureml:churn-env:1
instance_type: Standard_DS3_v2
instance_count: 1
```
Deploy the model:
```bash
az ml online-deployment create --file deployment.yml --all-traffic
```

---

## 3. API Client Testing

### POST `/predict` Example Request
Query the running microservice to predict churn for a customer:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customerID": "0001-ABCD"}'
```

Response JSON structure:
```json
{
  "customer_id": "0001-ABCD",
  "churn_score": 0.825,
  "confidence": 0.95,
  "top_3_shap_drivers": ["contract_type_encoded", "tenure", "TechSupport"],
  "recommended_action": "This customer is at high risk of churn due to Month-to-Month contract type. Playbooks recommend contract transition with free add-on trials..."
}
```

### POST `/outcome` Example Request
Log the result of a retention intervention:
```bash
curl -X POST http://127.0.0.1:8000/outcome \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "0001-ABCD", "intervention_type": "Loyalty Discount", "outcome": 1}'
```

---

## 4. Rollback Plan
If v2 exhibits memory leak or score drift:
1. Transition traffic back to the prior stable version:
   ```bash
   az ml online-endpoint update --name customer-churn-endpoint --traffic "churn-predictor-v1=100, churn-predictor-v2=0"
   ```
2. Inspect logs and update monitoring configurations.
