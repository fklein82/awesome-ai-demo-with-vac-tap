## MLOPS Demo setup with Jupiter, MLFlow, VMware Tanzu Application Catalog & Tanzu Application Platform

This README explains how to install JupyterHub and MLflow on Kubernetes, using VMware Application Catalog (VAC) for a Machine Learning Operations (MLOps) setup. It's meant to help you create an MLOps demo with AI applications on the Tanzu Application Platform (TAP), a Platform as a Service (PaaS) that runs on Kubernetes.


## JupyterHub

JupyterHub is a web-based platform that enables multiple users to collaboratively create and work with Jupyter notebooks on a shared server. It offers a secure and customizable environment, supports multiple users, and is commonly used in education, research, and data analysis for its collaborative and interactive capabilities.

### To install JupyterHub, use the following Helm command:

```bash
helm install jupyterhub oci://harbor.jkolaric.eu/vac-library/charts/ubuntu-22/jupyterhub
```
### After installation, you can access JupyterHub using the following URL:

```bash
export SERVICE_IP=$(kubectl get svc --namespace jupyter jupyterhub-proxy-public --template "{{ range (index .status.loadBalancer.ingress 0) }}{{ . }}{{ end }}")
echo "JupyterHub URL: http://$SERVICE_IP/"
```

### Admin user information:

```bash
echo Admin user: user
echo Password: $(kubectl get secret --namespace jupyter jupyterhub-hub -o jsonpath="{.data['values\.yaml']}" | base64 -d | awk -F: '/password/ {gsub(/[ \t]+/, "", $2);print $2}')
```
### You can access Jupyter notebooks using a URL like this:

```bash
http://20.67.149.113/user/user/lab/tree/opt/bitnami/jupyterhub-singleuser/Untitled.ipynb
```

### Test Jupyter Installation with a Deep Learning Model

The following code essentially demonstrates how to use a pre-trained deep learning model (MobileNetV2) to classify the content of an image fetched from a given URL and visualize the prediction along with the image. You can copy/paste to your Jupyter and execute it.


```code
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import os


# Télécharge une image depuis Internet
def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return img
    else:
        return None

# Prédit le contenu de l'image
def predict_image(model, img):
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    return decode_predictions(predictions, top=1)[0][0]

# URL de l'image
image_url = 'https://www.fklein.me/download/iphone2.jpg'  # Remplacez avec l'URL de l'image que vous souhaitez analyser

# Enregistrement du processus avec MLflow
# Télécharge et analyse l'image
img = download_image(image_url)
if img is not None:
    model = MobileNetV2(weights='imagenet')
    prediction = predict_image(model, img)

    # Affiche l'image et la prédiction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"\nObject: {prediction[1]} \n\n Confiance in the prediction : {prediction[2]*100:.3f}%\n")
    plt.show()
else:
    print("L'image n'a pas pu être téléchargée.")
```

### Prerequisite for MLflow
For using MLflow, install the Python package:
```bash
pip install mlflow
```
Restart the kernel after installation in Jupyter UI.

## MLFlow

MLflow is a tool that helps people who work with machine learning (ML) to do their work more easily. It helps with tracking and organizing ML experiments, packaging code, and deploying ML models. It's useful for managing the entire ML process, from trying out ideas to putting models into real-world applications.

### To install MLflow, use the following Helm command:

```bash
helm install mlflow oci://harbor.jkolaric.eu/vac-library/charts/redhatubi-8/mlflow -n mlflow --create-namespace
```

### Expose the MLflow service:

```bash
export SERVICE_IP=$(kubectl get svc --namespace mlflow mlflow-tracking --template "{{ range (index .status.loadBalancer.ingress 0) }}{{ . }}{{ end }}")
echo "MLflow URL: http://$SERVICE_IP/"
```

### Login credentials:

```bash
echo Username: $(kubectl get secret --namespace mlflow mlflow-tracking -o jsonpath="{ .data.admin-user }" | base64 -d)
echo Password: $(kubectl get secret --namespace mlflow mlflow-tracking -o jsonpath="{.data.admin-password }" | base64 -d)
```

### Use Jupyter to test the MLflow Installation

The following code uses MLflow to track and log experiment information, metrics, and artifacts while performing image classification with a pre-trained MobileNetV2 model. It also saves the model and the downloaded image for later reference and displays the image with the predicted object class and confidence score.

```code
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import mlflow
import os

os.environ['MLFLOW_TRACKING_USERNAME'] = 'user'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'K1aLXzz0QW'

# Configuration de MLflow
mlflow.set_tracking_uri('http://20.67.145.120:80')
mlflow.set_experiment("image_classification_experiment")

# Télécharge une image depuis Internet
def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return img
    else:
        return None

# Prédit le contenu de l'image
def predict_image(model, img):
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    return decode_predictions(predictions, top=1)[0][0]

# URL de l'image
image_url = 'https://www.fklein.me/download/iphone2.jpg'  # Remplacez avec l'URL de l'image que vous souhaitez analyser

# Enregistrement du processus avec MLflow
with mlflow.start_run():
    # Télécharge et analyse l'image
    img = download_image(image_url)
    if img is not None:
        model = MobileNetV2(weights='imagenet')
        prediction = predict_image(model, img)
        
        # Log information
        mlflow.log_param("image_url", image_url)
        mlflow.log_metric("prediction_confidence", float(prediction[2]))

        # Log l'image
        img.save("predicted_image.jpg")
        mlflow.log_artifact("predicted_image.jpg")
        
        # Log an instance of the trained model for later use
        mlflow.tensorflow.log_model(model, artifact_path="object-detection")

        # Affiche l'image et la prédiction
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"\nObject: {prediction[1]} \n\n Confiance in the prediction : {prediction[2]*100:.3f}%\n")
        plt.show()
    else:
        print("L'image n'a pas pu être téléchargée.")
```
## Image Classification Python Accelerator for TAP 🐍📸

The Python accelerator for TAP help you to deploy a serverless image classification function as a workload. The accelerator leverages the buildpacks provided by VMware's open-source Function Buildpacks for Knative project.

The accelerator includes the Python script that we execute before on Jupiter for image classification. It use the MobileNetV2 model and MLflow. It allows you to download an image from the internet, predict its contents, log the prediction and image in MLflow, and display the image with the prediction confidence. This serverless function can be easily integrated into your application or workflow.

### Prequesite :
Have a Tanzu Application Platform installed.

To add the accelerator to your Platform:

```bash
tanzu acc create awesome-python-ai-image-function --git-repo https://github.com/fklein82/awesome-ai-python-function.git --git-branch main --interval
```

Clone this repository to your local development environment:
```bash
git clone <repository-url>
cd python-accelerator-for-tanzu
```

Inside the python-function directory, you will find the func.py file. This Python function is invoked by default and serves as the entry point for your serverless image classification logic.

```bash
python-function
    └── func.py // EDIT THIS FILE
```
You can customize the code inside this file to implement your specific image classification logic.

If you want to explore more code samples for serverless functions that can be deployed within Tanzu Application Platform, you can check out the samples folder.

### Image Classification Function
The core functionality of this accelerator is the image classification function, which performs the following steps:

- Downloads an image from a specified URL.
- Predicts the content of the image using the MobileNetV2 model.
- Logs the prediction and image in MLflow for tracking.
- Displays the image with the prediction confidence.
This function can be integrated into various applications and workflows that require image analysis and classification.

### Deployment
For detailed instructions on how to build, deploy, and test your customized serverless image classification function using Tanzu Application Platform, please refer to the Tanzu website.

To deploy this application on VMware Tanzu Application Platform, follow these steps:

Ensure you have the Tanzu CLI installed and configured with access to your Tanzu Application Platform instance.

Navigate to your project directory:

```bash
cd [your-repo-directory]
```
Use the Tanzu CLI to deploy your application:
```bash
tanzu apps workload create -f config/workload.yaml
```
Monitor the deployment status:
```bash
tanzu apps workload tail awesome-python-ai-image-function --timestamp --since 1h
```
Once deployed, access your application via the URL provided by Tanzu Application Platform. You can find the URL with the following command:
```bash
tanzu apps workload get awesome-python-ai-image-function
```

## Conclusion
This guide explains how to set up JupyterHub and MLflow on Kubernetes using VMware Application Catalog and Tanzu Application Platform for MLOps. It helps manage AI projects more effectively, addressing common challenges in AI and machine learning. The setup enhances cloud-native app deployment and supports collaborative work in AI. 

The blog [Accelerate AI: VAC & TAP in Action](http://127.0.0.1:4000/post/2023/11/15/vac-tap-mlops.html) offers more details, including practical examples of how different roles like Platform Engineers and Data Scientists can use these tools in real-world AI projects.