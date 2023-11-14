# MLOps demo with VAC and TAP

This readme provides comprehensive instructions for setting up JupyterHub and MLflow on your Kubernetes (K8s) system, leveraging the power of the VMware Application Catalog (VAC). This setup is designed to facilitate the creation of a  MLOps (Machine Learning Operations) demonstration and showcase some  AI applications deployed on the Tanzu Application Platform (TAP)—a powerful Platform as a Service (PaaS) solution running seamlessly on top of Kubernetes.

With TAP, you can effortlessly deploy AI applications that utilize a Machine Learning API exposed by Kubeflow. Whether you're an AI enthusiast, data scientist, or a tech fan, this guide empowers you to harness the potential of K8s, VAC, and TAP for an awe-inspiring AI journey.

## What is VMware Application Catalog?

VMware Tanzu Application Catalog (Tanzu Application Catalog) is an enterprise solution that simplifies and secures the use of open-source software components for production. It offers a diverse catalog of rigorously tested open-source applications, automated maintenance, vulnerability insights, and more, streamlining development while ensuring security and compliance.

## What is Tanzu Application Platform?

VMware Tanzu Application Platform is a platform-as-a-service (PaaS) solution that simplifies the deployment and management of cloud-native applications and microservices in a Kubernetes environment. It offers developer productivity, container orchestration, self-service deployment, automation, monitoring, multi-cloud support, and security features.

## MLOps vs DevOps

MLOps and DevOps are related concepts focused on streamlining and automating software development and deployment processes, but they have different areas of application and emphasis. Here's a brief comparison:

### DevOps (Development and Operations):

- Focus: DevOps is a set of practices and cultural philosophies that aim to unify software development (Dev) and IT operations (Ops) teams. Its primary focus is on improving collaboration, communication, and automation throughout the software development lifecycle.

- Goal: The main goal of DevOps is to shorten the development cycle, increase the frequency of software releases, and improve the reliability and quality of software deployments.

- Key Practices: DevOps practices include continuous integration (CI), continuous delivery (CD), automated testing, infrastructure as code (IaC), and monitoring.

- Use Cases: DevOps is widely used in traditional software development, web application development, and software product development.

### MLOps (Machine Learning Operations):

- Focus: MLOps is an extension of DevOps tailored specifically for machine learning and artificial intelligence (AI) projects. It encompasses the end-to-end process of developing, deploying, and managing machine learning models in production.

- Goal: The primary goal of MLOps is to streamline and automate the ML lifecycle, ensuring that machine learning models are developed, tested, and deployed efficiently and effectively.

- Key Practices: MLOps practices include version control for ML models and data, automated model training and evaluation, model deployment automation, model monitoring, and feedback loops for model retraining.

- Use Cases: MLOps is applied to data science and AI projects where machine learning models are integrated into software applications or decision-making processes. It's common in fields like natural language processing, computer vision, and predictive analytics.

### Key Differences:

- Scope: DevOps is a broader discipline that addresses the entire software development lifecycle, while MLOps is specific to machine learning model management.

- Focus on Data: MLOps places a strong emphasis on data versioning, data pipelines, and the unique challenges of managing machine learning models that rely on data.

- Model Lifecycle: MLOps incorporates stages like model training, validation, deployment, and monitoring, which are not typically part of traditional DevOps.

- Tools and Technologies: While both DevOps and MLOps use automation and CI/CD practices, MLOps often involves specialized tools for ML model versioning (e.g., Git LFS), model training (e.g., TensorFlow, PyTorch), and model deployment (e.g., Kubernetes with GPU support).

In summary, DevOps focuses on improving the software development and deployment process in general, while MLOps is tailored specifically to address the unique challenges of managing machine learning models and data in production environments.


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

The code essentially demonstrates how to use a pre-trained deep learning model (MobileNetV2) to classify the content of an image fetched from a given URL and visualize the prediction along with the image.


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


