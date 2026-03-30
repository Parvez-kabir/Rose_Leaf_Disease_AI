# 🌹 Rose Leaf Disease Detection System
Website live Link:  https://huggingface.co/spaces/parvez-kabir/rose-leaf-disease-detection

A **production-grade, lightweight deep learning web application** for automatic detection of rose leaf diseases using **MobileNetV2**, deployed with **FastAPI** and hosted on **Hugging Face**.
<img width="1420" height="898" alt="image" src="https://github.com/user-attachments/assets/2706586f-17d9-48d3-a4f6-7b94c7417823" />


## 📌 Project Overview

Early detection of plant leaf diseases is essential for improving crop yield and reducing economic loss. Manual inspection is time-consuming and error-prone. This project proposes an **AI-powered rose leaf disease detection system** that classifies leaf images into multiple disease categories with high accuracy while remaining lightweight enough for real-world deployment.

The system combines:

* A **lightweight CNN model (MobileNetV2)**
* A **modern web interface (HTML + CSS)**
* A **FastAPI backend** for real-time inference
* **Hugging Face deployment** for accessibility

---

## 🎯 Objectives

* Detect rose leaf diseases automatically from images
* Maintain high accuracy with low model size
* Enable real-time inference via a web interface
* Ensure easy deployment and scalability

---

## 🧠 Dataset Details

* **Total Images:** 2,457
* **Source:** Collected from *Savar, Gilap Gram*
* **Number of Classes:** 5

### 🌿 Class Names

1. Healthy Leaf
2. Black Spot
3. Downy Mildew
4. Dry Leaf
5. Insect Hole

---

## 🏗️ Model Architecture & Selection

Multiple lightweight CNN models were evaluated:

* ResNet18
* MobileNetV2
* ShuffleNet
* SqueezeNet
* EfficientNetB0

### ✅ Why MobileNetV2?

MobileNetV2 was selected due to its **optimal balance between performance and efficiency**.

**Performance Metrics:**

* **Test Accuracy:** 98.17%
* **Precision:** 98.34%
* **Recall:** 98.17%
* **F1-score:** 98.15%

**Model Complexity:**

* **Parameters:** 2.23 million
* **Model Size:** 8.64 MB

This makes MobileNetV2 ideal for deployment on resource-constrained environments.

---

## 🖥️ System Architecture

```
Frontend (HTML + CSS)
        │
        ▼
FastAPI Backend (app.py)
        │
        ▼
MobileNetV2 Model (.pth)
```

---

## 📂 Project Structure

```
Rose Leaf Disease Detection System/
│
├── app.py
├── models/
│   └── mobilenetv2_final.pth
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

### 🔹 Deep Learning

* PyTorch
* MobileNetV2
* Transfer Learning
* Image Augmentation

### 🔹 Backend

* FastAPI
* Uvicorn

### 🔹 Frontend

* HTML5
* CSS3 (Modern UI)

### 🔹 Deployment

* Hugging Face Spaces

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/rose-leaf-disease-detection.git
cd rose-leaf-disease-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python app.py
```

### 4️⃣ Open in Browser

```
http://127.0.0.1:8000
```

---

## 🌐 Deployment

The project is deployed on **Hugging Face Spaces**, enabling public access to the application without local setup.

---

## 📊 Results Summary

| Metric        | Value   |
| ------------- | ------- |
| Test Accuracy | 98.17%  |
| Precision     | 98.34%  |
| Recall        | 98.17%  |
| Model Size    | 8.64 MB |
| Parameters    | 2.23M   |

---

## 🔮 Future Work

* Add Grad-CAM for explainable AI
* Extend dataset with more diseases
* Convert model to ONNX / TensorRT
* Mobile app integration

---

## 👨‍💻 Author

**Md. Parvez Kabir**
BSc in Computer Science & Engineering
Daffodil International University

---

## 📜 License

This project is intended for **research and educational purposes**.

---

⭐ If you find this project helpful, consider giving it a star on GitHub!
