# 🌿 Leaf vs Non-Leaf Detection

A deep learning–based image classification project that accurately identifies whether an input image contains a **leaf** or **non-leaf** object.  
This project uses a **Convolutional Neural Network (CNN)** trained on a labeled image dataset and deployed using **Streamlit** for easy interaction.

---

## 📌 Project Overview

The goal of this project is to build an intelligent system capable of distinguishing **leaf images from non-leaf images**.  
Such a system can be useful in:
- Agriculture automation  
- Plant disease detection pipelines  
- Environmental monitoring  
- Smart farming applications  

Due to the **large size of the trained model**, it has been uploaded separately on Google Drive.

---

## 🧠 Model Details

- **Model Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Input:** Image
- **Output:**  
  - Leaf  
  - Non-Leaf  

🔗 **Trained Model (Google Drive):**  
👉 https://drive.google.com/file/d/1JnirdqHbvxPLCHIrbPuPufwChjvdzI4L/view?usp=drive_link

> ⚠️ The model is hosted on Google Drive due to GitHub file size limitations.

---

## 📂 Dataset

- **Source:** Kaggle  
- **Name:** Leaf vs Non-Leaf Images  
- **Classes:** Leaf, Non-Leaf  

🔗 **Dataset Link:**  
👉 https://www.kaggle.com/datasets/robiulhasanjisan/leaf-vs-non-leaf-images

---

## 🧪 Model Training & Reference Code

The model training, preprocessing, and evaluation logic is based on the following Kaggle notebook:

🔗 **Reference Notebook:**  
👉 https://www.kaggle.com/code/sonalshinde123/leaf-vs-non-leaf-detection

This notebook includes:
- Image preprocessing
- CNN architecture
- Model training
- Performance evaluation

---

## 🖥️ Web Application (Streamlit)

A **Streamlit-based UI** is used for real-time prediction.

### Features:
- Upload an image
- Automatic preprocessing
- Real-time prediction (Leaf / Non-Leaf)
- Clean and user-friendly interface

---

## ⚙️ Tech Stack

- **Programming Language:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Frontend:** Streamlit  
- **Image Processing:** OpenCV, PIL  
- **Dataset:** Kaggle  

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Leaf-vs-Non-Leaf-Detection.git
cd Leaf-vs-Non-Leaf-Detection
```
📊 Results

- High accuracy in distinguishing leaf vs non-leaf images
- Robust performance on unseen images
- Lightweight and deployable model

---

📌 Future Improvements
- Add multi-class leaf classification
- Integrate plant disease detection
- Deploy as a cloud-based web application
- Improve accuracy using transfer learning (ResNet, MobileNet)
---
