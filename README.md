"""# 🧹 MNIST Curation: Detecting Ambiguous Digits with FiftyOne

This repository contains the solution for **Dataset Curation Lab 1**, part of the Applied Hands-On Computer Vision curriculum.

The objective of this project is to move beyond standard model training by implementing a **data-centric AI workflow**. Instead of forcing a model to classify poorly written digits, we identify ambiguous samples and reassign them to a new "I Don't Know" (IDK) class to improve model reliability.

---

## 🚀 Workflow Overview

The project pipeline consists of four main stages:

1.  **Embedding & Visualization**: 
    * We generate semantic embeddings for the MNIST dataset using the **CLIP** model.
    * The high-dimensional data is projected into 2D space using **PCA** (Principal Component Analysis) and **UMAP** for visualization within the FiftyOne App.
2.  **Curation Strategy**:
    * We calculate a **Uniqueness** metric to identify outliers.
    * The top **2% most unique/ambiguous images** (often called "Spaghetti Nines") are programmatically tagged as "questionable."
3.  **Dataset Modification**:
    * A new ground truth schema is created with **11 classes** (Digits 0-9 + Class 10: `IDK`).
    * Ambiguous samples are re-labeled to the `IDK` class.
4.  **Model Training**:
    * A custom **LeNet-5 CNN** is modified to accept 11 output classes.
    * The model is trained and evaluated to handle uncertainty gracefully.

---

## 📂 Project Resources

### 1. The Notebook
The complete code for curation, training, and publishing is available here:

* 📄 **File:** `Dataset_Curation_Lab1.ipynb`
* 🚀 **Run:** You can open this notebook directly in Google Colab to utilize free GPU resources.

### 2. The Curated Dataset
The final dataset, including the new `IDK` labels and embedding visualizations, has been hosted on Hugging Face:

* 🤗 **Hugging Face Dataset:** [SudarshanTarmale/mnist-curated]

---

## 🛠️ Tech Stack

* **Data Curation:** [FiftyOne](https://voxel51.com/)
* **Deep Learning:** PyTorch & Torchvision
* **Embeddings:** OpenCLIP
* **Dimensionality Reduction:** UMAP-learn & Scikit-learn
* **Version Control:** Hugging Face Hub

---

## 💻 How to Run

1.  **Environment Setup:**
    Open the notebook. The first cell automatically handles the installation of all required dependencies (FiftyOne, plugins, etc.).

2.  **Execution:**
    Run the cells sequentially. The notebook will:
    * Download MNIST.
    * Compute CLIP embeddings.
    * Perform the uniqueness analysis to find bad data.
    * Train the 11-class classifier.

3.  **Authentication (Optional):**
    To publish your own version of the dataset, you will need a Hugging Face write token. Enter it when prompted in the final section of the notebook.

---

## 📊 Results

By introducing the `IDK` class, the model is no longer forced to make low-confidence guesses on unintelligible handwriting. This mimics real-world "human-in-the-loop" scenarios where ambiguous data should be flagged for review rather than misclassified.
"""

