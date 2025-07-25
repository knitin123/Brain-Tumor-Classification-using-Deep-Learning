# Brain Tumor Detection using DenseNet-201

This project uses a deep learning model to classify brain tumors from MRI scans. The model is built using the DenseNet-201 architecture and PyTorch, and it achieves an accuracy of **99.24%** on the test dataset.

## 📖 Overview

The primary goal of this project is to accurately classify MRI images into four categories:

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary Tumor

A pre-trained DenseNet-201 model is fine-tuned for this specific classification task. The project demonstrates a complete workflow, from data preprocessing and augmentation to model training, evaluation, and visualization of results.

## 🖼️ Dataset

This project utilizes the **Brain Tumor MRI Dataset** from Kaggle. The dataset is organized into two main directories: `Training` and `Testing`, with subdirectories for each of the four classes.

You can find the dataset [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

## 🛠️ Technologies Used

* **Python**: The core programming language.
* **PyTorch**: The deep learning framework used for building and training the model.
* **scikit-learn**: Used for evaluating the model's performance with metrics like the classification report and confusion matrix.
* **Matplotlib & Seaborn**: For data visualization and plotting the results.
* **NumPy**: For numerical operations.
* **Jupyter Notebook**: For interactive development and experimentation.

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/knitin123/Brain-Tumor-Classification-using-Deep-Learning
   cd https://github.com/knitin123/Brain-Tumor-Classification-using-Deep-Learning
   ```

2. **Install the dependencies:**
   ```bash
   pip install torch torchvision scikit-learn matplotlib seaborn numpy jupyter
   ```

3. **Download the dataset:**
   Download the dataset from the link provided above and place the `Training` and `Testing` folders in the project's root directory.

4. **Run the Jupyter Notebook:**
   Launch the notebook to see the data processing, model training, and evaluation steps.
   ```bash
   jupyter notebook "brain-tumer-detection-densenet-201-99-24.ipynb"
   ```

## 📊 Results

The model achieved an outstanding accuracy of **99.24%** on the test set.

