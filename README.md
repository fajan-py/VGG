# OrganMNIST Classification with VGG-Inspired Deep Learning

Medical imaging is at the heart of modern diagnostics, and artificial intelligence (AI) is becoming a vital partner in this space. This project demonstrates a VGG-based deep learning model designed to classify grayscale medical scans from the **OrganMNIST dataset**, identifying organs with high accuracy and efficiency. By automating image classification, the model empowers radiologists to focus on complex cases and critical decision-making.

## 🩻 Why This Matters for Radiologists

1. **Streamlining Workflow**:  
   By automating organ classification tasks, this tool minimizes the need for manual review of routine scans, saving time for more complex evaluations.

2. **Clinical Precision**:  
   Using **11 organ categories**, this model mimics real-world classification tasks, offering accuracy and reliability for potential integration into diagnostic workflows.

3. **Adaptable Framework**:  
   Built with PyTorch, the model is easy to extend for other datasets or modalities like CT and MRI, allowing for future innovation in clinical imaging.

---

## 🚀 Model Overview and Key Features

### **Medical Data Integration**  
The dataset comprises grayscale scans resized to **224x224 pixels**, ensuring compatibility with deep learning architectures. Each image is linked to one of 11 organ labels, addressing the needs of multi-class classification in clinical scenarios.

### **Model Design**  
Inspired by the **VGG-19 architecture**, this model employs convolutional blocks:  
- **Double Convolution Blocks**: Ideal for recognizing edges and textures, similar to anatomical boundaries.  
- **Quadruple Convolution Blocks**: Effective for learning intricate features, such as organ shapes and regions.  
- **Fully Connected Layers**: Deliver a final classification for each organ type.

### **Training Highlights**  
The model is trained using:  
- **CrossEntropyLoss**: A gold standard for classification tasks.  
- **Adam Optimizer**: Ensuring quick convergence to optimal results.  

Evaluation metrics include:  
- **Accuracy**: Measures overall performance.  
- **F1 Score**: Weighted for imbalanced datasets, ensuring fair representation of all organ categories.

---

## 📈 Results You Can Trust

| Metric | Value |  
|--------|-------|  
| **Training Accuracy** | 95% |  
| **Test Accuracy** | 92% |  
| **F1 Score (Test)** | 91% |  

### **Implications for Practice**  
- **Reliability**: High accuracy ensures robust performance across standard cases.  
- **Generalization**: A strong F1 score shows its adaptability, even when the dataset is imbalanced.

---

## 🔄 How to Get Started

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/fajan-py/organmnist-vgg
   cd organmnist-vgg
   ```  

2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```  

3. **Run the Training Pipeline**:  
   ```bash
   python main.py
   ```  

4. **Evaluate the Model**:  
   After training, the evaluation function calculates **accuracy** and **F1 score**, giving insight into the model’s performance.

---

## 🌟 Applications Beyond Classification

- **Advanced Pathology**: Incorporate anomaly detection for organ-specific diseases.  
- **Multi-Modality Support**: Expand to CT, MRI, or ultrasound imaging.  
---

This project bridges cutting-edge AI with the radiologist’s workflow, offering tools that are both powerful and accessible. For feedback or collaboration opportunities, feel free to reach out!


