# Handwritten Digit Recognition using Deep Learning

## 📌 Project Overview
Comparative study of Neural Network architectures for MNIST handwritten digit classification, demonstrating the effectiveness of Convolutional Neural Networks (CNNs) over traditional Dense Neural Networks.

## 🎯 Objective
To build and compare multiple deep learning models for recognizing handwritten digits (0-9) from the MNIST dataset, analyzing their performance, and understanding the advantages of convolutional architectures in image classification tasks.

## 📊 Dataset
- **Source**: MNIST (Modified National Institute of Standards and Technology)
- **Training Samples**: 60,000
- **Test Samples**: 10,000
- **Image Size**: 28×28 pixels (grayscale)
- **Classes**: 10 (digits 0-9)

## 🏗️ Model Architectures

### Model 1: Simple Neural Network (Baseline)
- Flatten layer
- 2 Dense layers (128, 64 neurons)
- Dropout for regularization
- Output layer (10 classes)
- **Parameters**: 109,386

### Model 2: Convolutional Neural Network (CNN)
- 3 Convolutional blocks with Batch Normalization
- Max Pooling layers
- Dense layers with Dropout
- Output layer (10 classes)
- **Parameters**: 131,530

## 📈 Results

| Model      | Test Accuracy | Test Loss | Parameters |
|----------- |---------------|-----------|------------|
| Simple NN  | 97.77%        | 0.0728    | 109,386    |
| CNN        | **99.09%**    | 0.0289    | 131,530    |

**Key Findings:**
- CNN achieved **1.32%** higher accuracy with better feature extraction
- CNN showed better generalization with significantly lower test loss (0.0289 vs 0.0728)
- Most misclassifications occurred between visually similar digits (4-9, 3-5, 7-1)
- Per-class F1-scores ranged from 98.60% to 99.60%, showing consistent performance across all digits

## 🎯 Detailed Performance Metrics

### Per-Digit Performance (CNN Model):
| Digit | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 99.18%    | 99.18% | 99.18%   | 980     |
| 1     | 99.65%    | 99.56% | 99.60%   | 1,135   |
| 2     | 99.61%    | 98.93% | 99.27%   | 1,032   |
| 3     | 98.82%    | 99.50% | 99.16%   | 1,010   |
| 4     | 99.49%    | 98.98% | 99.23%   | 982     |
| 5     | 99.77%    | 97.98% | 98.87%   | 892     |
| 6     | 98.45%    | 99.27% | 98.86%   | 958     |
| 7     | 97.98%    | 99.22% | 98.60%   | 1,028   |
| 8     | 99.49%    | 99.18% | 99.33%   | 974     |
| 9     | 98.52%    | 98.91% | 98.71%   | 1,009   |

**Overall Accuracy: 99.09%**

## 🛠️ Technologies Used
- **Language**: Python 3.10+
- **Deep Learning**: TensorFlow 2.15, Keras
- **Data Processing**: NumPy
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: Scikit-learn

## 🚀 How to Run

### Prerequisites
```bash
Python 3.10 or higher
pip (Python package manager)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Project
```bash
python mnist_project.py
```

The script will:
1. Download and preprocess the MNIST dataset
2. Train both models (takes 10-15 minutes total)
3. Generate visualizations in the `results/` folder
4. Save detailed metrics in `results/summary.txt`

## 📁 Project Structure
```
mnist-digit-recognition/
│
├── mnist_project.py          # Main script
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
│
└── results/                  # Generated outputs
    ├── 01_sample_digits.png
    ├── 02_training_comparison.png
    ├── 03_confusion_matrix.png
    ├── 04_misclassified_examples.png
    ├── 05_correct_predictions.png
    └── summary.txt
```

## 📸 Visualizations

All visualizations are automatically generated and saved in the `results/` folder:

1. **Sample Digits**: Shows 15 random handwritten digits from the training set
2. **Training Comparison**: Plots accuracy and loss curves for both models
3. **Confusion Matrix**: Visualizes prediction performance across all digit classes
4. **Misclassified Examples**: Shows the first 20 incorrect predictions with confidence scores
5. **Correct Predictions**: Displays correctly classified samples with confidence scores

## 🔍 Key Insights

1. **CNN Superiority**: Convolutional layers automatically learn spatial hierarchies in images, making them more effective than fully connected networks for image classification tasks.

2. **Feature Learning**: CNNs learn hierarchical features (edges → shapes → digits) which provides better representation compared to flattened pixel inputs used by simple neural networks.

3. **Error Analysis**: Most errors occurred with digits that share visual similarities:
   - Digit 5 had the lowest recall (97.98%), often confused with 3 and 8
   - Digit 7 had lower precision (97.98%), sometimes predicted as 1 or 9
   - Digits 4 and 9 showed confusion due to similar curved tops

4. **Generalization**: CNN showed significantly better generalization with lower validation loss, indicating reduced overfitting through architectural inductive biases.

5. **Robustness**: High precision and recall across all classes (>97.9%) demonstrates model robustness to variations in handwriting styles.

## 🔮 Future Improvements

- [ ] Implement data augmentation (rotation, scaling, translation)
- [ ] Try deeper architectures (ResNet, DenseNet, EfficientNet)
- [ ] Experiment with different optimizers (SGD with momentum, AdamW)
- [ ] Add learning rate scheduling for better convergence
- [ ] Deploy model as a web application using Flask or Streamlit
- [ ] Test on real-world handwritten samples (EMNIST, custom images)
- [ ] Implement ensemble methods combining multiple models
- [ ] Add grad-CAM visualization to understand model decisions

## 💡 Real-World Applications

This digit recognition technology is used in:
- **Banking**: Automated check processing and form digitization
- **Postal Services**: ZIP code recognition for mail sorting
- **Healthcare**: Medical form digitization
- **Document Processing**: Converting handwritten documents to digital text
- **Mobile Apps**: Note-taking apps with handwriting recognition

## 📚 Learning Outcomes

This project demonstrates:
- Implementation of neural network architectures from scratch
- Understanding of CNN components (convolution, pooling, batch normalization)
- Model comparison and evaluation using multiple metrics
- Data preprocessing and normalization techniques
- Visualization of training dynamics and model performance
- Error analysis and model interpretation
- Best practices in deep learning experimentation

## 👨‍💻 Author
**Sanjay**
- GitHub: https://github.com/Sanjay284-beep
- LinkedIn: https://www.linkedin.com/in/sanjaynarra
- Location: Hyderabad, India

## 📝 License
This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments
- MNIST dataset by Yann LeCun and Corinna Cortes
- TensorFlow and Keras teams for the excellent deep learning framework
- The open-source community for inspiration and resources

## 📖 References
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition.
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras API Reference: https://keras.io/

---


A comparative study of neural network architectures for image classification