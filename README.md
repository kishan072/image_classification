# Intel Image Classification using CNN

A complete deep learning project for classifying images into 6 categories using Convolutional Neural Networks (CNN).

## 🎯 Project Overview

This project implements an image classification system that can categorize images into 6 different classes:
- 🏢 **Buildings**
- 🌲 **Forest** 
- 🏔️ **Glacier**
- ⛰️ **Mountain**
- 🌊 **Sea**
- 🛣️ **Street**

## 📊 Results

- **Test Accuracy:** 83.93%
- **Training Accuracy:** 82.42%
- **Validation Accuracy:** 83.44%

## 🗂️ Dataset

Uses the **Intel Image Classification** dataset from Kaggle containing ~25,000 images across 6 categories.

**Required folder structure:**
```
project_folder/
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
├── seg_test/
│   └── seg_test/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
└── seg_pred/ (optional)
```

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/HarshithaTech/intel-image-classification-using-CNN.git
cd intel-image-classification-using-CNN
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
- Download the "Intel Image Classification" dataset from Kaggle
- Extract and organize according to the folder structure above

### 4. Run the project
```bash
python image_classification_project.py
```

## 🏗️ Model Architecture

```
CNN Model:
├── Conv2D (32 filters) + MaxPooling2D
├── Conv2D (64 filters) + MaxPooling2D  
├── Conv2D (128 filters) + MaxPooling2D
├── Conv2D (128 filters) + MaxPooling2D
├── Flatten + Dropout (0.5)
├── Dense (512 units) + Dropout (0.5)
└── Dense (6 units, softmax)
```

**Model Parameters:**
- Input Shape: 150x150x3
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

## 🔧 Features

- ✅ **Data Augmentation:** Rotation, zoom, horizontal flip
- ✅ **Image Preprocessing:** Resize to 150x150, normalization
- ✅ **Training Visualization:** Accuracy and loss plots
- ✅ **Model Evaluation:** Test dataset performance
- ✅ **Model Persistence:** Save/load trained model
- ✅ **Single Image Prediction:** Classify new images

## 📈 Training Details

- **Epochs:** 20
- **Batch Size:** 32
- **Train/Validation Split:** 80/20
- **Data Augmentation:** Yes
- **Regularization:** Dropout layers

## 🧪 Testing

To test the model on a single image:

```python
predict_single_image('path_to_your_image.jpg', model)
```

Or run the test script:
```bash
python test_prediction.py
```

## 📁 Project Structure

```
├── image_classification_project.py  # Main training script
├── test_prediction.py              # Single image testing
├── quick_test.py                   # Quick model test
├── save_and_test.py               # Save model and test
├── requirements.txt               # Dependencies
├── README.md                     # Project documentation
└── .gitignore                   # Git ignore rules
```

## 🛠️ Technologies Used

- **Python 3.10+**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization
- **Pillow** - Image processing

## 📋 Requirements

```
tensorflow
matplotlib
numpy
Pillow
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Harshitha K S**
- GitHub: [@HarshithaTech](https://github.com/HarshithaTech)

## 🙏 Acknowledgments

- Intel for providing the image classification dataset
- Kaggle community for dataset hosting
- TensorFlow team for the amazing deep learning framework