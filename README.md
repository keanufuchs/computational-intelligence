# 🃏 Playing Card Classification with CNN

A comprehensive, educational Jupyter Notebook that demonstrates how to build a **Convolutional Neural Network (CNN)** for classifying playing cards. This project is designed as a teaching tool for beginners in Python, Machine Learning, and Neural Networks.

## 📚 What You'll Learn

This notebook covers:

1. **Neural Network Basics** - Neurons, weights, activation functions
2. **CNN Architecture** - Convolution, Pooling, ReLU, Softmax
3. **Training Process** - Forward pass, loss calculation, backpropagation
4. **Model Evaluation** - Accuracy, confusion matrix, error analysis
5. **Practical Application** - Testing with your own photos

## 🎯 Project Overview

| Aspect | Details |
|--------|---------|
| **Task** | Image Classification (53 classes) |
| **Dataset** | [Complete Playing Card Dataset](https://www.kaggle.com/datasets/jaypradipshah/the-complete-playing-card-dataset) |
| **Model** | Custom CNN with 4 Conv blocks |
| **Input Size** | 256 × 256 × 3 (RGB) |
| **Parameters** | ~13 Million |
| **Accuracy** | ~97-99% on validation set |

## 🏗️ Model Architecture

```
Input (256×256×3)
    ↓
┌─────────────────────────────────────┐
│  Conv2D(32) → ReLU → MaxPool(2×2)  │  Block 1: Edge detection
│  Conv2D(64) → ReLU → MaxPool(2×2)  │  Block 2: Corner/shape detection
│  Conv2D(128) → ReLU → MaxPool(2×2) │  Block 3: Pattern detection
│  Conv2D(256) → ReLU → MaxPool(2×2) │  Block 4: Symbol recognition
└─────────────────────────────────────┘
    ↓
Flatten (14×14×256 = 50,176 neurons)
    ↓
Dense(256) → ReLU → Dropout(0.5)
    ↓
Dense(53) → Softmax
    ↓
Output (probabilities for each card class)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- ~4GB free disk space (for dataset)
- ~2GB RAM minimum

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/keanufuchs/computational-intelligence.git
   cd computational-intelligence
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install tensorflow matplotlib scikit-learn numpy pillow seaborn
   ```
   
   For macOS with Apple Silicon (M1/M2/M3):
   ```bash
   pip install tensorflow-macos tensorflow-metal matplotlib scikit-learn numpy pillow seaborn
   ```

4. **Open the notebook**
   ```bash
   jupyter notebook spielkarten_cnn.ipynb
   # or
   jupyter lab
   ```

5. **Run all cells** - The notebook will:
   - Download the dataset automatically (~1.5GB)
   - Organize the data
   - Train the CNN (~5 minutes)
   - Evaluate and visualize results

## 📁 Project Structure

```
computational-intelligence/
├── spielkarten_cnn.ipynb    # Main notebook (fully documented)
├── README.md                 # This file
├── .gitignore
└── data/                     # Created automatically
    ├── playing-cards.zip     # Downloaded dataset
    ├── playing-cards/        # Extracted dataset
    │   ├── Images/Images/    # Original images
    │   ├── Annotations/      # (not used)
    │   └── YOLO_Annotations/ # (not used)
    ├── organized_images/     # Keras-compatible structure
    │   ├── 10C/              # 10 of Clubs
    │   ├── KH/               # King of Hearts
    │   └── ...               # 53 class folders
    └── input/                # Your test images go here!
```

## 🧪 Testing with Your Own Photos

1. Take a photo of a playing card
2. Copy it to `./data/input/`
3. Run the last cell in the notebook ("Eigene Bilder testen")
4. See the prediction!

**Tips for best results:**
- 📷 Photograph the card from above (not at an angle)
- 💡 Use good, even lighting
- 🎯 Card should fill most of the frame
- 🖼️ Supported formats: `.jpg`, `.jpeg`, `.png`

## 📊 Results

The model achieves:

| Metric | Value |
|--------|-------|
| Training Accuracy | ~98% |
| Validation Accuracy | ~97-99% |
| Training Time | ~5 minutes (CPU) |

### Sample Predictions

The notebook includes visualizations of:
- Training/validation curves
- Confusion matrix
- Example predictions with confidence scores
- Error analysis

## 🔧 Customization

### Hyperparameters (Cell 13)

```python
IMG_HEIGHT = 256      # Image height (try 128 for faster training)
IMG_WIDTH = 256       # Image width
BATCH_SIZE = 16       # Batch size (increase if you have more RAM)
EPOCHS = 10           # Number of training epochs
VALIDATION_SPLIT = 0.2  # 20% for validation
```

### Model Architecture (Cell 18)

You can modify the CNN by:
- Adding/removing Conv blocks
- Changing filter counts (32, 64, 128, 256)
- Adjusting Dense layer neurons
- Changing dropout rate

## 📖 Educational Content

The notebook is extensively documented in **German** with explanations of:

- Why each layer is used
- How convolution works
- What pooling does
- How backpropagation trains the network
- How to interpret results

Each code cell includes detailed comments explaining the "what" and "why".

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add translations
- Improve documentation

## 📄 License

This project is for educational purposes. The dataset is from Kaggle and subject to its own license.

## 🙏 Acknowledgments

- Dataset: [Jay Pradip Shah](https://www.kaggle.com/datasets/jaypradipshah/the-complete-playing-card-dataset) on Kaggle
- Framework: TensorFlow/Keras
- Inspiration: University lecture on Computational Intelligence

---

**Happy Learning! 🚀**

*If you found this helpful, please ⭐ the repository!*
