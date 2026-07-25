# 🔬 Breast Cancer IDC Classifier

A **Streamlit** web application for classifying **Invasive Ductal Carcinoma (IDC)** from breast histopathology images using a **MobileNetV2** deep learning model, with **Grad-CAM** visual explanations.

---

## 🚀 Features

- **Upload & Classify** – Upload breast histopathology images (JPG, JPEG, PNG)
- **IDC Prediction** – Classifies tissue as `positive_IDC` or `negative_IDC`
- **Confidence Score** – Displays prediction confidence as a percentage
- **Class Probabilities** – Shows probability distribution across all classes
- **Grad-CAM Visualization** – Generates a heatmap overlay highlighting regions that influenced the model's decision

---

## 🧠 Model

The app uses a **MobileNetV2** model trained to detect Invasive Ductal Carcinoma (the most common type of breast cancer) from histopathology images.

The model expects input images of size **224×224** pixels, preprocessed using MobileNetV2's built-in preprocessing pipeline.

### Classes

| Index | Class          |
|-------|----------------|
| 0     | `positive_IDC` |
| 1     | `negative_IDC` |
| 2     | `Unknown`      |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd BreastCancerAPP
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the model

Place the pre-trained model file `IDC_MobileNetV2_model.keras` in the project root directory.

> ⚠️ The model file is excluded from version control via `.gitignore`. You need to obtain it separately or train it yourself.

---

## ▶️ Usage

Run the app with Streamlit:

```bash
streamlit run app.py
```

Then open your browser at the URL shown in the terminal (usually `http://localhost:8501`).

### How to use:

1. Click **"Browse files"** to upload a histopathology image
2. Wait for the model to process the image
3. View the **prediction result**, **confidence score**, and **class probabilities**
4. Scroll down to see the **Grad-CAM heatmap** overlay explaining the model's decision

---

## 📁 Project Structure

```
BreastCancerAPP/
├── app.py                          # Streamlit application entry point
├── utils.py                        # Utility functions (preprocessing, Grad-CAM, overlay)
├── requirements.txt                # Python dependencies
├── IDC_MobileNetV2_model.keras     # Pre-trained MobileNetV2 model (not tracked)
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation (this file)
```

### Key Files

| File               | Description                                            |
|--------------------|--------------------------------------------------------|
| `app.py`           | Main Streamlit UI – handles upload, prediction, display |
| `utils.py`         | Image preprocessing, Grad-CAM generation, heatmap overlay |
| `requirements.txt` | Python package dependencies                            |

---

## ⚙️ Dependencies

- [TensorFlow](https://www.tensorflow.org/) – Deep learning framework
- [Streamlit](https://streamlit.io/) – Web app framework
- [NumPy](https://numpy.org/) – Numerical computing
- [OpenCV](https://opencv.org/) – Image processing (headless version)
- [Pillow](https://python-pillow.org/) – Image handling

---

## 🧪 How It Works

1. **Image Upload** – User uploads a histopathology image via Streamlit's file uploader
2. **Preprocessing** – The image is resized to 224×224 and normalized using MobileNetV2's `preprocess_input`
3. **Prediction** – The model predicts class probabilities
4. **Grad-CAM** – A heatmap is generated from the last convolutional layer to visualize which regions influenced the prediction
5. **Display** – Results, probabilities, and the heatmap overlay are shown in the UI

---

## 📃 License

This project is provided for educational and research purposes.

---

## 🙌 Acknowledgements

- Built with [Streamlit](https://streamlit.io/)
- Model architecture based on [MobileNetV2](https://arxiv.org/abs/1801.04381)
- Grad-CAM implementation adapted from the original [Grad-CAM paper](https://arxiv.org/abs/1610.02391)

