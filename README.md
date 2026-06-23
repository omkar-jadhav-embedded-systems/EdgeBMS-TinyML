# EdgeBMS-TinyML: Battery SoC & SoH Estimation using Edge AI 🔋🧠

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end **TinyML (Edge AI)** pipeline that trains, quantizes, and deploys a deep neural network to predict a battery's **State of Charge (SoC)** and **State of Health (SoH)**. Designed specifically for resource-constrained embedded systems and Battery Management Systems (BMS).

## 🚀 Project Overview

Accurate estimation of Battery SoC and SoH is notoriously difficult due to the highly non-linear electrochemical physics of lithium-ion cells. Traditional algorithms (like Kalman Filters) are computationally heavy or require complex tuning. 

This project solves this by using a Deep Neural Network (DNN) to map raw physical sensor data directly to SoC and SoH percentages. Using **TensorFlow Model Optimization**, the model is aggressively compressed (via Quantization-Aware Training) into an **INT8 C++ array** capable of running on bare-metal microcontrollers (e.g., STM32, ESP32, Arduino) without an operating system.

### **Inputs (Raw Sensor Data)**
1. **Voltage (V)** 
2. **Current (A)** *(Positive for charging, negative for discharging)*
3. **Temperature (°C)**

### **Outputs (State Estimation)**
1. **State of Charge (SoC %)**
2. **State of Health (SoH %)**

---

## 📊 Performance & Optimization Results

By utilizing an adaptive Normalization layer, a deep architecture (`32 -> 32 -> 16 -> 8 -> 2`), and a custom learning rate, the model achieves industry-viable accuracy while maintaining an ultra-low memory footprint.

* **Accuracy (within 4.5% tolerance):** `98.12%`
* **Mean Absolute Error:** `~1.50%`
* **Optimized Model Size:** `6.38 KB` (Fits easily in the flash memory of modern microcontrollers)
* **Model Parameters:** `1,882` (Quantized to INT8)

---

## 🛠️ Technical Workflow

This repository demonstrates a complete, production-ready AI-to-Embedded pipeline:

1. **Data Ingestion & Preprocessing:** Loads raw battery operational data from Excel datasets.
2. **Auto-Scaling:** Uses `scikit-learn's MinMaxScaler` layer to internally standardize features (Voltage, Current, Temp) with wildly different scales, eliminating the need for external libraries on the embedded device.
3. **Deep Learning Architecture:** Trains a multi-layer Keras Sequential model.
4. **Quantization-Aware Training (QAT):** Simulates low-precision (INT8) math during training to ensure the model doesn't lose accuracy when converted for hardware.
5. **TFLite Conversion:** Converts the trained model into a highly optimized, fully integer `.tflite` flatbuffer using a representative dataset for calibration.
6. **C++ Code Generation:** Automatically serializes the binary `.tflite` model into a standard C-style hexadecimal array (`model.h`) for direct compilation into embedded C/C++ firmware.

---

## 📂 Repository Structure

```text
EdgeBMS-TinyML/
├── Training_data.xlsx      # Synthetic operational training dataset (1,600 samples)
├── Testing_data.xlsx       # Synthetic operational testing dataset (400 samples)
├── train_bms_model.py      # Main pipeline script (Training -> Quantization -> C++ Export)
├── bms_model.h             # Auto-generated C++ header containing the optimized INT8 model
├── Final_model/            # Saved Keras/TensorFlow SavedModel directory
├── lite/                   # Directory containing the intermediate .tflite file
└── README.md               # Project documentation
```

---

## 💻 How to Run the Pipeline

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with the required machine learning and data processing libraries.

```bash
pip install tensorflow tensorflow-model-optimization pandas openpyxl numpy
```

### 2. Clone the Repository
```bash
git clone https://github.com/[Your-Username]/EdgeBMS-TinyML.git
cd EdgeBMS-TinyML
```

### 3. Execute the Training & Conversion Script
Run the main pipeline. The script will train the model, output evaluation metrics to the console, and automatically generate the `bms_model.h` file.

```bash
python train_bms_model.py
```

### 4. Deploying to Hardware
To use this in an embedded project, simply `#include "bms_model.h"` in your C/C++ application and load it using the TensorFlow Lite for Microcontrollers (TFLM) C++ interpreter.

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
