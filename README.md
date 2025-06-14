 # 🚦 Lightweight Sri Lankan Traffic Sign Detection with YOLOv12n

This project presents a **real-time traffic sign detection system** built using a streamlined version of the YOLOv12n object detection model. It focuses on **reducing computational complexity** through **structured pruning** and **post-training quantization**, while maintaining high detection accuracy—ideal for deployment on edge and low-resource devices.

##  Key Features

- 🔍 **Model Options**: Choose between the original, pruned, and quantized YOLOv12n models.
- 📷 **Image & Video Support**: Upload images or videos to detect traffic signs in real time.
- 📉 **Model Optimization**:
  - 30% L1-norm-based structured pruning
  - 50% FLOPs reduction
  - 70%+ model size reduction (from 10.15MB to 2.98MB)
- 📊 **Detection Dashboard**: Interactive dashboard with class distribution charts and detection stats.
- ⚡ **Final Accuracy**: 78% after quantization, with minimal drop from the original performance.

## 🖥️ Web App Interface

Built using **Streamlit**, the web interface allows users to:
- Upload and analyze images/videos
- View detected labels and confidence levels
- Visualize class distributions via dynamic bar charts

## 🚀 How to Run

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/traffic-sign-detection-yolov12n.git
   cd traffic-sign-detection-yolov12n
