# semantic-segmentation
# Efficient Semantic Segmentation using DeepLabV3+ on Pascal VOC

This project implements a lightweight DeepLabV3+ architecture with a MobileNetV2 backbone to perform semantic segmentation on the Pascal VOC 2012 dataset.

## 📌 Objective

Segment objects in images by assigning each pixel a class label using a DeepLabV3+ model trained on Pascal VOC.

---

## 📁 Dataset

- **Dataset Used:** Pascal VOC 2012
- **Classes:** 21 total (20 foreground object classes + background)
- **Source:** [Pascal VOC Official Site](http://host.robots.ox.ac.uk/pascal/VOC/)

---

## ⚙️ Implementation Steps

1. **Download & Prepare the Dataset**  
   Automatically downloads and extracts Pascal VOC 2012 data.

2. **Data Generator**  
   Custom `VOCDataGenerator` class to efficiently load and preprocess image-mask pairs using `tf.keras.utils.Sequence`.

3. **Model Architecture**  
   - Encoder: MobileNetV2 (pre-trained on ImageNet)  
   - Decoder: ASPP-style multi-scale feature extractor with dilation rates  
   - Final output layer: 21-class softmax segmentation map

4. **Training Setup**  
   - Loss: Categorical Crossentropy  
   - Optimizer: Adam  
   - Batch Size: 4  
   - Epochs: 5  
   - Callbacks: EarlyStopping & ModelCheckpoint

5. **Prediction & Visualization**  
   Outputs segmentation mask overlayed on input image.

---

## 🧪 Expected Outcome

- High-quality image segmentation
- Accurate per-pixel classification for 21 semantic classes
- Efficient training with a lightweight architecture

---

## 📸 Sample Output

Visualizes:
- Original Image
- Ground Truth Mask
- Predicted Segmentation Mask

---

## 🛠 Requirements

- TensorFlow 2.x
- NumPy
- OpenCV
- Matplotlib
- tqdm
- requests

You can install them using:

```run the file
python main.py

Model Summary
MobileNetV2 base used till block_13_expand_relu

ASPP-like decoder with upsampling

Fully convolutional for end-to-end pixel classification

Acknowledgments
Pascal VOC Dataset

TensorFlow

MobileNetV2 Paper
