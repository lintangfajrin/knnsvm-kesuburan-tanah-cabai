# Soil Fertility Classification using KNN & SVM in Rust

This Rust project implements a Soil Fertility Classifier using two machine learning algorithms that's K-Nearest Neighbors (KNN) and Support Vector Machine (SVM). It processes soil data (temperature, humidity, pH, moisture) and predicts whether the soil is fertile or not. The results are visualized using 2D plots with decision boundaries.

## Features

- Min-max normalization of features
- Custom train-test splitting (80% train, 20% test)
- Custom KNN classifier with Euclidean distance
- Custom linear SVM classifier (gradient descent style)
- Accuracy evaluation for both models
- Visualization of:
  - Input data distribution
  - KNN & SVM decision boundaries using `plotters`

## 📦 Dependencies

Make sure to add this in your `Cargo.toml`:

```toml
[dependencies]
plotters = "0.3"

