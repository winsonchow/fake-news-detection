# Fake News Detection with NLP and Machine Learning

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Evaluation](#evaluation)
- [Installation](#installation)

## Introduction
This project aims to tackle the growing problem of fake news by developing an advanced detection system leveraging Natural Language Processing (NLP) and Machine Learning (ML) techniques. By analyzing textual data, the system distinguishes between genuine and fabricated news, contributing to the integrity of public discourse. This project was also serves as my capstone project.

## Features
- Utilizes Python, PyTorch, and scikit-learn for development.
- Implements algorithms including Random Forest, Deep Learning models, and the Longformer model.
- Employs sentiment analysis and advanced feature engineering for improved detection accuracy.

## Dataset
The politifact dataset comprises diverse news types, processed and balanced to train and test the models effectively. It includes rebalancing techniques like SMOTE to prevent model bias.

## Models

This project employs a comprehensive approach to fake news detection by leveraging three distinct models: Random Forest, a Deep Learning Model, and the Longformer Model. Each model contributes uniquely to the robustness and accuracy of the detection system.

### Random Forest

- **Description**: Random Forest is an ensemble learning method that creates a 'forest' of decision trees during training. It outputs the mode of the classes (for classification) from individual trees, offering a balanced approach to handle high-dimensional data and ensure model reliability.
- **Rationale**: Chosen for its ability to provide insights into feature importance, thus helping identify key indicators of misinformation. Its ensemble nature helps mitigate overfitting, making it a stable foundation for our detection system.
- **Unique Features**: The interpretability of Random Forest facilitates understanding the decision-making process, crucial for analyzing fake versus genuine news characteristics.

### Deep Learning Model

- **Architecture**: Includes embedding, LSTM (Long Short-Term Memory) layers, attention mechanisms, layer normalization, fully connected layers, and dropout layers. This architecture is adept at processing sequential and textual data, capturing semantic relationships and learning long-term dependencies.
- **Training Process**: Utilizes Bayesian optimization for hyperparameter tuning to systematically explore the hyperparameter space, enhancing performance. Key parameters include embedding dimension, hidden dimension, number of LSTM layers, bidirectionality, dropout rate, and learning rate.
- **Rationale**: The deep learning model's comprehensive architecture allows for nuanced understanding of language, making it highly effective for detecting complex patterns of deception.

### Longformer Model

- **Description**: Extends traditional transformer architectures to efficiently process longer sequences of text. Its innovative attention mechanism scales linearly with sequence length, enabling the handling of extensive articles and documents.
- **Training Process**: Characterized by meticulous setup involving an optimizer, learning rate scheduler, dedicated training and evaluation functions, and an early stopping mechanism to ensure efficient and effective learning.
- **Unique Features**: The Longformer's ability to perform a detailed analysis of entire narratives, capturing not just local textual features but also broader discourse context, makes it a powerful tool for uncovering sophisticated tactics employed in the dissemination of fake news.

## Evaluation
The system is evaluated using metrics such as accuracy, precision, recall, and F1-score across multiple models, demonstrating the effectiveness of the applied methods in fake news detection.

## Installation
Instructions on setting up the project environment, including required libraries and dependencies.

```bash
git clone https://github.com/your-github/fake-news-detection
cd fake-news-detection
pip install -r requirements.txt
