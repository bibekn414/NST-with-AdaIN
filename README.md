# Neural Style Transfer using AdaIN

A deep learning-based **Neural Style Transfer (NST)** project that combines the content of one image with the artistic style of another image using **Adaptive Instance Normalization (AdaIN)**. The project uses a **pre-trained VGG-19 encoder**, a trainable **convolutional decoder**, **PyTorch** for deep learning, and a **Flask web application** for interactive image stylization.

The system allows users to upload a **content image** and a **style image**, control the strength of style transfer using an **alpha parameter**, and generate a stylized image through an easy-to-use web interface.

---

## Project Overview

**Neural Style Transfer** is a computer vision technique that generates a new image by preserving the semantic content of one image while transferring the visual characteristics of another image.

In this project, **Adaptive Instance Normalization (AdaIN)** is used to align the channel-wise mean and standard deviation of content and style feature representations.

The overall system consists of:

* Pre-trained **VGG-based feature encoder**
* **Adaptive Instance Normalization (AdaIN)**
* Trainable **convolutional decoder**
* Content and style image datasets
* PyTorch-based training pipeline
* Pre-trained decoder checkpoint
* Flask-based web interface
* Adjustable style-transfer strength using the **alpha parameter**
* Example input and output images

The implementation follows an encoder–AdaIN–decoder architecture, where the encoder extracts deep visual features, AdaIN transfers the style statistics, and the decoder reconstructs the stylized image.

---

## Overall Objective

The main objective of this project is to build an efficient and practical **Neural Style Transfer system** capable of transferring the artistic characteristics of a style image onto a content image while preserving the important structural information of the content image.

The project focuses on:

1. Understanding deep feature representations using a pre-trained VGG network.
2. Implementing **Adaptive Instance Normalization (AdaIN)** for fast style transfer.
3. Training a decoder to reconstruct images from transformed feature representations.
4. Balancing content preservation and style similarity through content and style losses.
5. Building an interactive web application for practical image stylization.
6. Providing control over the amount of style applied to the content image.

---

## Key Features

* **Neural Style Transfer**
* **Adaptive Instance Normalization (AdaIN)**
* **Deep Learning-based image stylization**
* **VGG feature extraction**
* **Encoder–decoder architecture**
* **Content feature preservation**
* **Style feature transformation**
* **Content and style loss optimization**
* **Adjustable style intensity**
* **GPU/CPU support through PyTorch**
* **Flask web interface**
* **Image upload functionality**
* **JPEG, JPG and PNG support**
* **Pre-trained model checkpoint**
* **Sample input/output images**
* **Interactive image stylization**

---

# Project Workflow

The complete workflow can be summarized as:

```text
             ┌─────────────────────┐
             │   Content Image     │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │    VGG Encoder      │
             │ Feature Extraction  │
             └──────────┬──────────┘
                        │
                        │ Content Features
                        ▼
                  ┌───────────┐
                  │   AdaIN   │◄──────────────┐
                  └─────┬─────┘               │
                        │                     │
                        │ Stylized Features   │
                        │                     │
             ┌──────────▼──────────┐   ┌──────┴──────────┐
             │   Decoder Network   │   │   Style Image   │
             │ Image Reconstruction│   └────────┬────────┘
             └──────────┬──────────┘            │
                        │                       ▼
                        │               ┌───────────────┐
                        │               │  VGG Encoder  │
                        │               │Style Features │
                        │               └───────────────┘
                        ▼
             ┌─────────────────────┐
             │  Stylized Image    │
             └─────────────────────┘
```

### Simplified Process

**Input Images → Preprocessing → VGG Feature Extraction → AdaIN → Feature Transformation → Decoder → Stylized Image**

---

# How AdaIN Works

Adaptive Instance Normalization is the core technique used for transferring style information.

For a content feature representation and a style feature representation, AdaIN first calculates their channel-wise statistics.

The content feature is normalized using its own mean and standard deviation:

```text
Normalized Content Feature = (Content Feature - Content Mean) / Content Standard Deviation
```

The normalized feature is then scaled and shifted using the statistics of the style feature:

```text
AdaIN Feature = Normalized Content Feature × Style Standard Deviation + Style Mean
```

This allows the model to transfer important statistical characteristics of the style image into the content feature representation.

In the implementation, the mean and standard deviation are calculated across the spatial dimensions of the feature maps.

---

# Model Architecture

## 1. VGG Encoder

The project uses a **VGG-based encoder** to extract hierarchical visual features from input images.

The encoder contains multiple convolutional, ReLU, padding, and pooling layers and uses features up to the **relu4-1** stage for the main AdaIN representation.

The encoder is initialized using pre-trained VGG weights and its parameters are frozen during decoder training.

### Role of the Encoder

The encoder converts an RGB image into a high-level feature representation containing information about:

* Shapes
* Edges
* Textures
* Patterns
* Semantic structures
* Visual appearance

The content and style images are independently passed through the encoder to obtain their feature representations.

---

## 2. Adaptive Instance Normalization

The extracted content and style features are passed to the **AdaIN layer**.

AdaIN aligns the statistical properties of the content features with those of the style features.

This enables the system to preserve the structure of the content image while introducing the visual characteristics of the style image.

---

## 3. Decoder

After AdaIN, the transformed feature representation is passed through a trainable convolutional decoder.

The decoder is responsible for converting the transformed deep feature representation back into an RGB image.

The decoder uses:

* Convolutional layers
* ReLU activation
* Reflection padding
* Nearest-neighbor upsampling
* Progressive feature-channel reduction

The final layer produces a **3-channel RGB image**.

---

# Training Pipeline:-

The decoder is trained while the VGG encoder remains frozen.

The training process follows these steps:

### Step 1 — Load Content and Style Images

Content and style images are loaded using a custom PyTorch `Dataset`.

The dataset supports:

* `.jpg`
* `.jpeg`
* `.png`

Images are converted to RGB format before being processed.

### Step 2 — Image Preprocessing

The images are resized and transformed into PyTorch tensors.

The training configuration supports:

* Content image resizing
* Style image resizing
* Random cropping
* Final image size control
* Batch processing

### Step 3 — Feature Extraction

Both content and style images are passed through the frozen VGG encoder.

```text
Content Image → VGG Encoder → Content Features

Style Image → VGG Encoder → Style Features
```

### Step 4 — Apply AdaIN

The deepest content and style feature representations are passed to AdaIN.

```text
Content Features + Style Features
              ↓
             AdaIN
              ↓
      Target Feature Representation
```

### Step 5 — Image Reconstruction

The decoder receives the AdaIN-transformed features and generates the stylized image.

```text
AdaIN Features
      ↓
   Decoder
      ↓
Generated Image
```

### Step 6 — Feature Re-extraction

The generated image is passed back through the VGG encoder.

This allows the model to compare the generated image's feature representation with the desired content and style representations.

### Step 7 — Calculate Loss

Two major losses are used:

* **Content Loss**
* **Style Loss**

The total loss is:

```text
Total Loss = Content Loss + Style Loss
```

The implementation uses **Mean Squared Error (MSE)** for these loss calculations.

---

# Content Loss

Content loss measures how closely the generated image preserves the target AdaIN content representation.

Conceptually:

```text
Content Loss = MSE(Generated Features, Target AdaIN Features)
```

A lower content loss means that the generated image is better at preserving the desired content representation.

The project uses a configurable **content weight** to control the contribution of this loss.

---

# Style Loss

Style loss measures the difference between the generated image and the style image using feature statistics.

For different VGG feature levels, the implementation compares:

* Feature mean
* Feature standard deviation

The style loss is calculated using MSE between the corresponding statistics of generated and style features.

This helps the generated image reproduce the texture and appearance characteristics of the style image.

---

# Alpha Parameter

The project provides an **alpha parameter** to control the strength of style transfer.

The transformed feature representation is combined with the original content feature representation as:

```text
Final Feature = α × AdaIN Feature + (1 - α) × Content Feature
```

Therefore:

* **α = 0** → mostly preserves the original content representation
* **α = 1** → applies the full AdaIN transformation
* **0 < α < 1** → provides a mixture of original content and transferred style

This makes the application more flexible because users can control how strongly the artistic style affects the final image.

---

# Web Application

The project also includes a **Flask-based web application** that provides an interactive interface for Neural Style Transfer.

The application allows users to:

1. Upload a content image.
2. Upload a style image.
3. Specify the style-transfer strength using alpha.
4. Run the style-transfer pipeline.
5. Generate and display the stylized image.

The Flask application uses the trained decoder and VGG encoder for inference. It automatically selects **CUDA when available**, otherwise it uses the CPU.

---

# Technologies and Software Used

## Programming Language

* **Python**

## Deep Learning Framework

* **PyTorch**

## Computer Vision / Image Processing

* **Torchvision**
* **Pillow (PIL)**

## Neural Network Architecture

* **VGG-based Encoder**
* **Convolutional Decoder**
* **Adaptive Instance Normalization (AdaIN)**

## Web Development

* **Flask**
* **Flask-WTF**
* **Flask-Bootstrap**
* **WTForms**

## Optimization

* **Adam Optimizer**
* **Learning Rate Scheduler**
* **Mean Squared Error (MSE) Loss**

## Development / Supporting Tools

* **Python Virtual Environment**
* **VS Code**
* **Git**
* **GitHub**

---

# Python Libraries

The major libraries used in the implementation include:

```text
torch
torchvision
Pillow
numpy
Flask
Flask-WTF
Flask-Bootstrap
WTForms
Werkzeug
tqdm
```

---

# Project Structure

```text
NST-with-AdaIN/
│
├── Demo_IO_Images/
│   ├── i-p/
│   │   ├── i_p image.jpg
│   │   ├── style 1.png
│   │   └── style 2.jpg
│   │
│   └── o-p/
│       ├── o_p style 1.jpg
│       └── o_p style 2.jpg
│
├── NST_Code/
│   │
│   ├── app.py
│   ├── train.py
│   ├── adain_algo.png
│   ├── vgg_normalised.pth
│   │
│   ├── content_data/
│   │   ├── avril.jpg
│   │   ├── blonde_girl.jpg
│   │   ├── brad_pitt.jpg
│   │   ├── chicago.jpg
│   │   ├── cornell.jpg
│   │   ├── flowers.jpg
│   │   ├── golden_gate.jpg
│   │   ├── lenna.jpg
│   │   ├── modern.jpg
│   │   ├── newyork.jpg
│   │   └── sailboat.jpg
│   │
│   ├── examples/
│   │   ├── content images
│   │   ├── style images
│   │   └── stylized outputs
│   │
│   ├── experiment/
│   │   └── final_exp/
│   │       ├── decoder_final.pth
│   │       ├── options.txt
│   │       └── sample outputs
│   │
│   └── utils/
│       ├── models.py
│       └── utils.py
│
└── README.md
```

The repository also contains demonstration input/output images and a trained decoder checkpoint.

---

# Training Configuration

The training script provides configurable parameters for the training pipeline.

Some important parameters include:

| Parameter           | Default |
| ------------------- | ------: |
| Final Image Size    |     256 |
| Content Image Size  |     512 |
| Style Image Size    |     512 |
| Batch Size          |       4 |
| Learning Rate       |  0.0001 |
| Learning Rate Decay | 0.00005 |
| Epochs              |       1 |
| Content Weight      |     1.0 |
| Style Weight        |       5 |

These parameters can be modified through command-line arguments in the training script.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/bibekn414/NST-with-AdaIN.git
```

Move into the project directory:

```bash
cd NST-with-AdaIN
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install torch torchvision pillow flask flask-wtf flask-bootstrap wtforms werkzeug tqdm
```

---

# Running the Web Application

Navigate to the code directory:

```bash
cd NST_Code
```

Run:

```bash
python app.py
```

The Flask application runs locally and can be accessed through the browser.

The application accepts:

* Content Image
* Style Image
* Alpha / Style Strength

and returns the generated stylized image.

---

# Example

The input consists of two images:

```text
Content Image + Style Image
      ↓
Neural Style Transfer
      ↓
Stylized Image
```

For example, the project contains sample content/style images and corresponding stylized outputs in the `Demo_IO_Images` and `examples` directories.

---

# Technical Concepts Demonstrated

This project demonstrates practical implementation of several important **Artificial Intelligence, Machine Learning, and Computer Vision** concepts:

### Deep Learning

* Convolutional Neural Networks
* Feature extraction
* Transfer learning
* Encoder–decoder architecture
* Model inference

### Computer Vision

* Image preprocessing
* Image representation
* Feature maps
* Image reconstruction
* Image stylization

### Neural Style Transfer

* Content representation
* Style representation
* Feature statistics
* Adaptive Instance Normalization
* Content loss
* Style loss

### PyTorch

* `torch.nn`
* Custom Dataset
* DataLoader
* Automatic differentiation
* GPU acceleration
* Model checkpoints
* Optimizers
* Learning-rate scheduling

### Web Application

* Flask
* File upload
* Image processing
* Model inference
* Dynamic output generation

---

# Why AdaIN?

Traditional optimization-based Neural Style Transfer can require iterative optimization for every new content-style image pair.

AdaIN provides a more efficient approach by directly transforming the content feature statistics using the style feature statistics.

This makes the approach suitable for **fast and flexible style transfer**, where different content and style images can be combined without retraining the complete network for every pair.

---

# Model Components

| Component               | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| VGG Encoder             | Extracts deep image features                   |
| AdaIN                   | Transfers style statistics                     |
| Decoder                 | Converts transformed features back to an image |
| Content Loss            | Preserves content representation               |
| Style Loss              | Matches style characteristics                  |
| Adam Optimizer          | Optimizes decoder parameters                   |
| Learning Rate Scheduler | Controls learning-rate decay                   |
| Flask                   | Provides interactive web interface             |
| PyTorch                 | Deep learning implementation                   |

---

# 🎯 Applications

Neural Style Transfer can be used in areas such as:

* Digital art generation
* Artistic image editing
* Photo stylization
* Creative content generation
* Image transformation
* Visual effects
* Computer vision research
* Generative AI applications
* Interactive image-editing tools
* Artistic content creation

---

# 🔮 Future Improvements

Possible improvements to the project include:

* Adding support for video style transfer
* Improving inference speed
* Supporting higher-resolution images
* Adding multiple style blending
* Adding a more advanced user interface
* Deploying the application using a cloud platform
* Adding GPU-optimized inference
* Supporting batch image processing
* Adding downloadable output functionality
* Improving model training with larger datasets
* Adding quantitative image-quality evaluation
* Comparing AdaIN with other Neural Style Transfer approaches

---

# Key Learning Outcomes

Through this project, the following concepts are explored:

* Understanding **Neural Style Transfer**
* Implementing **Adaptive Instance Normalization**
* Working with **pre-trained CNN models**
* Extracting deep feature representations
* Designing an encoder–decoder architecture
* Implementing custom training pipelines in PyTorch
* Designing content and style losses
* Training and saving neural network checkpoints
* Building a Flask-based AI application
* Integrating a deep learning model with a web interface
* Understanding practical computer vision workflows

---

# Keywords

```text
Neural Style Transfer
NST
Adaptive Instance Normalization
AdaIN
Deep Learning
Computer Vision
Generative AI
Image Stylization
Image Transformation
PyTorch
Python
VGG
VGG-19
CNN
Convolutional Neural Network
Encoder-Decoder
Feature Extraction
Feature Representation
Content Loss
Style Loss
MSE Loss
Transfer Learning
Adam Optimizer
Flask
Computer Vision
Image Processing
Deep Neural Networks
Artificial Intelligence
Machine Learning
GPU Acceleration
Model Inference
```

---

# Project Summary

This project implements a **Deep Learning-based Neural Style Transfer system using Adaptive Instance Normalization (AdaIN)**. A pre-trained VGG-based encoder extracts feature representations from the content and style images. AdaIN aligns the statistical properties of the content features with the style features, and a trained convolutional decoder reconstructs the final stylized image.

The decoder is trained using a combination of **content loss and style loss**, while the VGG encoder remains frozen. The trained model is then integrated into a **Flask web application**, allowing users to upload their own images and control the strength of style transfer using the alpha parameter.

Overall, the project demonstrates the complete pipeline from **deep feature extraction and neural style transfer to model training, inference, and web-based deployment**.

---

## Author

**Bibek Nayak**

GitHub: [@bibekn414](https://github.com/bibekn414)

---

## If you find this project useful

Feel free to explore the implementation, experiment with different content and style images, and extend the project with new Neural Style Transfer techniques.

**Built with Python, PyTorch, Computer Vision, and Deep Learning.**
