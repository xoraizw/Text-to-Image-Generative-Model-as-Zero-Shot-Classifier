# Text-to-Image Generative Model as Zero-Shot Classifier

## Overview

This project implements a zero-shot image classification algorithm using text-to-image generative models, specifically leveraging the power of latent diffusion models like Stable Diffusion. Inspired by the research paper *"Text-to-Image Diffusion Models are Zero-Shot Classifiers"* (Clark & Jaini, 2023), the goal is to classify images based on textual prompts by utilizing the denoising capabilities of diffusion models.

The deliverable is a function that accepts an image filepath and a list of category labels and predicts the most likely category for the image.

---

## Features

- Implements zero-shot classification using Stable Diffusion (v1-4).
- Utilizes the Diffusers library for efficient model operations.
- Operates in latent space for faster computation compared to pixel space.
- Handles the encoding, denoising, and decoding pipeline seamlessly.
- Outputs predictions based on aggregated denoising scores.

---

## Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed along with a CUDA-enabled GPU for acceleration.

### Install Dependencies

```bash
pip install --upgrade diffusers accelerate transformers datasets
```

---

## Usage

### 1. Importing Libraries

```python
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel
```

### 2. Loading the Model

```python
# Load VAE, UNet, and Tokenizer
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
```

### 3. Running Classification

```python
def classify_image(filepath, categories):
    # Your implementation here
    pass

prediction = classify_image("image.jpg", ["cat", "dog", "bird"])
print("Predicted Category:", prediction)
```

---

## Project Learnings

### Technical Skills

1. **Latent Diffusion Models**: Learned about latent-space operations for efficient noise addition and removal.
2. **Diffusers Library**: Explored and applied its modules for tasks like encoding, denoising, and decoding.
3. **Text Embeddings**: Integrated text embeddings from the CLIP model to condition the diffusion process.
4. **Zero-Shot Learning**: Developed classification techniques that require no explicit training on specific datasets.

### Research Insights

- **Power of Generative Models**: Realized how generative diffusion models outperform contrastive learning methods in certain zero-shot tasks.
- **Compositional Reasoning**: Investigated how attribute binding works better with generative models than contrastive models like CLIP.

---

## Technologies Used

- **Frameworks**: PyTorch, Hugging Face Diffusers
- **Models**: Stable Diffusion (v1-4), CLIP (Text and Vision Transformer)
- **Libraries**: NumPy, Matplotlib, TQDM

---

## References

1. Clark, K., & Jaini, P. (2023). [Text-to-Image Diffusion Models are Zero-Shot Classifiers](https://arxiv.org/abs/2303.15233).
2. [Hugging Face Diffusers Library](https://huggingface.co/docs/diffusers/).

---

## Future Work

- Extend the classifier to handle larger datasets with dynamic prompt engineering.
- Optimize for real-time inference by reducing computational overhead.
- Explore fine-tuning Stable Diffusion for domain-specific zero-shot classification.

---

## Acknowledgments

Special thanks to the course lectures and the Diffusers library documentation for providing a strong foundation for this project.

