# Fine-tuning Wav2Vec2 for Tamil Speech Recognition

This repository contains the Jupyter Notebook and resources for fine-tuning the Wav2Vec2 model for Tamil speech recognition using the Hugging Face Transformers library.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [License](#license)

## Introduction

Wav2Vec2 is a state-of-the-art model for automatic speech recognition (ASR). This project aims to adapt Wav2Vec2 for the Tamil language, leveraging available datasets to improve performance in recognizing spoken Tamil.

## Requirements

To run this project, ensure you have the following installed:

- Python 3.7 or higher
- Jupyter Notebook
- PyTorch
- Transformers
- Datasets
- Librosa
- Soundfile
- [CUDA](https://developer.nvidia.com/cuda-downloads)

You can install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Dataset
We use Tamil Speech Dataset for fine-tuning the model. The dataset consists of audio files in Tamil along with their transcriptions. Please ensure you download the dataset and place it in an accessible directory.
Refer datapreprocessing.py

## Training
To fine-tune the Wav2Vec2 model, open the Jupyter Notebook located at /home/sltlab/priya/wav2vec.ipynb and follow the instructions provided within the notebook to execute the training process.

## Inference
After training, you can perform inference using the code snippets provided in the Jupyter Notebook. Ensure to replace the paths with your specific audio files.

##  Results
The performance of the model can be evaluated using standard metrics such as Word Error Rate (WER). The notebook contains sections on evaluating the model's performance.
```bash
pip install jiwer

```
```python
import jiwer

original_transcript = "God is great"  # Example script replace with your transcription
output_transcription = "good is great"

# Compute WER
wer = jiwer.wer(reference, hypothesis)
print(f"Word Error Rate (WER): {wer:.2f}")

```

## License
This project is licensed under the MIT License.

## Acknowledgments
For further reference please visit: [Fairseq Wav2Vec2](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)


