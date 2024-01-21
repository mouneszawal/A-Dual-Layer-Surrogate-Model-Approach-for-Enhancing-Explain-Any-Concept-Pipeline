# A Dual-Layer Surrogate Model Approach for Enhancing Explain Any Concept Pipeline
Code for the paper "A Dual-Layer Surrogate Model Approach for Enhancing Explain Any Concept Pipeline".

## Requirements

This project requires the following software:

- Python 3+ 
- PyTorch (torch, torchvision)
- segment-anything
- sklearn
- PIL
- pycocotools
- numpy
- tqdm

Make sure you have the required software installed before proceeding.

## Overview
Experiments to enhance the performance proposed in the original paper "Explain Any Concept: Segment Anything Meets Concept-Based Explanation (EAC)" [Paper](https://openreview.net/forum?id=X6TBBsz9qi).

## Downloading the SAM backbone
We use ViT-H as our default SAM model. For downloading the pre-train model and installation dependencies, please refer [SAM repo](https://github.com/facebookresearch/segment-anything#model-checkpoints).

## Explain a hummingbird on your local pre-trained ResNet-50!
Simply run the following command:
```
python demo_samshap.py
```

## Running the evaluation codes
- Get ImageNet Dataset, please check this [link](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) to download it. Then move Data folder to imagnet directory in this project.
- Get COCO Dataset, please check this [link](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) to download it.
- You my need to modify some paths in the codes. ( SAM model's path, ImageNet dataset's path , etc.)
- For Insertion evaluation, set the delete variable in the main_imagenet.py and coco_evaluation.py to False.
- To activate the enhanced model, set net_type variable to "Enhanced", for the original model set it to "orig".
- GPU is required.
- Run the code simply from the command line or from any IDE.

## Acknowledgment
This code was built on the original paper's codes. please refer [original paper code](https://github.com/Jerry00917/samshap/tree/main).
