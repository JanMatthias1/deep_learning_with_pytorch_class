# Deep Learning with PyTorch Class

This repository contains six deep learning assignments and one final project. The work covers image classification, self-supervised learning, object detection, video classification, vision transformers, reinforcement learning, and molecular property prediction.

## Assignments

### Assignment 1: CIFAR-100 Classification
Builds a basic neural-network image classifier for CIFAR-100 using PyTorch. The assignment covers custom dataset loading, dataloaders, training loops, evaluation, and plotting loss/accuracy curves across batch-size experiments.

### Assignment 2: Self-Supervised Learning and Transfer
Uses rotation prediction as a self-supervised pretraining task, then transfers learned representations to supervised image classification on the Intel Images dataset. Experiments compare multiple seeds and model variants on GPU cluster jobs.

### Assignment 3: PASCAL VOC Object Detection
Prepares the PASCAL VOC 2012 dataset for YOLO-format object detection and trains YOLOv5 models with both pretrained and from-scratch settings. Extra notebooks explore YOLOv3 and YOLOv8 workflows.

### Assignment 4: UCF50 Video Classification
Implements action recognition on UCF50 using an LRCN architecture: a ResNet frame encoder followed by an LSTM temporal model. The pipeline includes frame extraction, group-aware train/validation/test splits, training, and evaluation.

### Assignment 5: CIFAR-100 with Swin Transformer
Trains a Swin-Base vision transformer from scratch on CIFAR-100. The assignment includes data augmentation, validation splitting, checkpointing, early stopping, and test-set metrics such as accuracy, F1, ROC-AUC, and confusion matrix.

### Assignment 6: Deep Q-Network
Implements a DQN agent from scratch for CartPole-v1 using PyTorch and Gymnasium. The project includes replay buffer training, target networks, epsilon-greedy exploration, evaluation runs, and W&B metric logging.

## Final Project: COMET for Lipid Nanoparticle Design

The final project extends COMET, a transformer-based model for predicting lipid nanoparticle transfection efficacy. It combines a frozen Uni-Mol backbone with GNN-derived embeddings from GIN, SchNet, and 3D Infomax to compare baseline and augmented molecular representations.

## Repository Layout

```text
assignment_1/   CIFAR-100 neural network classification
assignment_2/   Self-supervised rotation prediction and transfer learning
assignment_3/   PASCAL VOC object detection with YOLO
assignment_4/   UCF50 video action recognition
assignment_5/   Swin Transformer on CIFAR-100
assignment_6/   DQN reinforcement learning on CartPole
COMET_final/    Final molecular prediction project
```
