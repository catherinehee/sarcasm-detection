# Fine-tuning BERT for Sarcasm Detection with News Headlines

## Catherine He hecy@usc.edu

This project aims to classify the presence of sarcasm in news article headlines, leveraging the pretrained Transformer architecture BERT (Bidirectional Encoder Representations from Transformers), and Kaggle dataset News Headlines for Sarcasm Detection, a high quality dataset comprising of 28,619 headlines from news websites The Onion and HuffPost.

## Dataset
To classify sarcasm in news headlines, this project utilizes the **News Headlines Dataset for Sarcasm Detection**, composed of headlines collected from the two news websites: TheOnion and HuffPost. The chosen dataset offers several advantages over the commonly used Twitter dataset in classifying sarcasm. For instance, the inherent formal nature of news headlines produces error-free data samples and requires less data preprocessing; in addition, TheOnion solely aims to publish sarcastic news, which yields high-quality labels with minimal noise.

Over 28,000 records are provided, with a fair distribution of sarcastic (47% of records) and non-sarcastic (53% of records) headlines.

There are three attributes in the given dataset:

1.  `is_sarcastic`: a binary indicator for sarcasm
    
2.  `headline`: actual text of news article’s headline
    
3.  `article_link`: link to original news article

**Data Preprocessing**
For the purposes of this project, only the “headlines” and “is_sarcastic” (serving as the corresponding label) columns are used for each entry.
Since these headlines were professionally written under formal settings, this dataset doesn’t necessitate filtering other errors (spelling mistakes, grammatical errors, etc.)


## Model Development and Training

For this project, I chose to perform transfer learning by fine-tuning a pre-trained Transformer model: **BERT**. Pretrained Transformer architectures are efficient and enhanced by features (e.g. self-attention mechanism) that increase the model’s efficiency in NLP tasks and facilitates contextual understanding, among others. This makes Transformer models could discern the subtle linguistic cues indicative of sarcasm; more specifically, I used BERT for the current text classification task, due to its bidirectional nature and encoder architecture.

The **BertForSequenceClassification model** is the BERT model transformer with a linear sequence classification layer on top.

To facilitate the training process, I utilized *HuggingFace’s Trainer API* to fine-tune a BertForSequenceClassification model. This tool helps compact several components of the training and evaluation process we learned in curriculum (e.g. it forgoes the need to explicitly call forward/backward pass, specify gradient descent details, lr scheduler, optimizer, etc.).

Note: HuggingFace’s Trainer API sets default values for several components of the model’s training process, namely the AdamW optimizer, linear learning rate scheduler, clip gradient norm to default 1.0.

After experimenting with several variations of the following hyperparameters, I achieved the best results with the following values:

 - **Batch size**: 32 - authors recommended batch size of 16 or 32
 - **Learning rate** (Adam): 2e-5 - best performance out of set recommended for BERT (5e-5, 3e-5, 2e-5) 
 - **Number of epochs**: 4

## Model Evaluation & Results

To assess the performance of the BERT model, I used the following metrics: 
 1. **Accuracy** = (TP + TN) / (TP + TN + FP + FN) 
 2. **F1-score** = 2 * (PR * RE) / (PR + RE)

Loss was also calculated by default.

|               | # of samples | Loss | F1 | Accuracy |
| :---------------- | :------: | ----: | --:| --:|
| Validation (eval) |   1431   | 0.375 |0.839|0.846|
| Test           |   1431   | 0.334|0.860|0.866|

<img src="https://github.com/catherinehee/sarcasm-detection/assets/111953841/b6fe10c5-ac14-4df3-8f80-4bc45cd2d9af" width=400>
<img src="https://github.com/catherinehee/sarcasm-detection/assets/111953841/273e47d7-eeb0-432b-a2bb-0458471caeec" width=400>
<img src="https://github.com/catherinehee/sarcasm-detection/assets/111953841/0881b7bd-c319-409f-8316-1e29f8e20dee" width=600>



## Discussion

From a societal perspective, this project could yield several benefits. Sarcasm detection tools could assist modern attempts to reduce the dissemination of false information (from news headlines, in particular) and promote transparency in today’s society. 

Sentiment analysis is also a common NLP task; sarcasm presents a major obstacle in said task, due to its inherent ambiguity and linguistic nature of communicating the opposite of what is said. Thus, sarcasm detection, a task that is difficult even among human interactions, would indicate major progress in understanding human language.

One limitation to address is that the model has only been trained on headlines from two news sites, while there are many more news sources, whose affiliations, geographic belonging, and biases could affect the nature of its headlines. In addition, sarcasm isn’t limited to headlines–rather, it’s ubiquitous on many other platforms. 

In future studies, I would like to see how this model performs on a bigger dataset from a wider distribution of sources. In addition, this project could be expanded to train on data from colloquial contexts (i.e. social media), where sarcasm is more commonly displayed. 

In terms of implementation, it would also be interesting to observe how different architectures perform the task, like augmenting the BERT model with additional layers, or experimenting with other NLP techniques for text classification, like RNN/CNNs. 

## Citations
1. Misra, Rishabh and Prahal Arora. "Sarcasm Detection using News Headlines Dataset." AI Open (2023).

2. Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).

Note: This project also references materials from Fall 2023 CAIS++ Curriculum: “[L4] Intro to Natural Language Processing: Fine-Tuning BERT”
