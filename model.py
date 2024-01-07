import numpy as np
import pandas as pd
import os
import wandb
import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate

# Leverage Weight & Biases for logging
wandb.init(
    project='sarcasm-classifier',
)
output_dir = './models/bert-2'

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# ----------------
# Load the dataset
filepath = "Sarcasm_Headlines_Dataset_v2.json"
df = pd.read_json(filepath, lines=True)

# Data preprocessing
df = df.drop('article_link', axis = 1)
df = df.rename(columns={"is_sarcastic": "label", "headline": "headline"})

# 90-10 Train/Test split
train_df = df.sample(frac=0.9, random_state=328) # 90%
test_df = df.drop(train_df.index)
# 90-10 Train/Val split
val_df = train_df.sample(frac=0.1, random_state=290)
train_df = train_df.drop(val_df.index)

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", lowercase=True)

# Convert from Pandas DF to HuggingFace Dataset object
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
test_ds = Dataset.from_pandas(test_df)

print("Number of training samples", len(train_df))
print("Number of validation samples", len(val_df))
print("Number of test samples", len(test_df))

max_len = 0
# For every sentence...
for hl in df["headline"]:
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = ["[CLS]"] + tokenizer.tokenize(hl) + ["[SEP]"]
    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids)) 
print('Max sentence length: ', max_len)


# Tokenize each headline with BertTokenizer
def get_features(data):
  input_encodings= tokenizer(data["headline"], # Headline to encode
        add_special_tokens=True, # Add '[CLS]' and '[SEP]'
        max_length=max_len, 
        pad_to_max_length=True, # Pads to max_len 
        return_attention_mask=True) # Construct attention masks
  return input_encodings

# Retrieving features from each dataset split (input_ids, attention_mask, label)
train_data = train_ds.map(get_features)
validation_data = val_ds.map(get_features)
test_data = test_ds.map(get_features)

# -------------
# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top; ensure the model does NOT outputs attentions and hidden_states
config = BertConfig.from_pretrained("bert-base-uncased", output_attentions=False, output_hidden_states=False, num_labels=2)
model = BertForSequenceClassification(config)

BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 4
'''
Notable default values: 
    optim=adamw_torch
    lr_scheduler_type="linear"
    warmup_steps = 0
    max_grad_norm = 1.0
'''

args = TrainingArguments(
    output_dir=output_dir,
    save_strategy="steps", # save, evaluate, and log by steps
    evaluation_strategy = "steps",
    eval_steps=100,
    logging_steps=100,
    save_steps=500,
    save_total_limit=1, # Saves best checkpoint at end (based on accuracy)

    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="wandb"
)

# Compute accuracy metric to assess performance
def compute_metrics(pred):
    acc_metric = evaluate.load("accuracy") # Load HF's metric for accuracy: Accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1_metric = evaluate.load("f1") # Load HF's metric for accuracy: Accuracy = (TP + TN) / (TP + TN + FP + FN)
    logits, labels = pred # Get the loss and "logits" output by the model.
    predictions = np.argmax(logits, axis = 1)
    accuracy = acc_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]

    return {
        "accuracy": accuracy,
        "f1": f1
    }


# Construct Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data, # Dataset to use for training 
    eval_dataset=validation_data, # Dataset to use for evaluation
    compute_metrics=compute_metrics, # Function used to assess evaluation
    
)

trainer.train()


# ----- 
# Results / Evaluation
print("Validation set results:\n")
val_results = trainer.evaluate()
print(val_results)

print("Test set results:\n")
test_results = trainer.evaluate(eval_dataset=test_data)
print(test_results)

trainer.save_model(output_dir) # Saving model for later testing