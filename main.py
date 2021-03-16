"""
Sentence classification project, internship at National Science Library，Chinese Academy of Sciences.
main.py: training & testing of the model.
         Adapted from: https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
Author: Feilian Xie <fxie49@gatech.edu>
"""

# Uncomment the following to run as a jupyter notebook
# !pip install py-cpuinfo
# !pip install -qq transformers
# !pip install pandas
# !pip install seaborn
# !pip install scikit-learn

import gc
import json
import logging
import os
import random
import time
import warnings

import numpy as np
import pandas as pd  # csv format data
import torch
import torch.cuda as cutorch
import torch.nn.functional as F
from cpuinfo import get_cpu_info
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader  # Create custom PyTorch dataset and dataloader
from transformers import BertModel, BertForSequenceClassification, BertTokenizerFast, AdamW, \
    get_linear_schedule_with_warmup, logging as tlogging

from ANSI_color_codes import *


class CustomDataset(Dataset):
    """
    Provide a customized dataset.
    """

    def __init__(self, sentences, is_type, tokenizer, max_len):
        self.sentences = sentences
        self.is_type = is_type
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.is_type)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        is_type = self.is_type[item]

        encoding = self.tokenizer(
            text=sentence,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch torch.Tensor objects
        )

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'is_type': torch.tensor(is_type, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    """
    Create a customized PyTorch dataloader (which combines a dataset and a sampler, and provides an iterable over the given dataset.)
    :param df: pandas dataframe.
    :param tokenizer: BERT tokenizer of huggingface.
    :param max_len: maximum input length of the tokenizer.
    :param batch_size: number of data instances loaded at a time.
    :return: a customized PyTorch dataloader.
    """

    ds = CustomDataset(
        sentences=df["2"].to_numpy(),
        is_type=df["IsType"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4
    )


class SentenceClassifier(nn.Module):
    """
    Our model:
    BERT encoder followed by a MLP classifier with or without a hidden layer (followed by a dropout layer).
    In either case, a dropout layer is applied to BERT's output.
    """

    def __init__(self, n_classes, drop_rates, xavier_init=False, hidden_size=None):
        super(SentenceClassifier, self).__init__()
        '''
        # Uncomment the triple single quote and comment the following line when running the script for the first time.
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        bert_model.save_pretrained('./Model/')
        '''
        self.bert = BertModel.from_pretrained('./Model/')

        if hidden_size is None:  # No hidden layer
            self.hidden_layer = False
            self.drop = nn.Dropout(p=drop_rates)
            self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        else:  # Single hidden layer
            self.hidden_layer = True
            self.drop1 = nn.Dropout(p=drop_rates[0])
            self.hidden = nn.Linear(self.bert.config.hidden_size, hidden_size)
            self.drop2 = nn.Dropout(p=drop_rates[1])
            self.out = nn.Linear(hidden_size, n_classes)

        if xavier_init:
            nn.init.xavier_uniform_(self.out.weight, gain=1.0)  # Xavier initialization

    def forward(self, input_ids, attention_mask):
        model_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = model_output[1]
        if self.hidden_layer:
            return self.out(self.drop2(self.hidden(self.drop1(pooled_output))))
        else:
            return self.out(self.drop(pooled_output))


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, is_bfsc=False):
    """
    Models one training epoch (forward prop + back prop).
    :param model: instance of the class "SentenceClassifier" when is_bfsc=False, instance of "BertForSequenceClassification" otherwise.
    :param data_loader: instance of the class "create_data_loader".
    :param loss_fn: loss function.
    :param optimizer:
    :param device:
    :param scheduler: scheduler which updates the learning rate.
    :param n_examples: number of training examples.
    :param is_bfsc: True if "BertForSequenceClassification" provides the model.
    :return: training precision, recall, f1 score, accuracy, loss per training example.
    """

    model = model.train()  # Put Model in train mode (different behavior for the dropout layer)

    # For one epoch
    tot_loss = 0
    tn_tot, fp_tot, fn_tot, tp_tot = 0, 0, 0, 0  # Confusion matrix

    for d in data_loader:  # For one batch of data
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["is_type"].to(device)

        # Forward prop
        if is_bfsc:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            loss = outputs[0]
            outputs = outputs[1]
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss = loss_fn(outputs, targets)

        _, preds = torch.max(outputs, dim=1)  # preds.get_device()?
        tot_loss += loss.item()  # The item() method extracts the loss's value as a Python float

        targets = targets.cpu()
        preds = preds.cpu()
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        tn_tot += tn
        fp_tot += fp
        fn_tot += fn
        tp_tot += tp

        # Back prop
        loss.backward()  # Computes dloss/dx for every parameter x which has requires_grad=True. Then x.grad += dloss/dx
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # Updates the value of x using the gradient x.grad. x += -lr * x.grad
        scheduler.step()  # Update the learning rate
        optimizer.zero_grad()  # Clears x.grad for every parameter x

    precision = tp_tot / (tp_tot + fp_tot)
    recall = tp_tot / (tp_tot + fn_tot)
    f1 = 2 * precision * recall / (precision + recall)
    if np.isnan(f1):
        f1 = 0
    acc = (tp_tot + tn_tot) / n_examples

    return precision, recall, f1, acc, tot_loss / n_examples


def eval_model(model, data_loader, loss_fn, device, n_examples, is_bfsc=False):
    """
    Evaluation of model performance (forward prop) on validation/testing dataset.
    :param model: instance of the class "SentenceClassifier" when is_bfsc=False, instance of "BertForSequenceClassification" otherwise.
    :param data_loader: instance of the class "create_data_loader".
    :param loss_fn: loss function.
    :param device:
    :param n_examples: number of validation/testing examples.
    :param is_bfsc: True if "BertForSequenceClassification" provides the model.
    :return: validation/testing precision, recall, f1 score, accuracy, loss per validation/testing example.
    """

    model = model.eval()  # Put Model in eval mode (different behavior for the dropout layer)

    tot_loss = 0
    tn_tot, fp_tot, fn_tot, tp_tot = 0, 0, 0, 0  # Confusion matrix

    with torch.no_grad():  # Temporarily set all the requires_grad flag to false
        for d in data_loader:  # For one batch of data
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["is_type"].to(device)

            # Forward prop
            if is_bfsc:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets
                )
                loss = outputs[0]
                outputs = outputs[1]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = loss_fn(outputs, targets)

            tot_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)

            targets = targets.cpu()
            preds = preds.cpu()
            tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
            tn_tot += tn
            fp_tot += fp
            fn_tot += fn
            tp_tot += tp

    precision = tp_tot / (tp_tot + fp_tot)
    recall = tp_tot / (tp_tot + fn_tot)
    f1 = 2 * precision * recall / (precision + recall)
    if np.isnan(f1):
        f1 = 0
    acc = (tp_tot + tn_tot) / n_examples

    return precision, recall, f1, acc, tot_loss / n_examples


def get_predictions(model, data_loader, device, save_preds_as_csv=True, is_bfsc=False):
    """
    Generate predictions with model.
    :param model: instance of the class "SentenceClassifier" when is_bfsc=False, of "BertForSequenceClassification" otherwise.
    :param data_loader: instance of the class "create_data_loader".
    :param device:
    :param save_preds_as_csv: True if predictions are also saved in a csv file while outputting by the function as a pandas dataframe.
    :param is_bfsc: True if "BertForSequenceClassification" provides the model.
    :return: predictions (pandas dataframe).
    """

    model = model.eval()  # Put Model in eval mode (different behavior for the dropout layer)

    sentences = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():  # Temporarily set all the requires_grad flag to false
        for d in data_loader:  # For one batch of data
            texts = d["sentence"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["is_type"].to(device)

            if is_bfsc:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets
                )[1]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)

            sentences.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    sentences = np.array(sentences)
    predictions = torch.stack(predictions).cpu().numpy()
    prediction_probs = torch.stack(prediction_probs).cpu().numpy()
    prediction_probs = prediction_probs[np.arange(prediction_probs.shape[0]), predictions]
    real_values = torch.stack(real_values).cpu().numpy()

    # Build dataframe containing test results
    result = np.vstack((sentences, predictions, prediction_probs, real_values)).transpose()
    df_result = pd.DataFrame(result, columns=['Sentences', 'Predictions', 'Probabilities', 'True labels'])
    if save_preds_as_csv:
        df_result.to_csv("./Results/Test_preds.csv", index=False)
    else:
        return df_result


if __name__ == '__main__':
    # Define logger
    logging.basicConfig(
        filename="./Logging/" + time.strftime("%Y%m%d_%H%M%S") + ".log",
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d - %(name)s - %(filename)s - line %(lineno)s - %(levelname)s - %(message)s',
        datefmt="%Z %Y-%m-%d %H:%M:%S"  # Time zone yyyy-MM-dd HH:mm:ss.SSS
    )
    logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(OKBLUE + '%(asctime)s - %(message)s' + RESET, "%H:%M:%S"))
    logger.addHandler(console_handler)

    # Initialization
    logger.info('Initializing:')

    code_testing = False  # Whether we are testing the code
    logger.info(f'Code being tested: {code_testing}')

    is_bfsc = False  # Whether the "BertForSequenceClassification" model is used instead of the plain BERT model
    logger.info(f'"BertForSequenceClassification" model is used instead of the plain BERT model: {is_bfsc}')

    dataset = 1  # 1: type one sentences; 2: type two sentences
    logger.info(f'Dataset: type {dataset} sentences')

    warnings.filterwarnings("ignore")  # Ignore Python warnings to suppress invalid value warnings
    tlogging.set_verbosity_error()  # Transformers: Set the verbosity to the ERROR level to suppress warnings

    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Avoid warning messages caused by multiprocessing (dataloader)
    logger.info(f'Tokenizer parallelization: {os.environ["TOKENIZERS_PARALLELISM"]}')

    cpu = get_cpu_info()['brand_raw']
    logger.info(f'CPU type: {cpu}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    RANDOM_SEED = 42  # Constants are named all CAPITAL
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    logger.info(f'Random seed: {RANDOM_SEED}')

    PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    '''
    tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL_NAME)
    tokenizer.save_pretrained("./Model/")
    '''
    tokenizer = BertTokenizerFast.from_pretrained("./Model/")

    if code_testing:
        EPOCHS = 2
    else:
        EPOCHS = 10
    logger.info(f'Number of epochs: {EPOCHS}')
    '''
    Max length chosen according to training sequence (token) length distribution.
    Max length 256 -> ~8/13 gb GPU memory needed for batch size = 16/32
    '''
    MAX_LEN = 256
    logger.info(f'Max length = {MAX_LEN}')

    # Load datasets
    logger.info('')
    logger.info("Loading datasets:")

    if dataset == 1:
        df = pd.read_csv("Datasets/Dataset_1.csv")
    elif dataset == 2:
        df = pd.read_csv("Datasets/Dataset_2.csv")
    else:
        logger.critical(f'Wrong dataset type: {dataset}')
        raise Exception('Dataset/sentence type can only be 1 or 2!')
    positive_sentence_ratio = df['IsType'].value_counts()[1] / len(df.index)  # Dataset balanced?
    logger.info(f'Type {dataset} sentence ratio: {positive_sentence_ratio}')

    if code_testing:  # Smaller train/dev/test set for code testing
        df_train = df[0:10]
        df_val = df[10:20]
        df_test = df[20:30]
    else:
        df_train = df[0:10000]
        df_val = df[10000:15000]
        df_test = df[15000:20000]
    logger.info(
        f'Number of sentences used for training/validation/testing: {len(df_train.index)}/{len(df_val.index)}/{len(df_test.index)}')

    # Grid search
    logger.info('')
    logger.info("Grid search begins: ")

    learning_rates = [2e-6, 6e-6, 1e-5]
    batch_sizes = [16, 32]
    logger.info(f'Batch sizes: {batch_sizes}')
    logger.info(f'Initial learning rates: {learning_rates}')
    if not is_bfsc:
        drop_rates = 0.3  # Drop rates of the dropout layers
        logger.info(f"Drop rates: {drop_rates}")
        xavier_init = False
        logger.info(f'Xavier initialization: {xavier_init}')
        hidden_size = None  # None: no hidden layer; 512: single hidden layer of 512 neurons
        logger.info(f'Hidden layer: {hidden_size}')

    loss_fn = nn.CrossEntropyLoss().to(device)

    val_f1_grid = np.zeros((len(batch_sizes), len(learning_rates)))
    epoch_grid = np.zeros((len(batch_sizes), len(learning_rates)))
    best_val_f1_overall = 0

    t_tot = 0  # Total grid search time

    for i, bs in enumerate(batch_sizes):
        for j, lr in enumerate(learning_rates):

            if is_bfsc:
                model = BertForSequenceClassification.from_pretrained(
                    PRE_TRAINED_MODEL_NAME,
                    num_labels=2,
                    output_attentions=False,  # Whether the model returns attentions weights.
                    output_hidden_states=False,  # Whether the model returns all hidden-states.
                )
            else:
                model = SentenceClassifier(n_classes=2, drop_rates=drop_rates, xavier_init=xavier_init,
                                           hidden_size=hidden_size)

            model = model.to(device)  # Move Model to GPU

            '''
            Custom dataset and dataloader.
            Dataloader (an iterable): load a batch of data each iteration
            '''
            train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, bs)
            val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, bs)

            optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)

            total_steps = len(train_data_loader) * EPOCHS  # len(train_data_loader) = number of (train) batches

            scheduler = get_linear_schedule_with_warmup(  # Scheduler: change learning rate of optimizer
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )

            logger.info('-' * 100)
            logger.info(f'Batch size: {bs}, initial learning rate: {lr}, total steps: {total_steps}')
            logger.info('-' * 100)

            best_val_f1 = 0
            best_epoch = 0
            t0 = time.time()

            for epoch in range(EPOCHS):  # Training loop

                logger.info(f'Epoch {epoch + 1}/{EPOCHS}')

                train_precision, train_recall, train_f1, train_acc, train_loss = train_epoch(
                    model,
                    train_data_loader,
                    loss_fn,
                    optimizer,
                    device,
                    scheduler,
                    len(df_train),
                    is_bfsc
                )

                logger.info(
                    f'Train loss {train_loss} accuracy {train_acc} f1 {train_f1} precision {train_precision} recall '
                    f'{train_recall}')

                val_precision, val_recall, val_f1, val_acc, val_loss = eval_model(
                    model,
                    val_data_loader,
                    loss_fn,
                    device,
                    len(df_val),
                    is_bfsc
                )

                logger.info(
                    f'Val loss {val_loss} accuracy {val_acc} f1 {val_f1} precision {val_precision} recall {val_recall}')

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch + 1
                    if val_f1 > best_val_f1_overall:
                        torch.save(model, 'Model/best_model_state.bin')
                        best_val_f1_overall = best_val_f1
                        best_epoch_overall = best_epoch
                        best_bs = bs
                        best_lr = lr

            val_f1_grid[i, j] = best_val_f1
            epoch_grid[i, j] = best_epoch
            t = time.time() - t0
            t_tot += t

            logger.info('')
            logger.info(f'Best validation f1 {best_val_f1} reached at epoch {best_epoch}')
            logger.info(f'Training time for {EPOCHS} epochs: {t / 60} min')

    logger.info('-' * 100)
    logger.info('Grid search finished')
    logger.info(f'Time consumed: {t_tot / 60} min')
    logger.info(f'Validation f1 values {val_f1_grid} reached at epochs {epoch_grid} respectively')
    logger.info(
        f'Best parameters: batch size = {best_bs}, learning rate = {best_lr}, training epoch = {best_epoch_overall}')

    # Load best model
    logger.info('')
    logger.info('Loading best model: ')
    model = torch.load('Model/best_model_state.bin')
    model.eval()
    model = model.to(device)  # Move Model to GPU

    # Test Model performance on testing set
    logger.info('Testing model performance: ')
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, best_bs)
    test_precision, test_recall, test_f1, test_acc, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(df_test),
        is_bfsc
    )

    logger.info(f'Test accuracy {test_acc} f1 {test_f1} precision {test_precision} recall {test_recall}')

    # Get predictions for test set
    _ = get_predictions(
        model,
        test_data_loader,
        device,
        save_preds_as_csv=True,
        is_bfsc=is_bfsc
    )

    # Save experiment record to .json file
    exp_record = {
        "Code testing": code_testing,
        "Dataset/sentence type": dataset,
        "Tokenizer parallelization": os.environ["TOKENIZERS_PARALLELISM"],
        "CPU": cpu,
        "Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "Random seed": RANDOM_SEED,
        "Bert model": PRE_TRAINED_MODEL_NAME,
        "Is 'BertForSequenceClassification'": is_bfsc,
        "Nb of sentences for training/validation/testing": [len(df_train.index), len(df_val.index), len(df_test.index)],
        "Total grid search time (min)": t_tot / 60,
        "Epochs": EPOCHS,
        "Max length": MAX_LEN,
        "Initial learning rates": learning_rates,
        "Batch sizes": batch_sizes,
        "Drop rate": drop_rates if not is_bfsc else None,
        "Xavier initialization": xavier_init if not is_bfsc else None,
        "Hidden layer": hidden_size if not is_bfsc else None,
        "Validation f1 values": val_f1_grid.tolist(),
        "Corresponding epoch numbers": epoch_grid.tolist(),
        "Best parameters (bs, lr, epoch)": [best_bs, best_lr, best_epoch_overall],
        "Testing performances (acc, f1, precision, recall)": [test_acc, test_f1, test_precision, test_recall]
    }

    with open("Results/exp_record_" + time.strftime("%Y%m%d_%H%M%S") + ".json", "w") as outfile:
        json.dump(exp_record, outfile)

    # Release GPU memory
    logger.info('')
    logger.info('Releasing GPU memory: ')
    del model, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, scheduler
    gc.collect()  # Collect garbage
    cutorch.empty_cache()
    logger.info(f'GPU memory occupied: {cutorch.memory_reserved() / 1e9} gb')
