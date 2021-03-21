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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # csv format data
import seaborn as sns  # Statistical plot
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
    :param tokenizer: BERT tokenizer (huggingface).
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
    logger.debug("Dataloader: shuffle=True")

    return DataLoader(
        ds,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4
    )


class SentenceClassifier(nn.Module):
    """
    Our model:
    BERT encoder followed by a MLP classifier with or without a hidden layer (followed by a dropout layer).
    In either case, a dropout layer is applied to BERT's output.
    """

    def __init__(self, n_classes, drop_rates, xavier_init=False, hidden_size=None, use_amp=True):
        super(SentenceClassifier, self).__init__()

        '''
        # Uncomment the triple single quote and comment the following line when running the script for the first time.
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        bert_model.save_pretrained('./Model/')
        '''
        self.bert = BertModel.from_pretrained('./Model/')
        self.use_amp = use_amp

        if hidden_size is None:  # No hidden layer
            self.hidden_layer = False
            self.drop = nn.Dropout(p=drop_rates)
            self.embed_to_logits = nn.Linear(self.bert.config.hidden_size, n_classes)
        else:  # Single hidden layer
            self.hidden_layer = True
            self.drop_embed = nn.Dropout(p=drop_rates[0])
            self.embed_to_hidden = nn.Linear(self.bert.config.hidden_size, hidden_size)
            self.drop_hidden = nn.Dropout(p=drop_rates[1])
            self.hidden_to_logits = nn.Linear(hidden_size, n_classes)

        if xavier_init:
            nn.init.xavier_uniform_(self.out.weight, gain=1.0)  # Xavier initialization

    def forward(self, input_ids, attention_mask):
        with cutorch.amp.autocast(enabled=self.use_amp):
            model_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_output = model_output[1]
            if self.hidden_layer:
                hidden_units = F.relu(self.embed_to_hidden(self.drop_embed(pooled_output)))
                logits = self.hidden_to_logits(self.drop_hidden(hidden_units))
            else:
                logits = self.embed_to_logits(self.drop(pooled_output))
            return logits


def train_epoch(model, data_loader, loss_fn, optimizer, device, num_instances, scheduler, logger, scaler,
                use_bfsc=False, use_amp=True):
    """
    Models one training epoch (forward prop + back prop).
    :param model: instance of the class "SentenceClassifier" when use_bfsc=False, instance of "BertForSequenceClassification" otherwise.
    :param data_loader: instance of the class "create_data_loader".
    :param loss_fn: loss function.
    :param optimizer:
    :param device:
    :param num_instances:
    :param scheduler: scheduler which updates the learning rate.
    :param logger:
    :param scaler: provides gradient scaling necessary when automatic mixed precision (AMP) is used.
    :param use_bfsc: True if "BertForSequenceClassification" provides the model.
    :param use_amp: whether to use automatic mixed precision or not.
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
        num_instance_batch = list(input_ids.shape)[0]  # Number of data instances in batch (full batch or not)

        # Forward prop
        with cutorch.amp.autocast(enabled=use_amp):
            if use_bfsc:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets
                )
                loss = outputs[0]  # "CrossEntropyLoss"
                outputs = outputs[1]  # logits
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )  # logits
                loss = loss_fn(outputs, targets)

        _, preds = torch.max(outputs, dim=1)  # preds.get_device()?
        tot_loss += loss.item() * num_instance_batch  # The item() method extracts the loss's value as a Python float

        targets = targets.cpu()
        preds = preds.cpu()
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        tn_tot += tn
        fp_tot += fp
        fn_tot += fn
        tp_tot += tp

        # Back prop
        '''
        Computes dloss/dx for every parameter x which has requires_grad=True. Then x.grad += dloss/dx.
        Use loss.backward() without gradient scaling.
        '''
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        '''
        Updates the value of x using the gradient x.grad. x += -lr * x.grad, use optimizer.step() without gradient 
        scaling.
        '''
        scaler.step(optimizer)
        scaler.update()  # Updates the scale for next iteration/batch.
        scheduler.step()  # Update the learning rate
        optimizer.zero_grad()  # Clears x.grad for every parameter x

    precision = tp_tot / (tp_tot + fp_tot)
    recall = tp_tot / (tp_tot + fn_tot)
    f1 = 2 * precision * recall / (precision + recall)
    if np.isnan(f1):
        f1 = 0
    acc = (tp_tot + tn_tot) / num_instances
    ave_loss = tot_loss / num_instances

    logger.info(f'Train loss (ave) {ave_loss} accuracy {acc} f1 {f1} precision {precision} recall {recall}')

    return precision, recall, f1, acc, ave_loss


def eval_model(model, data_loader, loss_fn, device, num_instances, logger, use_bfsc=False, use_amp=True):
    """
    Evaluation of model performance (forward prop) on validation/testing dataset.
    :param model: instance of the class "SentenceClassifier" when use_bfsc=False, instance of "BertForSequenceClassification" otherwise.
    :param data_loader: instance of the class "create_data_loader".
    :param loss_fn: loss function.
    :param device:
    :param num_instances: number of validation/testing examples.
    :param use_bfsc: True if "BertForSequenceClassification" provides the model.
    :param logger:
    :param use_amp: whether to use automatic mixed precision or not.
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
            num_instance_batch = list(input_ids.shape)[0]  # Number of data instances in batch (full batch or not)

            # Forward prop
            with cutorch.amp.autocast(enabled=use_amp):
                if use_bfsc:
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

            tot_loss += loss.item() * num_instance_batch
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
    acc = (tp_tot + tn_tot) / num_instances
    ave_loss = tot_loss / num_instances

    logger.info(f'Eval loss (ave) {ave_loss} accuracy {acc} f1 {f1} precision {precision} recall {recall}')

    return precision, recall, f1, acc, ave_loss


def get_predictions(model, data_loader, device, save_preds_as_csv=True, use_bfsc=False, use_amp=True):
    """
    Generate predictions with model.
    :param model: instance of the class "SentenceClassifier" when use_bfsc=False, of "BertForSequenceClassification" otherwise.
    :param data_loader: instance of the class "create_data_loader".
    :param device:
    :param save_preds_as_csv: True if predictions are also saved in a csv file while outputting by the function as a pandas dataframe.
    :param use_bfsc: True if "BertForSequenceClassification" provides the model.
    :param use_amp: whether to use automatic mixed precision or not.
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

            with cutorch.amp.autocast(enabled=use_amp):
                if use_bfsc:
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
    # Initialization
    logging.basicConfig(  # Define logger
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

    logger.info('Initializing:')

    code_testing = True  # Whether we are testing the code
    logger.info(f'Code being tested: {code_testing}')

    use_bfsc = False  # Whether the "BertForSequenceClassification" model is used instead of the plain BERT model
    logger.info(f'"BertForSequenceClassification" model is used instead of the plain BERT model: {use_bfsc}')

    '''
    The use of automatic mixed precision (AMP) leads to large speed gain on Nvidia datacenter gpus (rich of tensor 
    cores) like V100/A100, or relatively smaller speed gain on desktop gpus like GeForce RTX 3080/3090.
    '''
    use_amp = True if torch.cuda.is_available() else False
    logger.info(f'Use automatic mixed precision: {use_amp}')

    dataset = 1  # 1: type one sentences; 2: type two sentences
    logger.info(f'Dataset: type {dataset} sentences')

    warnings.filterwarnings("ignore")  # Ignore Python warnings to suppress invalid value warnings
    tlogging.set_verbosity_error()  # Transformers: Set the verbosity to the ERROR level to suppress warnings

    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Avoid warning messages caused by multiprocessing (dataloader)
    logger.info(f'Tokenizer parallelization: {os.environ["TOKENIZERS_PARALLELISM"]}')

    cpu = get_cpu_info()['brand_raw']
    logger.info(f'CPU: {cpu}')

    # GPUs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    logger.info(f"Device count: {gpu_count}")
    device_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device name(s): {device_names}")
    # os.environ['MASTER_ADDR'] = '127.0.0.1'  # '127.0.0.1' = 'localhost'
    # os.environ['MASTER_PORT'] = '1234'  # a random number

    RANDOM_SEED = 42  # Constants are named ALL CAPITAL
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
        EPOCHS = 4
    logger.info(f'Number of epochs: {EPOCHS}')

    # Load datasets
    logger.info('')
    logger.info("Loading datasets:")

    if dataset == 1:
        df_train = pd.read_csv("Datasets/Train_1.csv")
        df_eval = pd.read_csv("Datasets/Eval_1.csv")
    elif dataset == 2:
        df_train = pd.read_csv("Datasets/Train_2.csv")
        df_eval = pd.read_csv("Datasets/Eval_2.csv")
    else:
        logger.critical(f'Wrong dataset type: {dataset}')
        raise Exception('Dataset/sentence type can only be 1 or 2!')

    if code_testing:  # Smaller train/dev/test set for code testing
        df_train = df_train[0:10]
        df_val = df_eval[0:10]
        df_test = df_eval[10:20]
    else:
        df_val = df_eval[0:5000]
        df_test = df_eval[5000:10000]
    logger.info(
        f'Number of sentences used for training/validation/testing: {len(df_train.index)}/{len(df_val.index)}/{len(df_test.index)}')
    pos_ratio_train = df_train['IsType'].value_counts()[1] / len(df_train.index)  # Train set class distribution
    pos_ratio_val = df_val['IsType'].value_counts()[1] / len(df_val.index)  # Val set class distribution
    pos_ratio_test = df_test['IsType'].value_counts()[1] / len(df_test.index)  # Test set class distribution
    logger.info(
        f'Type {dataset} sentence ratio in train/val/test set: {pos_ratio_train}/{pos_ratio_val}/{pos_ratio_test}')

    plot_input_len_dist = False  # True if plot distribution of training sequence (token) lengths
    if plot_input_len_dist:
        token_lens = []
        for txt in df_train["2"]:
            # The encode method can also be used
            tokens = tokenizer(
                text=txt,
                padding=False,
                truncation=True,
                max_length=512,
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                return_attention_mask=False,
                return_token_type_ids=False
            )['input_ids']
            token_lens.append(len(tokens))
        sns.distplot(token_lens, kde=False)
        plt.xlabel('Token count (train set)')
        plt.show()
    '''
    Max length chosen according to training sequence (token) length distribution.
    Max length 256 -> ~7-8/11-13gb GPU memory required for batch size = 16/32
    '''
    MAX_LEN = 256
    logger.info(f'Max length = {MAX_LEN}')

    # Grid search
    logger.info('')
    logger.info("Grid search begins: ")

    learning_rates = [1e-5]
    batch_sizes = [32]
    weight_decay_rates = [0.1]
    logger.info(f'Batch sizes: {batch_sizes}')
    logger.info(f'Initial learning rates: {learning_rates}')
    logger.info(f'Weight decay: {weight_decay_rates}')
    if not use_bfsc:
        xavier_init = False
        logger.info(f'Xavier initialization: {xavier_init}')
        drop_rates = 0.3  # Drop rates of the dropout layers
        logger.info(f"Drop rates: {drop_rates}")
        hidden_size = None  # None: no hidden layer; 512: single hidden layer of 512 neurons
        logger.info(f'Hidden layer: {hidden_size}')

    loss_fn = nn.CrossEntropyLoss().to(device)

    val_f1_grid = np.zeros((len(batch_sizes), len(learning_rates), len(weight_decay_rates)))
    epoch_grid = np.zeros((len(batch_sizes), len(learning_rates), len(weight_decay_rates)))
    best_val_f1_overall = -0.1
    t_tot = 0  # Total grid search time

    for i, bs in enumerate(batch_sizes):
        for j, lr in enumerate(learning_rates):
            for k, wd in enumerate(weight_decay_rates):

                '''
                "Gradient scaling helps prevent gradients with small magnitudes from flushing to zero 
                (“underflowing”) when training with mixed precision." 
                '''
                scaler = cutorch.amp.GradScaler(enabled=use_amp)

                '''
                Custom dataset and dataloader.
                Dataloader (an iterable): load a batch of data each iteration
                '''
                train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, bs)
                val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, bs)
                total_steps = len(train_data_loader) * EPOCHS  # len(train_data_loader) = number of (train) batches

                # Build model
                if use_bfsc:
                    model = BertForSequenceClassification.from_pretrained(
                        PRE_TRAINED_MODEL_NAME,
                        num_labels=2,
                        output_attentions=False,  # Whether the model returns attentions weights.
                        output_hidden_states=False,  # Whether the model returns all hidden-states.
                    )
                else:
                    model = SentenceClassifier(n_classes=2, drop_rates=drop_rates, xavier_init=xavier_init,
                                               hidden_size=hidden_size, use_amp=use_amp)

                if gpu_count > 1:  # Multiple GPUs
                    model = nn.DataParallel(model)
                model = model.to(device)  # Move Model to GPU(s)

                '''
                Weight decay (L2 regularization): data -= lr * weight_decay
                '''
                optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False, weight_decay=wd)
                scheduler = get_linear_schedule_with_warmup(  # Scheduler: change learning rate of optimizer
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=total_steps
                )

                logger.info('-' * 100)
                logger.info(
                    f'Batch size: {bs}, initial learning rate: {lr}, weight decay rates: {wd}, total steps: {total_steps}')
                logger.info('-' * 100)

                best_val_f1 = -0.1
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
                        len(df_train),
                        scheduler,
                        logger,
                        scaler,
                        use_bfsc,
                        use_amp
                    )

                    val_precision, val_recall, val_f1, val_acc, val_loss = eval_model(
                        model,
                        val_data_loader,
                        loss_fn,
                        device,
                        len(df_val),
                        logger,
                        use_bfsc,
                        use_amp
                    )

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_epoch = epoch + 1
                        if val_f1 > best_val_f1_overall:
                            torch.save(model, 'Model/best_model_state_' + str(dataset) + '.bin')
                            best_val_f1_overall = best_val_f1
                            best_epoch_overall = best_epoch
                            best_bs = bs
                            best_lr = lr
                            best_wd = wd

                val_f1_grid[i, j, k] = best_val_f1
                epoch_grid[i, j, k] = best_epoch
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
        f'Best parameters: batch size = {best_bs}, learning rate = {best_lr}, weight decay rate = {best_wd}, training '
        f'epoch = {best_epoch_overall}')

    # Load best model
    logger.info('')
    logger.info('Loading best model: ')
    model = torch.load('Model/best_model_state_' + str(dataset) + '.bin')
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
        logger,
        use_bfsc,
        use_amp
    )

    # Get predictions for test set
    _ = get_predictions(
        model,
        test_data_loader,
        device,
        save_preds_as_csv=True,
        use_bfsc=use_bfsc,
        use_amp=use_amp
    )

    # Save experiment record to .json file
    exp_record = {
        "Code testing": code_testing,
        "Dataset/sentence type": dataset,
        "Tokenizer parallelization": os.environ["TOKENIZERS_PARALLELISM"],
        "CPU": cpu,
        "GPU count": gpu_count,
        "Device names": device_names,
        "Random seed": RANDOM_SEED,
        "Bert model": PRE_TRAINED_MODEL_NAME,
        "Use 'BertForSequenceClassification'": use_bfsc,
        "Nb of sentences for training/validation/testing": [len(df_train.index), len(df_val.index), len(df_test.index)],
        "Total grid search time (min)": t_tot / 60,
        "Epochs": EPOCHS,
        "Max length": MAX_LEN,
        "Initial learning rates": learning_rates,
        "Batch sizes": batch_sizes,
        "Weight decay rates": weight_decay_rates,
        "Drop rate": drop_rates if not use_bfsc else None,
        "Xavier initialization": xavier_init if not use_bfsc else None,
        "Hidden layer": hidden_size if not use_bfsc else None,
        "Validation f1 values": val_f1_grid.tolist(),
        "Corresponding epoch numbers": epoch_grid.tolist(),
        "Best parameters (bs, lr, wd, epoch)": [best_bs, best_lr, best_wd, best_epoch_overall],
        "Testing performances (acc, f1, precision, recall)": [test_acc, test_f1, test_precision, test_recall]
    }

    with open("Results/exp_record_" + time.strftime("%Y%m%d_%H%M%S") + ".json", "w") as outfile:
        json.dump(exp_record, outfile)

    # Clean up: release GPU memory
    logger.info('')
    logger.info('Cleaning up: ')
    del model, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, scheduler, scaler
    gc.collect()  # Collect garbage
    if torch.cuda.is_available():
        cutorch.empty_cache()
        logger.info(f'GPU memory occupied: {cutorch.memory_reserved() / 1e9} gb')
        # dist.destroy_process_group()
    logger.info('Finished')
