# Uncomment the following line to run as Jupyter notebook
# !pip install -qq transformers

import os
import time
from collections import defaultdict  # Provides a default val for the key that does not exists->never raises a KeyError.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # csv format data
import seaborn as sns  # Statistical plot
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader  # Create custom PyTorch dataset and dataloader
from transformers import BertModel, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup


# Create custom PyTorch dataset
class CustomDataset(Dataset):

    def __init__(self, sentences, is_type_one, is_type_two, tokenizer, max_len):
        self.sentences = sentences
        self.is_type_one = is_type_one
        self.is_type_two = is_type_two
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.is_type_one)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        is_type_one = self.is_type_one[item]
        is_type_two = self.is_type_two[item]

        encoding = self.tokenizer(
            text=sentence,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'is_type_one': torch.tensor(is_type_one, dtype=torch.long),
            'is_type_two': torch.tensor(is_type_two, dtype=torch.long)
        }


# Create custom PyTorch dataloader (Combines a dataset and a sampler, and provides an iterable over the given dataset.)
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CustomDataset(
        sentences=df["2"].to_numpy(),
        is_type_one=df["IsTypeOne"].to_numpy(),
        is_type_two=df["IsTypeTwo"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4
    )


# We use a dropout layer for some regularization and a fully-connected layer for our output
class SentenceClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        model_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = model_output[1]
        return self.out(self.drop(pooled_output))


# One training epoch
# def train_epoch(Model,
#         data_loader,
#         loss_fn,
#         optimizer,
#         device,
#         scheduler,
#         n_examples
#                 ):
#     Model = Model.train()  # Put Model in train mode (different behavior for the dropout layer)
#
#     losses = []
#     correct_predictions = 0
#
#     for d in data_loader:  # For one batch of data
#         input_ids = d["input_ids"].to(device)
#         attention_mask = d["attention_mask"].to(device)
#         targets = d["is_type_one"].to(device)
#
#         # Forward prop
#         outputs = Model(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         _, preds = torch.max(outputs, dim=1)
#         loss = loss_fn(outputs, targets)
#         correct_predictions += torch.sum(preds == targets)
#         losses.append(loss.item())  # The item() method extracts the loss's value as a Python float
#
#         # Back prop
#         loss.backward()  # Computes dloss/dx for every parameter x which has requires_grad=True. Then x.grad += dloss/dx
#         nn.utils.clip_grad_norm_(Model.parameters(), max_norm=1.0)
#         optimizer.step()  # Updates the value of x using the gradient x.grad. x += -lr * x.grad
#         scheduler.step()  # Update the learning rate
#         optimizer.zero_grad()  # Clears x.grad for every parameter x
#
#     return correct_predictions.double() / n_examples, np.mean(losses)


# One training epoch
def train_epoch(model,
                data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                n_examples
                ):
    model = model.train()  # Put Model in train mode (different behavior for the dropout layer)

    # For one epoch
    tot_loss = 0
    tn_tot, fp_tot, fn_tot, tp_tot = 0, 0, 0, 0  # Confusion matrix

    for d in data_loader:  # For one batch of data
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["is_type_one"].to(device)

        # Forward prop
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)  # preds.get_device()?
        loss = loss_fn(outputs, targets)
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


# def eval_model(Model, data_loader, loss_fn, device, n_examples):
#     Model = Model.eval()  # Put Model in eval mode (different behavior for the dropout layer)
#
#     losses = []
#     correct_predictions = 0
#
#     with torch.no_grad():  # Temporarily set all the requires_grad flag to false
#         for d in data_loader:  # For one batch of data
#             input_ids = d["input_ids"].to(device)
#             attention_mask = d["attention_mask"].to(device)
#             targets = d["is_type_one"].to(device)
#
#             outputs = Model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             )
#             _, preds = torch.max(outputs, dim=1)
#
#             loss = loss_fn(outputs, targets)
#
#             correct_predictions += torch.sum(preds == targets)
#             losses.append(loss.item())
#
#     return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()  # Put Model in eval mode (different behavior for the dropout layer)

    tot_loss = 0
    tn_tot, fp_tot, fn_tot, tp_tot = 0, 0, 0, 0  # Confusion matrix

    with torch.no_grad():  # Temporarily set all the requires_grad flag to false
        for d in data_loader:  # For one batch of data
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["is_type_one"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            tot_loss += loss_fn(outputs, targets)

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


def get_predictions(model, data_loader, device, save_preds_as_csv=True):
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
            targets = d["is_type_one"].to(device)

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
    return df_result


if __name__ == '__main__':
    # Initialization
    print("Initialization:")

    code_testing = False  # Whether we are testing the code
    print(f'Code being tested: {code_testing}')

    first_time_run = False  # BERT needs to be downloaded if the code is run for the first time

    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Avoid warning messages caused by multiprocessing (dataloader)
    print(f'Tokenizer parallelization: {os.environ["TOKENIZERS_PARALLELISM"]}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL_NAME)

    model = SentenceClassifier(n_classes=2)
    model = model.to(device)  # Move Model to GPU

    if first_time_run:  # First time run
        bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        bert_model.save_pretrained('./Model/')
    else:  # From the second time
        bert_model = BertModel.from_pretrained('./Model/')

    EPOCHS = 10
    print(f'Number of epochs: {EPOCHS}')

    BATCH_SIZE = 16
    print(f"Batch size: {BATCH_SIZE}")

    MAX_LEN = 256  # Max length chosen according to training sequence (token) length distribution
    print(f'Max length = {MAX_LEN}')

    # Example with sample text
    # print("\nExample of tokenization with sample text:")
    #
    # sample_txt = "为什么没有starter code？"
    #
    # encoding = tokenizer(  # __call__ method
    #     text=sample_txt,
    #     padding="max_length",
    #     truncation=True,
    #     max_length=16,  # Default to be 512 for the BERT Model
    #     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    #     return_tensors='pt',  # Return PyTorch tensors
    #     return_token_type_ids=False,
    #     return_attention_mask=True,
    # )
    #
    # print(f'Sentence with special tokens added: {tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])}')
    # print(f'Token IDs: {encoding["input_ids"][0]}')
    # print(f'Attention mask: {encoding["attention_mask"][0]}')
    #
    # output = bert_model.forward(  # output format: tuple
    #     input_ids=encoding['input_ids'],
    #     attention_mask=encoding['attention_mask']
    # )
    #
    # last_hidden_state = output[0]
    # '''
    # pooler_output: Last layer hidden-state of the first token of the sequence (classification token) further processed
    # by a Linear layer and a Tanh activation function. It acts as a summary of the content, according to BERT.
    # '''
    # pooler_output = output[1]
    #
    # print(f"Shape of last hidden state: {last_hidden_state.shape}")  # batch_size * sequence_length * hidden_size
    # print(f"Shape of pooler output: {pooler_output.shape}")  # batch_size * hidden_size

    # Load datasets
    print("\nLoad datasets:")

    '''
    Train set
    '''
    df_train = pd.read_csv("Datasets/Train.csv")
    Type_one_sentence_ratio_train = df_train['IsTypeOne'].value_counts()[1] / len(df_train.index)  # Train set balanced?
    print(f'Type one sentence ratio for train set: {Type_one_sentence_ratio_train}')

    token_lens = []  # Distribution of training sequence (token) lengths
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

    if code_testing:  # Smaller train set for code testing
        df_train = df_train[0:10]
    print(f'Number of sentences used for training: {len(df_train.index)}')

    '''
    Dev/test set
    '''
    df_test = pd.read_csv("Datasets/Test.csv")
    Type_one_sentence_ratio_test = df_test['IsTypeOne'].value_counts()[1] / len(df_test.index)  # Train set balanced?
    print(f'Type one sentence ratio for dev and test sets: {Type_one_sentence_ratio_test}')

    token_lens = []  # Distribution of dev/test sequence (token) lengths
    for txt in df_test[0:10000]["2"]:
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
    plt.xlabel('Token count (dev/test set)')
    plt.show()

    if code_testing:  # Smaller dev and test sets for code testing
        df_test = df_test[0:10]
    else:
        df_test = df_test[0:10000]
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    print(f'Number of sentences used for validation: {len(df_val.index)}')
    print(f'Number of sentences used for testing: {len(df_test.index)}')

    # Custom dataset and dataloader
    '''
    Dataloader (an iterable): load a batch of data each iteration
    '''
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # Example of one batch of validation data
    # print('\nExample of one batch of validation data:')
    #
    # data = next(iter(val_data_loader))
    # input_ids = data['input_ids'].to(device)
    #
    # attention_mask = data['attention_mask'].to(device)
    # print(f"Shape of input_ids: {data['input_ids'].shape}")
    # print(f"Shape of attention_mask: {data['attention_mask'].shape}")
    # print(f"Shape of is_type_one/two: {data['is_type_one'].shape}")
    #
    # output_proba = F.softmax(model(input_ids, attention_mask), dim=1)  # Apply softmax to the outputs
    # print(f"Shape of output probability tensors (n_sentence=batch_size*1 batch, n_classes=2): {output_proba.shape}")

    # Training
    print("\nTraining: ")

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS  # len(train_data_loader) = number of (train) batches
    print(f"Total steps: {total_steps}")

    scheduler = get_linear_schedule_with_warmup(  # Scheduler: change learning rate of optimizer
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)  # Default value: empty list
    best_f1 = 0
    t_tot = 0

    val_precision, val_recall, val_f1, val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )

    print(f'Initial val loss {val_loss} accuracy {val_acc} f1 {val_f1} precision {val_precision} recall {val_recall}')

    for epoch in range(EPOCHS):  # Training loop

        t0 = time.time()
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 100)

        train_precision, train_recall, train_f1, train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc} f1 {train_f1} precision {train_precision} recall '
              f'{train_recall}')

        val_precision, val_recall, val_f1, val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )

        print(f'Val loss {val_loss} accuracy {val_acc} f1 {val_f1} precision {val_precision} recall {val_recall}')

        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_loss'].append(val_loss)

        if val_f1 > best_f1:
            torch.save(model.state_dict(), 'Model/best_model_state.bin')
            best_f1 = val_f1

        t_epoch = time.time() - t0
        t_tot += t_epoch
        print(f'Epoch time: {t_epoch / 60} min')

    print(f'\nTotal training time: {t_tot / 60} min')
    print(f'Best validation f1: {best_f1}')

    # Plot accuracies
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title('Training history (accuracies)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()
    plt.show()

    # Plot f1 curves
    plt.plot(history['train_f1'], label='train f1')
    plt.plot(history['val_f1'], label='validation f1')

    plt.title('Training history (f1)')
    plt.ylabel('f1')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()
    plt.show()

    # Test Model performance on testing set
    print('\nTesting Model performance: ')
    test_precision, test_recall, test_f1, test_acc, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(df_test)
    )

    print(f'Test accuracy {test_acc} f1 {test_f1} precision {test_precision} recall {test_recall}')

    # Get predictions for test set
    df_test_result = get_predictions(
        model,
        test_data_loader,
        device,
        save_preds_as_csv=True
    )
