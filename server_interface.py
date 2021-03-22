"""
TODO:
    Write docstrings (see main.py).
"""

import re

import numpy as np
import pandas as pd
import torch
import torch.cuda as cutorch
import torch.nn.functional as F
from transformers import BertTokenizerFast

from main import SentenceClassifier


def untag(sentence):
    # TODO:
    #   Write docstrings.

    delete_first7_chars = (' 背景与目的:', ' 背景与目的：')
    delete_first6_chars = ('背景与目的:', '背景与目的：', ' 背景与目的')
    delete_first5_chars = ('背景与目的',
                           ' 【讨论】', ' 【简介】', ' 【背景】', ' 【目的】', ' 【方法】', ' 【结果】', ' 【结论】',
                           ' [讨论]', ' [简介]', ' [背景]', ' [目的]', ' [方法]', ' [结果]', ' [结论]')
    delete_first4_chars = ('【讨论】', '【简介】', '【背景】', '【目的】', '【方法】', '【结果】', '【结论】',
                           '[讨论]', '[简介]', '[背景]', '[目的]', '[方法]', '[结果]', '[结论]',
                           ' 讨论：', ' 简介：', ' 背景：', ' 目的：', ' 方法：', ' 结果：', ' 结论：',
                           ' 讨论:', ' 简介:', ' 背景:', ' 目的:', ' 方法:', ' 结果:', ' 结论:',
                           ' 讨论·', ' 简介·', ' 背景·', ' 目的·', ' 方法·', ' 结果·', ' 结论·')
    delete_first3_chars = ('讨论：', '简介：', '背景：', '目的：', '方法：', '结果：', '结论：',
                           '讨论:', '简介:', '背景:', '目的:', '方法:', '结果:', '结论:',
                           '讨论·', '简介·', '背景·', '目的·', '方法·', '结果·', '结论·',
                           ' 讨论', ' 简介', ' 背景', ' 目的', ' 方法', ' 结果', ' 结论', '目的 ', '方法 ', '结果 ', '结论 ')
    delete_first2_chars = ('简介', '背景', '目的', '方法', '结果', '结论')
    if sentence.startswith(delete_first7_chars):
        sentence = sentence[7:]
    if sentence.startswith(delete_first6_chars):
        sentence = sentence[6:]
    if sentence.startswith(delete_first5_chars):
        sentence = sentence[5:]
    if sentence.startswith(delete_first4_chars):
        sentence = sentence[4:]
    if sentence.startswith(delete_first3_chars):
        sentence = sentence[3:]
    if sentence.startswith(delete_first2_chars) and not sentence.startswith('结果表明'):
        sentence = sentence[2:]
    return sentence


def paragraph_segmentation_untag(paragraph):
    """
    Segments a paragraph in Chinese into sentences with tags removed.
    :param paragraph: paragraph in Chinese.
    :return: a list of strings (sentences).
    """

    def paragraph_segmentation(para):
        """
        An iterator that yields one sentence a time in order from a Chinese paragraph.
        :param para: paragraph in Chinese.
        :return:
        """

        for sent in re.findall(u'[^！？。]+[！？。]?', para, flags=re.U):
            yield sent

    sentences = []
    for sentence in list(paragraph_segmentation(paragraph)):
        sentence = untag(sentence)
        sentences.append(sentence)
    return sentences


def get_prediction(model, device, sentences_list, encodings, use_amp=True):
    # TODO:
    #   Write docstrings.

    model = model.eval()  # Put Model in eval mode (different behavior for the dropout layer)

    # sentences = []
    predictions = []
    prediction_probs = []
    # real_values = []

    input_ids = []

    with torch.no_grad():  # Temporarily set all the requires_grad flag to false
        for i, sent in enumerate(sentences_list):
            # texts = d["sentence"]
            input_ids = encodings[i]["input_ids"].to(device)
            attention_mask = encodings[i]["attention_mask"].to(device)
            # targets = d["is_type"].to(device)

            with cutorch.amp.autocast(enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)

            # sentences.extend(sent)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            # real_values.extend(targets)

    # sentences = np.array(sentences)
    predictions = torch.stack(predictions).cpu().numpy()
    prediction_probs = torch.stack(prediction_probs).cpu().numpy()
    prediction_probs = prediction_probs[np.arange(prediction_probs.shape[0]), predictions]
    # real_values = torch.stack(real_values).cpu().numpy()

    # Build dataframe containing test results
    result = np.vstack((sentences_list, predictions, prediction_probs)).transpose()
    df_result = pd.DataFrame(result, columns=['Sentences', 'Predictions', 'Probabilities'])
    return df_result


if __name__ == '__main__':
    MAX_LEN = 256
    USE_AMP = True if torch.cuda.is_available() else False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SentenceClassifier(use_amp=USE_AMP)
    model.load_state_dict(torch.load('Model/best_model_state_1.pt', map_location=device))  # 目的语步模型
    model = model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained("./Model/")

    while True:
        abstract = input()
        segmented_abstract = paragraph_segmentation_untag(paragraph=abstract)

        encodings = []
        for sentence in segmented_abstract:
            encodings.append(tokenizer(text=sentence, add_special_tokens=True, padding='max_length', truncation=True,
                                       max_length=MAX_LEN, return_token_type_ids=False, return_attention_mask=True,
                                       return_tensors='pt'))
        df = get_prediction(model, device, segmented_abstract, encodings, use_amp=USE_AMP)
        print(df)
