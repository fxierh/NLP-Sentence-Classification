"""
Adapted from: Liu Huan's script
"""

import re
import time

import numpy as np
import pandas as pd
import torch
import torch.cuda as cutorch
import torch.nn.functional as F
from flask import Flask, render_template, request
from transformers import BertTokenizerFast

from main import SentenceClassifier

app = Flask(__name__, template_folder='Templates')

MAX_LEN = 256
USE_AMP = True if torch.cuda.is_available() else False

pd.set_option("display.max_rows", None, "display.max_columns", None, "max_colwidth", 30)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()
device_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if torch.cuda.is_available() else 'cpu'

model_1 = SentenceClassifier(use_amp=USE_AMP)  # 目的语步模型
model_1.load_state_dict(torch.load('./Model/best_model_state_1.pt', map_location=device))
model_1 = model_1.to(device)

model_2 = SentenceClassifier(use_amp=USE_AMP)  # 方法语步模型
model_2.load_state_dict(torch.load('./Model/best_model_state_2.pt', map_location=device))
model_2 = model_2.to(device)

tokenizer = BertTokenizerFast.from_pretrained("./Model/")


def untag(sent):
    # TODO:
    #   Write docstrings.

    delete_first7_chars = (' 背景与目的:', ' 背景与目的：')
    delete_first6_chars = ('背景与目的:', '背景与目的：', ' 背景与目的')
    delete_first5_chars = ('背景与目的', '资料与方法', '材料与方法', '材料和方法',
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
    if sent.startswith(delete_first7_chars):
        sent = sent[7:]
    if sent.startswith(delete_first6_chars):
        sent = sent[6:]
    if sent.startswith(delete_first5_chars):
        sent = sent[5:]
    if sent.startswith(delete_first4_chars):
        sent = sent[4:]
    if sent.startswith(delete_first3_chars):
        sent = sent[3:]
    if sent.startswith(delete_first2_chars) and not sent.startswith('结果表明'):
        sent = sent[2:]
    return sent


def paragraph_segmentation_untag(paragraph):
    """
    Segments a paragraph in Chinese into sentences with tags removed.
    :param paragraph: paragraph in Chinese.
    :return: a list of strings (sentences).
    """

    def paragraph_segmentation(para):
        """
        An iterator that yields one sentence a time in order from a Chinese paragraph.
        :param para: paragraph in Chinese, "。" is used as sentence separator.
        :return:
        """

        for sent in re.findall(u'[^！？。]+[！？。]?', para, flags=re.U):
            yield sent

    paragraph = paragraph.strip()  # Remove spaces and "\n" operators at the beginning and at the end of the string
    sentences = []
    if '。' in paragraph:
        for sent in list(paragraph_segmentation(paragraph)):
            sent = untag(sent)
            sentences.append(sent)
    elif '. ' in paragraph:
        sentence_start_indices = [0] + [m.end(0) for m in
                                        re.finditer(r"[^a-zA-Z！？!?.][！？!?.][ ]", paragraph, flags=re.U)]
        for i in range(len(sentence_start_indices)):
            if i + 1 < len(sentence_start_indices):
                sentences.append(paragraph[sentence_start_indices[i]:sentence_start_indices[i + 1]].strip())
            else:
                sentences.append(paragraph[sentence_start_indices[i]:].strip())
    else:
        raise Exception('The abstract cannot be segmented into sentences!')

    return sentences


def get_predictions(mdl, device, sentences_list, enc, use_amp=True):
    # TODO:
    #   Write docstrings.

    mdl = mdl.eval()  # Put Model in eval mode (different behavior for the dropout layer)

    predictions = []
    prediction_probs = []

    with torch.no_grad():  # Temporarily set all the requires_grad flag to false
        for i, sent in enumerate(sentences_list):
            input_ids = enc[i]["input_ids"].to(device)
            attention_mask = enc[i]["attention_mask"].to(device)

            with cutorch.amp.autocast(enabled=use_amp):
                outputs = mdl(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)

    predictions = torch.stack(predictions).cpu().numpy()
    prediction_probs = torch.stack(prediction_probs).cpu().numpy()
    prediction_probs = prediction_probs[np.arange(prediction_probs.shape[0]), predictions]

    # Build dataframe containing test results
    result = np.vstack((sentences_list, predictions, prediction_probs)).transpose()
    df_result = pd.DataFrame(result, columns=['Sentences', 'Predictions', 'Probabilities'])
    return df_result


@app.route('/', methods=['POST', 'GET'])
def enter_abstract():
    return render_template('form.html', form=enter_abstract)


@app.route('/results', methods=['POST', 'GET'])
def sentence_classification():
    if request.method == 'GET':
        return f"The URL /classification is accessed directly. Try going back to '/' to submit form first."
    if request.method == 'POST':
        abstract = request.form["Abstract"]

        t0 = time.time()
        segmented_abstract = paragraph_segmentation_untag(paragraph=abstract)
        encodings = []
        for sentence in segmented_abstract:
            encodings.append(tokenizer(text=sentence, add_special_tokens=True, padding='max_length', truncation=True,
                                       max_length=MAX_LEN, return_token_type_ids=False, return_attention_mask=True,
                                       return_tensors='pt'))

        df_1 = get_predictions(model_1, device, segmented_abstract, encodings, use_amp=USE_AMP)
        df_2 = get_predictions(model_2, device, segmented_abstract, encodings, use_amp=USE_AMP)
        df = pd.concat([df_1.rename(columns={"Predictions": "研究问题句", "Probabilities": "概率", "Sentences": "句子"}),
                        df_2.drop(['Sentences'], axis=1).rename(columns={"Predictions": "研究方法句",
                                                                         "Probabilities": "概率"})], axis=1)
        return render_template('data.html',
                               dataframe=df.to_html(col_space=['25cm', '3cm', '2cm', '3cm', '2cm'], justify='center'),
                               time_consumed=time.time() - t0, device=device_names)


if __name__ == '__main__':
    app.run(host='localhost', debug=True)
