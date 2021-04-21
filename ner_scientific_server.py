#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/1/10 10:49

import logging
import re
import time
from collections import Counter

import numpy as np
import torch
from BIO_ann import bio_anno
from embed.models.transformers import BertForTokenClassification, BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from tst import text_to_word_list, split_zh_en, whole_text_connection, remove_space_by_slash_and_Hyphen, C_trans_to_E
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_list, get_lexicon, get_pos

logging.basicConfig(level=logging.INFO)


def load_and_cache_english_examples(text):
    global word_list
    word_list = bio_anno(text)  ###word_list是[(字，标签)]的格式

    print(word_list)
    # print(len(word_list))
    examples = read_examples_from_list(word_list)
    features = convert_examples_to_features(examples, labels, pos, lexicon, MAX_SEQ_LENGTH, tokenizer,
                                            cls_token_at_end=bool(MODEL_TYPE in ["xlnet"]),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if MODEL_TYPE in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(MODEL_TYPE in ["roberta"]),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(MODEL_TYPE in ["xlnet"]),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if MODEL_TYPE in ["xlnet"] else 0,
                                            pad_token_label_id=pad_token_label_id,

                                            )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lexicon_ids = torch.tensor([f.lexicon_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_pos_ids, all_label_ids, all_lexicon_ids)
    return dataset


def load_and_cache_chinese_examples(text):
    global word_list
    word_list = text_to_word_list(text)  ###word_list是[(字，标签)]的格式
    print(word_list)
    # print(len(word_list))

    examples = read_examples_from_list(word_list)
    features = convert_examples_to_features(examples, labels, pos, lexicon, MAX_SEQ_LENGTH, tokenizer,
                                            cls_token_at_end=bool(MODEL_TYPE in ["xlnet"]),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if MODEL_TYPE in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(MODEL_TYPE in ["roberta"]),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(MODEL_TYPE in ["xlnet"]),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if MODEL_TYPE in ["xlnet"] else 0,
                                            pad_token_label_id=pad_token_label_id,

                                            )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lexicon_ids = torch.tensor([f.lexicon_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_pos_ids, all_label_ids, all_lexicon_ids)
    return dataset


def predict_keywords_bio_lexicon(text):
    if is_contains_chinese(text):
        eval_dataset = load_and_cache_chinese_examples(text)
    else:
        eval_dataset = load_and_cache_english_examples(text)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[4],
                      "lexicon_ids": batch[5]}
            if MODEL_TYPE != "distilbert":
                inputs["token_type_ids"] = batch[2] if MODEL_TYPE in ["bert",
                                                                      "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)
    # print(preds)
    # print(len(preds[0]))

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    # print(out_label_list)
    # print(len(out_label_list[0]))
    # print(preds_list)
    # print(len(preds_list[0]))

    # word_list = ['目', '的', ' ', '探', '讨', '经', '皮', '内', '镜', '椎', '间', '孔', '入', '路', '微', '创', '治', '疗']
    # label_list = ['O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'B', 'I', 'I', 'I', 'I', 'B', 'I', 'O', 'O']
    global word_list
    word_list = [i[0] for i in word_list[:len(preds_list[0])]]  ##第二个元素是标签
    label_list = preds_list[0]  ##已经将pad的去掉了
    all_keys = []
    for i, token in enumerate(word_list):
        if label_list[i].startswith('B'):
            if i == len(word_list) - 1 and (split_zh_en(token)[0][0] == 1 or token in elements):
                all_keys.append([token, label_list[i].split('-')[-1].strip()])
            if i != len(word_list) - 1 and label_list[i + 1].startswith('O'):
                if split_zh_en(token)[0][0] == 1 or token in elements:
                    all_keys.append([token, label_list[i].split('-')[-1].strip()])
                else:
                    continue
            if i != len(word_list) - 1 and label_list[i + 1].startswith('I'):
                keys = [token]
                all_labels = [label_list[i].split('-')[-1].strip()]
                for z in range(i + 1, len(word_list)):
                    if label_list[z].startswith('I'):
                        keys.append(word_list[z])
                        all_labels.append(label_list[z].split('-')[-1].strip())
                    else:
                        break
                if len(list(set(all_labels))) == 1:
                    extract_key = whole_text_connection(keys)
                    if '和' == extract_key[-1] or '的' == extract_key[-1]:
                        pass
                    ## todo 直接删除缺失括号的
                    # elif extract_key.count("）")!=extract_key.count("（") and (extract_key.count("）")!=0 or extract_key.count("（")!=0):
                    #     pass
                    elif extract_key.count('）') > extract_key.count('（') and len(extract_key) > 1:
                        pass
                        new_extract_key = "（" + extract_key
                        all_keys.append([new_extract_key, all_labels[0]])
                    elif extract_key.count('）') < extract_key.count('（') and len(extract_key) > 1:
                        pass
                        new_extract_key = extract_key + '）'
                        all_keys.append([new_extract_key, all_labels[0]])
                    else:
                        all_keys.append([extract_key, all_labels[0]])

    # all_keys = list(set(all_keys))
    ## todo 如果一个实体不止有一个类别，则统计实体在文中的出现次数，如果相同则两个都保留吧，kk是一篇摘要的所有实体
    # new_all_keys=[]
    # all_keyphrase_str = [str(ele) for ele in all_keys]
    # sorted_keys = sorted(Counter(all_keyphrase_str).items(),key=lambda x:x[1], reverse=True)  ## todo 相当于去重一起做了
    # list_sorted_keys = [eval(ele) for ele in sorted_keys]
    #
    # for i, keyphrase in enumerate(list_sorted_keys):
    #     del_keys = []
    #     for z in range(i + 1, len(list_sorted_keys)):
    #         if keyphrase[0] == list_sorted_keys[z][0] and z != i:
    #             del_keys.append(list_sorted_keys[z])
    #     for ele in del_keys:
    #         list_sorted_keys.remove(ele)
    # # print(list_sorted_keys)
    # new_all_keys.extend(list_sorted_keys)

    ## todo 进行去重，对于列表的去重，使用Counter()
    all_keyphrase_str = [str(ele) for ele in all_keys]
    new_all_keys = [eval(ele) for ele in Counter(all_keyphrase_str).keys()]

    #####################todo 将具有包含关系的去掉
    del_ele = []
    for i, keyphrase in enumerate(new_all_keys):
        for z in range(len(new_all_keys)):
            if keyphrase[0] in new_all_keys[z][0] and z != i and keyphrase[1] == new_all_keys[z][
                1]:  ## todo 类别相同的具有包含关系的才删除
                del_ele.append(keyphrase)
    del_ = []
    for zk in del_ele:
        if zk not in del_:  ## todo 不能直接用set，因为元素是列表
            del_.append(zk)
    if len(del_) != 0:
        for ele in del_:
            new_all_keys.remove(ele)

    ## todo 去掉以特殊字符结尾的
    pattern_end = re.compile(r'(.*)[_:∶！。，（；、？——+=]$')
    new_article_keys = []
    for i in new_all_keys:
        ## todo 如果只有（而且还不在末尾一般是错的，
        if "（" in i[0] and "）" not in i[0] and "（" != i[0][-1]:
            pass
        elif i[0].count("（") > i[0].count("）") and "（" == i[0][-1]:  ## todo 存在两个括号以上的情况，所以用count
            i = [i[0][:-1], i[1]]
            new_article_keys.append(i)
        elif pattern_end.match(i[0]) != None:
            i = [i[0][:-1], i[1]]
            new_article_keys.append(i)

        else:
            new_article_keys.append(i)

    return new_article_keys


def entity_type_combination(article_keys):
    question = []
    method = []
    metric = []
    dataset = []
    scientist = []
    theory = []
    equipment = []
    software = []
    all = []
    for ele in article_keys:
        if ele[1] == '研究问题':
            question.append(ele[0])
        if ele[1] == '研究方法及模型算法':
            method.append(ele[0])
        if ele[1] == '效应指标':
            metric.append(ele[0])
        if ele[1] == '数据集':
            dataset.append(ele[0])
        if ele[1] == '科学家':
            scientist.append(ele[0])
        if ele[1] == '理论原理':
            theory.append(ele[0])
        if ele[1] == '仪器设备':
            equipment.append(ele[0])
        if ele[1] == '软件系统':
            software.append(ele[0])
    all.append([question, method, metric, dataset, scientist, theory, equipment, software])
    return all


# 检验是否含有中文字符
def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


# 检验是否全是中文字符
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


if __name__ == '__main__':

    # Some Parameters
    output_dir = r'output_multilingual'  # Raw string
    MAX_SEQ_LENGTH = 512
    MODEL_TYPE = 'bert'
    EVAL_BATCH_SIZE = 16
    DEVICE = torch.device("cuda")

    # Pre-Load model
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    model = BertForTokenClassification.from_pretrained(output_dir)
    model.to('cuda')

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    labels = get_labels(path='labels.txt')
    pos = get_pos(None)
    lexicon = get_lexicon(path='lexicon_map.txt')

    # 读取化学元素
    with open(r'化学元素.txt', 'r', encoding='utf-8') as f:
        elements = f.readlines()
    elements = [ele.strip() for ele in elements]

    while True:
        # text='''盆腔操联合心理干预对盆腔炎性不孕症患者负性情绪的影响。目的改善盆腔炎性不孕症患者焦虑、抑郁的负性情绪.方法将168例盆腔炎性不孕症患者随机分为心理干预组55例、盆腔操组56例、联合组57例.分别给予心理干预、盆腔操练习及在盆腔操练习的同时实施心理干预的护理措施.采用抑郁自评量表、焦虑自评量表分别于入院时、干预1用后、干预2周后进行测评.结果盆腔操组及联合组不同时间焦虑、抑郁评分比较,差异有统计学意义(均P<0.05);干预后三组焦虑、抑郁评分比较,差异有统计学意义(均P<0.05).结论盆腔操及盆腔操结合心理干预的联合措施能减轻盆腔炎性不孕症患者焦虑和抑郁,联合组更有利于患者心理健康.'''
        text = input()
        processed_text = C_trans_to_E(remove_space_by_slash_and_Hyphen(text))
        start = time.time()
        all_keys = predict_keywords_bio_lexicon(text)
        all_entities = entity_type_combination(all_keys)[0]
        new_all_entities = [[], [], [], [], [], [], [], []]
        for i, kk in enumerate(all_entities):
            for entity_type in enumerate(kk):
                new_entity = C_trans_to_E(entity_type[1].lower())
                try:
                    start_loc = processed_text.lower().index(new_entity)
                    end_loc = start_loc + len(entity_type[1])
                    new_all_entities[i].append(processed_text[start_loc:end_loc])
                except:
                    new_all_entities[i].append(new_entity)
                    print('报错啦!!!!!!{}'.format(entity_type))

        for i, ele in enumerate(new_all_entities):
            if i == 0 and len(ele) != 0:
                print('研究问题：{}'.format('   '.join(ele)))
            if i == 1 and len(ele) != 0:
                print('研究方法及模型算法：{}'.format('   '.join(ele)))
            if i == 2 and len(ele) != 0:
                print('度量指标：{}'.format('   '.join(ele)))
            if i == 3 and len(ele) != 0:
                print('数据集：{}'.format('   '.join(ele)))
            if i == 4 and len(ele) != 0:
                print('科学家：{}'.format('   '.join(ele)))
            if i == 5 and len(ele) != 0:
                print('理论原理：{}'.format('   '.join(ele)))
            if i == 6 and len(ele) != 0:
                print('仪器设备：{}'.format('   '.join(ele)))
            if i == 7 and len(ele) != 0:
                print('软件系统：{}'.format('   '.join(ele)))
        end = time.time()
        print('用时:{}秒'.format(end - start))
