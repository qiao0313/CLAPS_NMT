import os
import json
from tqdm import tqdm
import torch
from transformers import (MBart50Tokenizer, 
                          MBartTokenizer, 
                          MBartForConditionalGeneration,
                          M2M100Tokenizer,
                          M2M100ForConditionalGeneration)


def read_text(data_file):
    res = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            src = line.strip()
            res.append(src)
    return res


def read_text_pair(data_file):
    res = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            src, tgt = line.strip().split('\t')
            res.append([src, tgt])
    return res


def extract_features(data_path, out_path, model_path, max_length):
    centric_lang = "zh"
    noncentric_lang = "ar"

    zh_ar_train_file = f"{centric_lang}_{noncentric_lang}.train"
    zh_ar_dev_file = f"{centric_lang}_{noncentric_lang}.dev"
    zh_ar_test_file = f"{centric_lang}_{noncentric_lang}.test"
    zh_ar_train_features_file = f"{centric_lang}_{noncentric_lang}_features.train"
    zh_ar_dev_features_file = f"{centric_lang}_{noncentric_lang}_features.dev"
    zh_ar_test_features_file = f"{centric_lang}_{noncentric_lang}_features.test"

    zh_ar_train_data = read_text_pair(os.path.join(data_path, zh_ar_train_file))
    zh_ar_dev_data = read_text_pair(os.path.join(data_path, zh_ar_dev_file))
    zh_ar_test_data = read_text(os.path.join(data_path, zh_ar_test_file))

    tokenizer = M2M100Tokenizer.from_pretrained(model_path, src_lang="zh", tgt_lang="ar")

    zh_ar_train_features, zh_ar_dev_features, zh_ar_test_features = [], [], []

    for idx, data in enumerate(zh_ar_train_data):
        src_text, tgt_text = data[0], data[1]
        feature = {}
        encoded_sample = tokenizer(src_text, 
                                   text_target=tgt_text, 
                                   return_tensors="pt",
                                   truncation=True,
                                   max_length=max_length)
        encoded_zh = encoded_sample['input_ids'][0]
        encoded_ar = encoded_sample['labels'][0]
        feature['encoded_src'] = encoded_zh
        feature['encoded_tgt'] = encoded_ar
        zh_ar_train_features.append(feature)
    
    torch.save(zh_ar_train_features, os.path.join(out_path, zh_ar_train_features_file))

    for idx, data in enumerate(zh_ar_dev_data):
        src_text, tgt_text = data[0], data[1]
        feature = {}
        encoded_sample = tokenizer(src_text, 
                                   text_target=tgt_text, 
                                   return_tensors="pt",
                                   truncation=True,
                                   max_length=max_length)
        encoded_zh = encoded_sample['input_ids'][0]
        encoded_ar = encoded_sample['labels'][0]
        feature['encoded_src'] = encoded_zh
        feature['encoded_tgt'] = encoded_ar
        zh_ar_dev_features.append(feature)
    
    torch.save(zh_ar_dev_features, os.path.join(out_path, zh_ar_dev_features_file))

    for idx, data in enumerate(zh_ar_test_data):
        src_text = data
        feature = {}
        encoded_sample = tokenizer(src_text, 
                                   return_tensors="pt",
                                   truncation=True,
                                   max_length=max_length)
        encoded_zh = encoded_sample['input_ids'][0]
        feature['encoded_src'] = encoded_zh
        zh_ar_test_features.append(feature)
    
    torch.save(zh_ar_test_features, os.path.join(out_path, zh_ar_test_features_file))

    ar_zh_train_file = f"{noncentric_lang}_{centric_lang}.train"
    ar_zh_dev_file = f"{noncentric_lang}_{centric_lang}.dev"
    ar_zh_test_file = f"{noncentric_lang}_{centric_lang}.test"
    ar_zh_train_features_file = f"{noncentric_lang}_{centric_lang}_features.train"
    ar_zh_dev_features_file = f"{noncentric_lang}_{centric_lang}_features.dev"
    ar_zh_test_features_file = f"{noncentric_lang}_{centric_lang}_features.test"

    ar_zh_train_data = read_text_pair(os.path.join(data_path, ar_zh_train_file))
    ar_zh_dev_data = read_text_pair(os.path.join(data_path, ar_zh_dev_file))
    ar_zh_test_data = read_text(os.path.join(data_path, ar_zh_test_file))

    tokenizer = M2M100Tokenizer.from_pretrained(model_path, src_lang="ar", tgt_lang="zh")

    ar_zh_train_features, ar_zh_dev_features, ar_zh_test_features = [], [], []

    for idx, data in enumerate(ar_zh_train_data):
        src_text, tgt_text = data[0], data[1]
        feature = {}
        encoded_sample = tokenizer(src_text, 
                                   text_target=tgt_text, 
                                   return_tensors="pt",
                                   truncation=True,
                                   max_length=max_length)
        encoded_zh = encoded_sample['input_ids'][0]
        encoded_ar = encoded_sample['labels'][0]
        feature['encoded_src'] = encoded_zh
        feature['encoded_tgt'] = encoded_ar
        ar_zh_train_features.append(feature)
    
    torch.save(ar_zh_train_features, os.path.join(out_path, ar_zh_train_features_file))

    for idx, data in enumerate(ar_zh_dev_data):
        src_text, tgt_text = data[0], data[1]
        feature = {}
        encoded_sample = tokenizer(src_text, 
                                   text_target=tgt_text, 
                                   return_tensors="pt",
                                   truncation=True,
                                   max_length=max_length)
        encoded_zh = encoded_sample['input_ids'][0]
        encoded_ar = encoded_sample['labels'][0]
        feature['encoded_src'] = encoded_zh
        feature['encoded_tgt'] = encoded_ar
        ar_zh_dev_features.append(feature)
    
    torch.save(ar_zh_dev_features, os.path.join(out_path, ar_zh_dev_features_file))

    for idx, data in enumerate(ar_zh_test_data):
        src_text = data
        feature = {}
        encoded_sample = tokenizer(src_text, 
                                   return_tensors="pt",
                                   truncation=True,
                                   max_length=max_length)
        encoded_ar = encoded_sample['input_ids'][0]
        feature['encoded_src'] = encoded_ar
        ar_zh_test_features.append(feature)
    
    torch.save(ar_zh_test_features, os.path.join(out_path, ar_zh_test_features_file))

    print("all done!")


if __name__ == '__main__':
    data_path = "./processed_datasets"
    out_path = "./processed_datasets"
    model_path = "./pretrained_model/m2m100_1.2B"
    max_length = 64
    extract_features(data_path, out_path, model_path, max_length)
