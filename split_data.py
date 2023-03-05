import os
import numpy as np
from tqdm import tqdm


def write_file(res, file):
    with open(file, 'w', encoding='utf-8') as f:
        res = [data[0] + "\t" + data[1] + "\n" for data in res]
        f.writelines(res)
    print(f'write to {file} success, total {len(res)} lines.')


def read_text_pair(data_file):
    res = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            src, tgt = line.strip().split('\t')
            res.append([src, tgt])
    return res


def shuffle(res, dev_len, seed=1):
    ''' dev_len若[0,1]则按比例取，若>1则按条数取 '''
    assert dev_len > 0 , "dev set ratio should lager than 0."
    # shuffle
    shuffled_res = []
    shuffle_idx = np.random.RandomState(seed=seed).permutation(np.arange(0, len(res))).tolist()
    for idx in tqdm(shuffle_idx):
        shuffled_res.append(res[idx])
    # dev len
    dev_len = int(dev_len) if dev_len >= 1 else int(len(res) * dev_len)
    # split
    train_set = shuffled_res[:-dev_len]
    dev_set = shuffled_res[-dev_len:]
    
    return train_set, dev_set


def split_data(data_path, out_path, dev_len):
    centric_lang = "zh"
    noncentric_lang = "ar"
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    zh_ar_file = f"{centric_lang}_{noncentric_lang}.train"
    ar_zh_file = f"{noncentric_lang}_{centric_lang}.train"

    zh_ar_res = read_text_pair(os.path.join(data_path, zh_ar_file))
    zh_ar_train_data, zh_ar_dev_data = shuffle(zh_ar_res, dev_len=dev_len)
    zh_ar_train_file = f"{centric_lang}_{noncentric_lang}.train"
    zh_ar_dev_file = f"{centric_lang}_{noncentric_lang}.dev"
    write_file(zh_ar_train_data, os.path.join(out_path, zh_ar_train_file))
    write_file(zh_ar_dev_data, os.path.join(out_path, zh_ar_dev_file))

    ar_zh_res = read_text_pair(os.path.join(data_path, ar_zh_file))
    ar_zh_train_data, ar_zh_dev_data = shuffle(ar_zh_res, dev_len=dev_len)
    ar_zh_train_file = f"{noncentric_lang}_{centric_lang}.train"
    ar_zh_dev_file = f"{noncentric_lang}_{centric_lang}.dev"
    write_file(ar_zh_train_data, os.path.join(out_path, ar_zh_train_file))
    write_file(ar_zh_dev_data, os.path.join(out_path, ar_zh_dev_file))


if __name__ == '__main__':
    data_path = "./datasets"
    out_path = "./processed_datasets"
    dev_len = 1000
    split_data(data_path, out_path, dev_len)

