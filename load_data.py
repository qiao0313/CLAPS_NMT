import os
import torch
from copy import deepcopy

from torch.utils.data import DataLoader, Dataset
from transformers import (MBart50Tokenizer, 
                          MBartTokenizer, 
                          MBartForConditionalGeneration,
                          M2M100Tokenizer,
                          M2M100ForConditionalGeneration)


class NMTDataset(Dataset):
    def __init__(self, args, file_path, data_name, model_path):

        self.directions = {"zh_ar": ["zh", "ar"],
                           "ar_zh": ["ar", "zh"]}
        
        self.direct = args.direct
        self.src_lang, self.tgt_lang = self.directions[self.direct][0], self.directions[self.direct][1]
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_path, 
                                                         src_lang=self.src_lang, 
                                                         tgt_lang=self.tgt_lang)
        self.dataset = torch.load(os.path.join(file_path, data_name))
        self.total_size = len(self.dataset)

    def __getitem__(self, index):
        
        data = self.dataset[index]

        if 'encoded_tgt' in data:
            src_ids, tgt_ids = data['encoded_src'], data['encoded_tgt']
            
            start_token_id = self.tokenizer.eos_token_id
            start_token_id = torch.tensor([start_token_id], dtype=torch.long)
            
            tgt_ids = torch.cat([start_token_id, tgt_ids], dim=0)
        else:
            src_ids, tgt_ids = data['encoded_src'], None

        return src_ids, tgt_ids

    def __len__(self):
        return self.total_size


def collate_fn(data):
    def mask(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        mask = torch.sign(padded_seqs)
        return mask

    def pad(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs

    src_ids, tgt_ids = zip(*data)

    input_ids = pad(deepcopy(src_ids))
    enc_mask = mask(deepcopy(src_ids))
    
    if None not in tgt_ids:
        dec_mask = mask(deepcopy(tgt_ids))[:, :-1]
        tgt_ids = pad(tgt_ids)
        decoder_input_ids = deepcopy(tgt_ids[:, :-1])
        lm_labels = deepcopy(tgt_ids[:, 1:])
        # ignore pad id in labels
        # lm_labels = lm_labels.masked_fill(lm_labels == 1, -100)
    else:
        decoder_input_ids = None
        dec_mask = None
        lm_labels = None

    return input_ids, enc_mask, decoder_input_ids, dec_mask, lm_labels


def get_loader(args,
               file_path,
               data_name,
               model_path,
               batch_size, 
               shuffle=False,
               num_workers=0):
    f = collate_fn
    
    dataset = NMTDataset(args, file_path, data_name, model_path)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=f,
                            num_workers=num_workers)
    return dataloader

