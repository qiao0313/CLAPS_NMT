import json
import sys
from tqdm import tqdm
import logging
import os
import time
import numpy as np
import torch

from initialization import *
from args import get_arguments
from model import NMTModel
from optimization import set_optimizer
from hyperparams import hyperparam_path
from load_data import get_loader
from transformers import (MBart50Tokenizer, 
                          MBartForConditionalGeneration,
                          M2M100Tokenizer,
                          M2M100ForConditionalGeneration)
import sacrebleu
from eval import eval_qg


class Trainer():
    def __init__(self, args):
        
        self.args = args
        self.model = NMTModel(self.args)
        self.model_name = self.args.pretrained_model
        self.directions = {"zh_ar": ["zh", "ar"],
                           "ar_zh": ["ar", "zh"]}
        self.src_lang, self.tgt_lang = self.directions[self.args.direct][0], self.directions[self.args.direct][1]
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name,
                                                         src_lang=self.src_lang, 
                                                         tgt_lang=self.tgt_lang)
        self.device = set_device(self.args.device)

    def train(self):
        exp_path = hyperparam_path(self.args)
        logger = set_logger(exp_path, self.args.testing)
        set_seed(self.args.seed)
        logger.info("Initialization finished ...")
        logger.info("Output path is %s" % (exp_path))
        logger.info("Random seed is set to %d" % (self.args.seed))
        logger.info("Use GPU with index %s" % (self.args.device) if self.args.device >= 0 else "Use CPU as target torch device")

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        start_time = time.time()
        step_size = self.args.train_batch_size // self.args.grad_accumulate
        train_dataset = get_loader(self.args,
                                   self.args.train_data_path,
                                   self.args.direct + "_features.train",
                                   self.args.pretrained_model,
                                   step_size,
                                   shuffle=True)
        
        dev_dataset = get_loader(self.args,
                                 self.args.dev_data_path,
                                 self.args.direct + "_features.dev",
                                 self.args.pretrained_model,
                                 self.args.dev_batch_size,
                                 shuffle=False)
        
        logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
        logger.info("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset) * step_size, len(dev_dataset) * self.args.dev_batch_size))
        
        self.model.to(self.device)
        if self.args.read_model_path:
            check_point = torch.load(open(os.path.join(self.args.read_model_path, 'model.bin'), 'rb'), 
                map_location=self.device)
            self.model.bart_model.load_state_dict(check_point['model'])
            logger.info("Load saved model from path: %s" % (self.args.read_model_path))

        num_training_steps = ((len(train_dataset) * step_size + self.args.train_batch_size - 1) // self.args.train_batch_size) * self.args.num_epoch
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        logger.info('Total training steps: %d;\t Warmup steps: %d' % (num_training_steps, num_warmup_steps))
        optimizer, scheduler = set_optimizer(self.model, self.args, num_warmup_steps, num_training_steps)
    
        start_epoch, best_result = 1, {'dev_blue': 0.}
        
        if self.args.read_model_path:
            optimizer.load_state_dict(check_point['optim'])
            scheduler.load_state_dict(check_point['scheduler'])
            start_epoch = check_point['epoch'] + 1

        self.model.zero_grad()
        for epoch in range(start_epoch, self.args.num_epoch + 1):
            start_time = time.time()
            self.model.train()
            batch_nb = len(train_dataset)
            accumulating_loss, count = 0, 0
            
            for batch_idx, batch in enumerate(train_dataset, start=1):
                count += 1
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, enc_mask, decoder_input_ids, dec_mask, lm_labels = batch

                inputs = {"input_ids": input_ids,
                          "attention_mask": enc_mask,
                          "decoder_input_ids": decoder_input_ids,
                          "decoder_attention_mask": dec_mask,
                          "lm_labels": lm_labels}
                inputs["adv"] = True
                loss = self.model(**inputs)
                accumulating_loss += loss.item()
                # loss = loss / args.grad_accumulate
                loss.backward()
                
                if count == args.grad_accumulate or batch_idx == batch_nb:
                    count = 0
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    logger.info("Training: \tEpoch: %d/%d \tBatch: %d/%d \tTime: %.4f \tTraining loss: %.4f" % (epoch, self.args.num_epoch, batch_idx, batch_nb, time.time() - start_time, accumulating_loss))
                    accumulating_loss = 0

            if epoch < self.args.eval_after_epoch:
                continue
            else:
                start_time = time.time()
                dev_loss, blue_score = self.evaluate(dev_dataset)
                logger.info('Evaluation: \tEpoch: %d/%d \tTime: %.4f \tDev loss: %.4f \tDev BLUE: %.4f' % (epoch, self.args.num_epoch, time.time() - start_time, dev_loss, blue_score))

                if blue_score > best_result['dev_blue']:
                    best_result['dev_blue'], best_result['iter'] = blue_score, epoch
                    torch.save({
                            'epoch': epoch, 'model': self.model.bart_model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()
                    }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
                    logger.info('NEW BEST MODEL: \tEpoch: %d \tDev BLUE: %.4f' % (epoch, blue_score))
        
        logger.info('FINAL BEST RESULT: \tEpoch: %d \tDev BLUE: %.4f' % (best_result['iter'], best_result['dev_blue']))
    
    def evaluate(self, dev_dataset):
        self.model.eval()
        dev_loss = 0
        batch_nb = len(dev_dataset)
        pred_toks = []

        for batch_idx, batch in enumerate(dev_dataset, start=1):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, enc_mask, decoder_input_ids, dec_mask, lm_labels = batch

            inputs = {"input_ids": input_ids, 
                      "attention_mask": enc_mask,
                      "decoder_input_ids": decoder_input_ids,
                      "decoder_attention_mask": dec_mask,
                      "lm_labels": lm_labels}
            
            with torch.no_grad():
                loss = self.model(**inputs)
                dev_loss += loss

                outputs = self.model.bart_model.generate(input_ids=input_ids, 
                                                         attention_mask=enc_mask,
                                                         max_new_tokens=self.args.max_decode_step,
                                                         num_beams=self.args.beam_size, 
                                                         forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang])
                
                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    pred_ids = outputs[i].cpu()
                    pred_tok = self.tokenizer.decode(pred_ids,
                                                     skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
                    pred_toks.append(pred_tok)
        
        blue_score = self.compute_bleu_score(self.args, pred_toks)

        return dev_loss.item() / batch_nb, blue_score

    def compute_bleu_score(self, args, pred_toks):
        dev_file = os.path.join(args.dev_data_path, args.direct + ".dev")

        refs = []

        with open(dev_file) as f:
            for idx, line in enumerate(f.readlines()):
                src_txt, tgt_txt = line.strip().split('\t')
                refs.append(tgt_txt)
        
        bleu = sacrebleu.corpus_bleu(pred_toks, [refs])
        
        return bleu.score


class Predictor():
    def __init__(self, args):
        
        self.args = args
        self.model_name = self.args.pretrained_model
        self.directions = {"zh_ar": ["zh", "ar"],
                           "ar_zh": ["ar", "zh"]}
        self.src_lang, self.tgt_lang = self.directions[self.args.direct][0], self.directions[self.args.direct][1]
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name,
                                                         src_lang=self.src_lang, 
                                                         tgt_lang=self.tgt_lang)
        self.model = NMTModel(self.args)
        self.device = set_device(self.args.device)

    def predict(self):
        test_dataset = get_loader(self.args,
                                  self.args.test_data_path,
                                  self.args.direct + "_features.test",
                                  self.args.pretrained_model,
                                  self.args.test_batch_size,
                                  shuffle=False)
        
        if not self.args.read_model_path:
            print("Need trained model path !")
        else:
            check_point = torch.load(open(os.path.join(self.args.read_model_path, 'model.bin'), 'rb'), 
                map_location="cpu")
            self.model.bart_model.load_state_dict(check_point['model'])
            print("Load saved model from path: %s" % (self.args.read_model_path))

        self.model.eval()
        self.model = self.model.to(self.device)

        if not os.path.exists(self.args.res_dir):
            os.makedirs(self.args.res_dir)
        pred_file = os.path.join(self.args.res_dir, self.args.direct + ".rst")
        pred_fw = open(pred_file, "w")

        for batch in tqdm(test_dataset, total=len(test_dataset)):
            input_ids, enc_mask, _, _, _ = batch
            input_ids = input_ids.to(self.device)
            enc_mask = enc_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model.bart_model.generate(input_ids=input_ids, 
                                                         attention_mask=enc_mask,
                                                         max_new_tokens=self.args.max_decode_step,
                                                         num_beams=self.args.beam_size,
                                                         forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang])
                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    pred_ids = outputs[i].cpu()
                    pred_tok = self.tokenizer.decode(pred_ids,
                                                     skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
                    pred_fw.write(pred_tok.strip() + "\n")
                    pred_fw.flush()

        pred_fw.close()


def main(args):

    if not args.testing:
        trainer = Trainer(args)
        trainer.train()
    
    if args.testing:
        predictor = Predictor(args)
        predictor.predict()


if __name__ == '__main__':
    args = get_arguments(sys.argv[1:])
    main(args)

