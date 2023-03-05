import torch
from transformers import (MBart50Tokenizer, 
                          MBartTokenizer, 
                          MBartForConditionalGeneration,
                          M2M100Tokenizer,
                          M2M100ForConditionalGeneration)


model_name = "/data5/wenqiao_data/nmt/pretrained_model/m2m100_1.2B"

tokenizer = M2M100Tokenizer.from_pretrained(model_name)

tokenizer.src_lang = "ar"
tokenizer.tgt_lang = "zh"

# model = M2M100ForConditionalGeneration.from_pretrained(model_name)

src1 = "لا أستطيع الذهاب إلى الجنة لأن هناك شخص ما تحت الأرض ."
src2 = "أما بالنسبة للإرتباطات السلبية فليس هناك مثل هذا التقييد ."

tgt1 = "我不能上天,因为地下有人。"
tgt2 = "至于负相关,则没有这种限制。"

# all_input_ids = []
# all_labels = []
# all_input_ids.append(encoded_src1)
# all_input_ids.append(encoded_src2)
# all_labels.append(encoded_tgt1)
# all_labels.append(encoded_tgt2)

# src_encoded = tokenizer.pad({'input_ids': all_input_ids}, return_tensors='pt')
# tgt_encoded = tokenizer.pad({'input_ids': all_labels}, return_tensors='pt')

# print(src_encoded)
# print(tgt_encoded)

# print(tokenizer.encode("<s>")) # 0
# print(tokenizer.encode("</s>")) # 2
# print(tokenizer.encode("<pad>")) # 1
# print(tokenizer.encode("<mask>")) # 250053

print(tokenizer.bos_token_id) # 0
print(tokenizer.eos_token_id) # 2
print(tokenizer.pad_token_id) # 1
# print(M2M100ForConditionalGeneration.config.decoder_start_token_id)

# encode_text = tokenizer(src1, text_target=tgt1, return_tensors='pt', truncation=True, max_length=10)
# print(encode_text)

# src_encoded = tokenizer(src1, return_tensors='pt', truncation=True, max_length=128)

# generated_tokens = model.generate(
#     **src_encoded,
#     max_new_tokens=64,
#     forced_bos_token_id=tokenizer.lang_code_to_id["zh"]
# )
# print(generated_tokens)
# pred_tok = tokenizer.batch_decode(generated_tokens, 
#                                   skip_special_tokens=True)
# print(pred_tok)

# print(tokenizer._convert_id_to_token(250025)) # zh_CN
# print(tokenizer._convert_id_to_token(250001)) # ar_AR
# print(tokenizer._convert_id_to_token(250004)) # en_XX
# print(tokenizer._convert_id_to_token(250020)) # ro_RO

# batch_encoding = tokenizer.prepare_seq2seq_batch(
#         src_texts=[src1],
#         tgt_texts=[tgt1],
#         max_length=128,
#         max_target_length=128,
#         return_tensors="pt",
#     ).data
# print(batch_encoding)


# def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
#     """
#     Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
#     have a single `decoder_start_token_id` in contrast to other Bart-like models.
#     """
#     prev_output_tokens = input_ids.clone()

#     if pad_token_id is None:
#         raise ValueError("self.model.config.pad_token_id has to be defined.")
#     # replace possible -100 values in labels by `pad_token_id`
#     prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

#     index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
#     decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
#     prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
#     prev_output_tokens[:, 0] = decoder_start_tokens

#     return prev_output_tokens


# input_ids = encode_text['labels']
# decoder_input_ids = shift_tokens_right(input_ids, tokenizer.pad_token_id)
# print(decoder_input_ids)
