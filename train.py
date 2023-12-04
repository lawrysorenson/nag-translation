from dataset import TrainDataset, get_pad_to_longest
from model import get_model_tokenizer
from torch.utils.data import Dataset, DataLoader
import torch

model, tokenizer = get_model_tokenizer()

train, val, test = TrainDataset(tokenizer=tokenizer).train_test_split()

PAD = tokenizer.pad_token_id
padder = get_pad_to_longest(PAD)

batch_size = 2
test_loader = DataLoader(test, batch_size=batch_size, collate_fn=padder)
    
for epoch in range(1, 5):

    # model.train()
    # for part in train:
    #     print(part)
    #     break


    break
        

with torch.no_grad():
    model.eval()
    for src, src_mask, tgt_mask, tgt_attn_mask, tgt in test_loader:
        print(tgt_mask)

        outputs = model(
            input_ids=src,
            attention_mask=src_mask,
            decoder_input_ids=tgt_mask,
            decoder_attention_mask=tgt_attn_mask,
            labels=tgt
        )

        preds = outputs.logits.argmax(dim=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        print(tokenizer.batch_decode(src, skip_special_tokens=True))
        print(tokenizer.batch_decode(tgt_mask))
        print(preds)
        print(tokenizer.batch_decode(tgt, skip_special_tokens=True))
        break