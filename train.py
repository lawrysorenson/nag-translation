from dataset import TrainDataset, get_pad_to_longest
from model import get_model_tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import os

device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

print('Using device', device)

def to_device(tensors):
    return [(None if i is None else i.to(device)) for i in tensors]

model, tokenizer = get_model_tokenizer()

train, val, test = TrainDataset(tokenizer=tokenizer).train_test_split()

PAD = tokenizer.pad_token_id
padder = get_pad_to_longest(PAD)

batch_size = 32
train_loader = DataLoader(train, batch_size=batch_size, collate_fn=padder, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, collate_fn=padder)
test_loader = DataLoader(test, batch_size=1, collate_fn=padder, shuffle=True)
    
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if os.path.exists('weights.pt'):
    print('Loading previous checkpoint')
    model.load_state_dict(torch.load('weights.pt', map_location='cpu'))

model.to(device)

for epoch in range(1, 51):

    model.train()
    stop = 2000
    bar = tqdm(train_loader, total=min(len(train_loader), stop))
    i = 0
    for src, src_mask, tgt_mask, tgt_attn_mask, tgt in map(to_device, bar):

        outputs = model(
            input_ids=src,
            attention_mask=src_mask,
            decoder_input_ids=tgt_mask,
            decoder_attention_mask=tgt_attn_mask,
            labels=tgt
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        bar.set_description(f'Train Epoch {epoch} Loss: {loss.item():.4f}')

        i+=1
        if i == stop: break

    with torch.no_grad():
        model.eval()
        bar = tqdm(val_loader)
        total_loss = 0
        count = 0
        for src, src_mask, tgt_mask, tgt_attn_mask, tgt in map(to_device, bar):

            outputs = model(
                input_ids=src,
                attention_mask=src_mask,
                decoder_input_ids=tgt_mask,
                decoder_attention_mask=tgt_attn_mask,
                labels=tgt
            )

            total_loss += outputs.loss.item()
            count += src.size(0) / batch_size

            bar.set_description(f'Val Epoch {epoch} Loss: {total_loss/count:.4f}')

    torch.save(model.state_dict(), 'weights.pt')

with torch.no_grad():
    model.eval()
    sst=False
    i = 0
    for src, src_mask, tgt_mask, tgt_attn_mask, tgt in map(to_device, test_loader):
        i += 1
        if i > 20: break

        outputs = model(
            input_ids=src,
            attention_mask=src_mask,
            decoder_input_ids=tgt_mask,
            decoder_attention_mask=tgt_attn_mask,
            labels=tgt
        )

        preds = outputs.logits.argmax(dim=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=sst)

        print(tokenizer.batch_decode(src, skip_special_tokens=sst))
        print(tokenizer.batch_decode(tgt_mask, skip_special_tokens=sst))
        print(preds)
        print(tokenizer.batch_decode(tgt, skip_special_tokens=sst))
        print()