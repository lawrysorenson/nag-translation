import torch
from model import get_model_tokenizer
from dataset import prep_data
import os

device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

print('Loading model and tokenizer')
model, tokenizer = get_model_tokenizer()

if os.path.exists('weights.pt'):
    print('Loading previous checkpoint')
    model.load_state_dict(torch.load('../weights-chk3.pt', map_location='cpu'))
    
print('Finished loading model')

model.eval()

def infer(srcs, tgts):
    inputs = prep_data(tokenizer, srcs, tgts)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    return preds