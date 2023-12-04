from dataset import TrainDataset
from model import get_model_tokenizer
import random

model, tokenizer = get_model_tokenizer()

train, val, test = TrainDataset(tokenizer).train_test_split()

print(type(train))