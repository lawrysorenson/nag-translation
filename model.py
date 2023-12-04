from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

seed = 'Helsinki-NLP/opus-mt-es-en'

def get_model_tokenizer():
    model = AutoModelForSeq2SeqLM.from_pretrained(seed)
    tokenizer = AutoTokenizer.from_pretrained(seed)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<mask>']})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
