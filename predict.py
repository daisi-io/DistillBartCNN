import os
import pathlib
import zipfile
from transformers import BartTokenizer, BartForConditionalGeneration


def unzip_model(zip_model_path):
    p = pathlib.Path(zip_model_path)
    root = str(p.parents[0])
    print(f'start unzip: {root}')
    model_path = os.path.join(root, os.path.splitext(os.path.basename(zip_model_path))[0] + '/')
    if not os.path.isdir(model_path):
        with zipfile.ZipFile(zip_model_path, "r") as zip_ref:
            zip_ref.extractall(root)
        
    print(f"finish zipping model to {model_path}")
    return model_path

def load_model_and_tokenizer(model_path):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def generate_summary(inputs, model, tokenizer):
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=5)
    summarization = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return summarization