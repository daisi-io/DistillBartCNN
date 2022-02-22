# from https://huggingface.co/sshleifer/distilbart-cnn-6-6

import torch

from predict import unzip_model, load_model_and_tokenizer, generate_summary

# get model_zip_path
# model_zip_path = "/pebble_source/c3334ce8-fe28-4f0f-9fc7-a7b203eec73f/distilbart-cnn-6-6.zip"
# model_path = unzip_model(model_zip_path)
model_path = "/pebble_tmp/tmp/model_distilbart"
model, tokenizer = load_model_and_tokenizer(model_path)

def compute(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt")

    summarization = generate_summary(inputs, model, tokenizer)

    return {"result": summarization}

if __name__ == "__main__":
    text = "My friends are cool but they eat too many carbs."

    print(compute(text))