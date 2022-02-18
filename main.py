# from https://huggingface.co/sshleifer/distilbart-cnn-6-6

from predict import unzip_model, load_model_and_tokenizer, generate_summary

# get model_zip_path
model_zip_path = "/Users/zhenshanjin/Documents/Belmont/sandy/UtilityDaisies/DistillBartCNN_6_6/tmp/distilbart-cnn-6-6.zip"
model_path = unzip_model(model_zip_path)
model, tokenizer = load_model_and_tokenizer(model_path)

def compute(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt")

    summarization = generate_summary(inputs, model, tokenizer)

    return {"result": summarization}

if __name__ == "__main__":
    text = "My friends are cool but they eat too many carbs."

    print(compute(text))