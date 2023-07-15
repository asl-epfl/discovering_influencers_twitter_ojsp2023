from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import torch
device = torch.device("cpu") ##cuda gave some errors

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.to(device)
#model.save_pretrained(MODEL)
"""
text = "Covid cases are increasing fast!"
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input.to(device)
output = model(**encoded_input)
scores = output[0][0].detach().cpu().numpy()
scores = softmax(scores)
"""


"""
# Print labels and scores
ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = config.id2label[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")
"""

#print(scores[0])

def eval_sentences(sentence_list):
    score_list = []
    if type(sentence_list)==str:
        text = sentence_list
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        encoded_input.to(device)
        output = model(**encoded_input)
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)
        negative_score = scores[0]
        positive_score = scores[2]
        new_negative = negative_score/(negative_score+positive_score)
        new_positive = positive_score/(negative_score+positive_score)
        score_list.append((new_positive,new_negative))
    else:
        for text in sentence_list:
            text = preprocess(text)
            encoded_input = tokenizer(text, return_tensors='pt')
            encoded_input.to(device)
            output = model(**encoded_input)
            scores = output[0][0].detach().cpu().numpy()
            scores = softmax(scores)
            negative_score = scores[0]
            positive_score = scores[2]
            new_negative = negative_score/(negative_score+positive_score)
            new_positive = positive_score/(negative_score+positive_score)
            score_list.append((new_positive,new_negative))
    return score_list

def eval_sentences_positive(sentence_list):
    score_list = []
    if type(sentence_list)==str:
        text = sentence_list
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        encoded_input.to(device)
        output = model(**encoded_input)
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)
        negative_score = scores[0]
        positive_score = scores[2]
        new_negative = negative_score/(negative_score+positive_score)
        new_positive = positive_score/(negative_score+positive_score)
        score_list.append(new_positive)
    else:
        for text in sentence_list:
            text = preprocess(text)
            encoded_input = tokenizer(text, return_tensors='pt')
            encoded_input.to(device)
            output = model(**encoded_input)
            scores = output[0][0].detach().cpu().numpy()
            scores = softmax(scores)
            negative_score = scores[0]
            positive_score = scores[2]
            new_negative = negative_score/(negative_score+positive_score)
            new_positive = positive_score/(negative_score+positive_score)
            score_list.append(new_positive)
    return score_list

if __name__ == "__main__":
    target_list = ["I like none","I approve every decision this president makes.", "Trump sucks", "Biden sucks","Unless otherwise stated, I'm adding Bitcoin every week","#Bitcoin for the win","#Bitcoin lost a lot of value today, #wasteful",
    r"Bitcoin falls from $66K highs, Tesla down 3% after Elon Musk warns he could sell more #Bitcoin", "it is stupid not to buy bitcoin"]

    scores = eval_sentences_positive(target_list)

    print(scores)
