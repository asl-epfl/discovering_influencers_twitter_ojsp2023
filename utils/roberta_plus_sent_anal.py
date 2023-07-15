from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import torch
device = torch.device("cpu") ##cuda gave some errors

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
def preprocess(text):
    
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# PT
model = AutoModelForSequenceClassification.from_pretrained("/home/cemri/graph_analysis/utils//xlm-roberta-sentiment-multilingual")
model.to(device)
"""
text = "Good night 😊"
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
"""
"""
# Print labels and scores
ranking = np.argsort(scores)
for i in range(scores.shape[0]):
    s = scores[ranking[i]]
    print(i, s)
"""

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

#target_list = ["I like none","I approve every decision this president makes.", "Trump sucks", "Biden sucks"]

#scores = eval_sentences_positive(target_list)

#print(scores)