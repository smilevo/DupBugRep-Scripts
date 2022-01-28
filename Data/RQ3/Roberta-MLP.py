# -*- coding: utf-8 -*-

#Roberta

import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import RobertaModel, RobertaTokenizer


class Settings:
    batch_size=86
    max_len=300
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #seed = 200 #318

#dataset
class TrainValidDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.text = df["clean_pair"].values
        #self.target = df["target"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        texts = self.text[idx]
        tokenized = self.tokenizer.encode_plus(texts, truncation=True, add_special_tokens=True,
                                               max_length=self.max_len, padding="max_length")
        ids = tokenized["input_ids"]
        mask = tokenized["attention_mask"]
        #targets = self.target[idx]
        return {
            "ids": torch.LongTensor(ids),
            "mask": torch.LongTensor(mask),
            #"targets": torch.tensor(targets, dtype=torch.float32)
        }   

#model
class CommonLitRoBERTa(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        
    def forward(self, ids, mask):
        output = self.roberta(ids, attention_mask=mask)
        return output    

model = CommonLitRoBERTa("/content/roberta-transformers-pytorch/roberta-base")
model.to(Settings.device)
#Get embeddings
tokenizer = RobertaTokenizer.from_pretrained("/content/roberta-transformers-pytorch/roberta-base")


df_train = pd.read_csv("/content/eclipse_training.csv")
#df_train.shape

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
import string

def clean_doc(doc):
	tokens = doc.split()
	table = str.maketrans('', '', string.punctuation)
	#nltk.max_length=300
	tokens = [w.translate(table) for w in tokens]
	tokens = [word for word in tokens if word.isalpha()]
	stop_words = list(set(stopwords.words('english')))
	newStopWords = ['java','com','org']
	stop_words.extend(newStopWords)
	tokens = [w for w in tokens if not w in stop_words]
	tokens = [word for word in tokens if len(word) > 1]
	ps = PorterStemmer()
	tokens=[ps.stem(word) for word in tokens]
	#max_length(tokens)
	return tokens




from nltk.stem.porter import PorterStemmer
df_train["pair"]= df_train["title1"] + df_train["description1"] + [" [SEP] "] + df_train["title2"] + df_train["description2"]
df_train['clean_pair'] =df_train['pair'].apply(lambda x: clean_doc(x))


train_dataset = TrainValidDataset(df_train[10492 : 10578 ], tokenizer, Settings.max_len)
train_loader = DataLoader(train_dataset, batch_size=Settings.batch_size)#shuffle=True, num_workers=8, pin_memory=True

batch = next(iter(train_loader))


ids = batch["ids"].to(Settings.device)
mask = batch["mask"].to(Settings.device)


print(ids.shape)
print(mask.shape)

output = model(ids, mask)
output

last_hidden_state = output[0]
print("shape:", last_hidden_state.shape)


cls_embeddings = last_hidden_state[:, 0, :].detach()

print("shape:", cls_embeddings.shape)

df = pd.DataFrame(cls_embeddings.numpy())
df.head()


#MLP

import pandas as pd
df_train=pd.read_csv('eclipse_train_dataset.csv')
df_test=pd.read_csv('eclipse_test_dataset.csv')

#
df_train.shape
df_test.shape

train_features = df_train.drop(columns=['Label'], axis=1)
print(train_features.shape)

train_labels =df_train['Label']
print(train_labels.shape)

test_features = df_test.drop(columns=['Label'], axis=1)
print(test_features.shape)

test_labels =df_test['Label']
print(test_labels.shape)

#Importing MLPClassifier
from sklearn.neural_network import MLPClassifier

#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(100,), alpha=0.0001, activation = 'relu',solver='adam',learning_rate='adaptive')

#Fitting the training data to the network
classifier.fit(train_features, train_labels)


y_true, y_pred = test_labels , classifier.predict(test_features)
np.set_printoptions(threshold=np.inf)
y_pred

from sklearn.metrics import classification_report, confusion_matrix
print('Results on the test set:')
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))



from sklearn.metrics import accuracy_score
#score=accuracy_score(y_test, predictions)
test_score=accuracy_score( test_labels, y_pred)
print('Accuracy:',test_score)