# -*- coding: utf-8 -*-

############################ Sentence-BERT ##########################################

import pandas as pd
df= pd.read_csv('/content/test_thunderbird.csv')

from gensim.utils import simple_preprocess
cleaned_title1 = df['title1'].apply(lambda x: ' '.join(simple_preprocess(x)))
cleaned_description1 = df['description1'].apply(lambda x: ' '.join(simple_preprocess(x)))
cleaned_title2 = df['title2'].apply(lambda x: ' '.join(simple_preprocess(x)))
cleaned_description2 = df['description2'].apply(lambda x: ' '.join(simple_preprocess(x)))

BR1= cleaned_title1 + cleaned_description1
BR2= cleaned_title2 + cleaned_description2

word_embedding_model = models.Transformer('bert-base-uncased',max_seq_length=150)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Each sentence is encoded as a 1-D vector with 768 columns
BR1_embeddings = model.encode(BR1)
print(BR1_embeddings)
BR2_embeddings= model.encode(BR2)
print(BR2_embeddings)

import numpy as np
features =np.concatenate([BR1_embeddings,BR2_embeddings], axis=1)
print(features)

pd.DataFrame(features).to_csv("thunderbird_test_SBert.csv")



############################### MLP Classification ####################################
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=100)

parameter_space = {
    'hidden_layer_sizes': [(50,100,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(train_features, train_labels)

# Best paramete set
print('Best parameters found:\n', clf.best_params_)

import numpy as np
from sklearn.metrics import accuracy_score
y_pred = clf.predict(test_features)

print('Accuracy: {:.2f}'.format(accuracy_score(test_labels, y_pred)))

from sklearn.metrics import classification_report, confusion_matrix
print('Results on the test set:')
print(classification_report(test_labels, y_pred))
print(confusion_matrix(test_labels, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(test_labels,y_pred)
print(cm)

plt.figure(figsize=(9,9))
plt.rcParams.update({'font.size': 16})
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.8, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
