"""
Never knowledge IN Binary Classification model & function

- learning method: supervised learning, binary classification
- learning algorithm: RNN, LSTM

- datasets: '../data/
- features: peded rows?
- label: 0, 1
- label translate: 0: others, 1: psychiatry
- frame work: Pytorch
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optima
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset

import torchinfo, torchmetrics
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryConfusionMatrix
from torchinfo import summary

import string
import spacy
import soynlp
import konlpy
from konlpy.tag import Okt


# 함수 선언하기 전에 설정 하는 것
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 항상 설정하지만.. 안쓴다
RANDOM_STATE = 10
torch.manual_seed(RANDOM_STATE)
LABEL_TRANSLATE = {0:'others', 1:'psychiatry'}
DATA_ROOT = './data/'
ko_model = 'ko_core_news_sm'
ko_spacy = spacy.load(ko_model)
ko_okt = Okt()
ko_soynlp = soynlp.word.WordExtractor()
punct = string.punctuation

# 기본적으로 확인하는 utils
def utils():
    """
    utils: package version, random_state, device
    ---
    - device
    - random_state
    ----
    - versions:
    pandas, numpy, matplotlib, torch, torchinfo, torchmetrics,
    spacy, konlpy, soynlp
    """
    print('----- Notice -----')
    print(f"device: {DEVICE}")
    print(f"random state: {RANDOM_STATE}\n")
    
    print(f"pandas ver: {pd.__version__}")
    print(f"numpy ver: {np.__version__}")
    print(f"matplotlib ver: {matplotlib.__version__}\n")
    
    print(f"torch ver: {torch.__version__}")
    print(f"torchinfo ver: {torchinfo.__version__}")
    print(f"torchmatrics ver: {torchmetrics.__version__}\n")
    
    print(f"spacy ver: {spacy.__version__}")
    print(f"soynlp ver: {soynlp.__version__}")
    print(f"konlpy ver: {konlpy.__version__}\n")
   

# load file list function
# ---------------------------------
# function name: load_file_list
# function parameters: data_root, num_range
# function return: file_list (dict)

def load_file_list(data_root='', num_range=2):
    """
    load file list to dict.

    Args:
        data_root (str): data root path
        num_range (int, optional): folder length. Defaults to 8.

    Returns:
        dict: file list
    """
    file_list_dict = {}
    for i in range(num_range):
        data_path = data_root+f'{i}/'
        file_list_dict[i] = os.listdir(data_path)
        
    return file_list_dict


# read file function
# ----------------------------------
# function name: read_file
# function parameters: data_root, file_dict
# function return: sentence (just one sentence)

def read_file(file_list, i, type_='train/'):
    """
    read data in file list

    Args:
        file_dict (dict): file list dict

    Yields:
        list: lines data
    """
    
    for file_name in file_list:
        with open(DATA_ROOT+type_+f'{i}/'+file_name, mode='rt', encoding='utf-8') as f:
            all_text = []
            while True:
                text = f.readline()
                text = text.replace('\n', '')
                text = text.replace('\t', '')
                text = re.sub('[^ㄱ-ㅎ가-힣]+', ' ', text)
                
                if not text:
                    break
                else:
                    all_text.append(text)
        yield all_text



# create data frame function
# ----------------------------------------
# function name: create_df
# function parameters: file_dict
# function return: data_df

def create_df(file_dict, type_='train/'):
    """
    create dataframe

    Args:
        file_dict (dict): file list dict

    Returns:
        DataFrame: data DataFrame
    """
    data_df = pd.DataFrame(columns=['text', 'label'])
    
    for i in range(len(file_dict)):
        carry = []
        for text_list in read_file(file_dict[i], i, type_):
            for text in text_list:
                carry.append(text)
                
        carry_df = pd.DataFrame(columns=['text', 'label'])
        carry_df['text'] = carry
        carry_df['label'] = i
        
        data_df = pd.concat([data_df, carry_df], ignore_index=True)
    return data_df


# load korean stopword
# ----------------------------------
# function name: load_ko_stop
# function parameter: None
# function return: ko_stopwords (list)

def load_ko_stopwrod():
    """
    load korean stopwords

    Returns:
        list: stopwords
    """
    ko_stopwords = []
    
    with open(r'C:\WorkSpace\KDT\TORCH_TEXT\D1010\cgi-bin\stopwords-ko.txt', mode='tr', encoding='utf-8') as f:
        while True:
            word = f.readline().replace('\n', '')
            if not word:
                break
            else:
                ko_stopwords.append(word)
    return ko_stopwords


# 만들어진거 저장해주는 함수
def add_df(data_df, addtion, name):
    data_df[name] = addtion
    return data_df


# delete stopwords function
# ---------------------------------------------
# function name: delete_stopwords
# function parameters: data_df, stopwords, nlp_type
# function return: token_list

def delete_stopwords(data_df, stopwords, nlp_type=ko_okt):
    """
    delete stopwords

    Args:
        data_df (DataFrame): text data
        stopwords (list): korean stopwords
        nlp_type (instance, optional): korean nlp instance. Defaults to ko_okt.

    Yields:
        list: del stopwords token list
    """
    if nlp_type == ko_spacy:
        for text in data_df['text']:
            doc = ko_spacy(text)
            token_list = []
            for token in doc:
                if not token in stopwords:
                    token_list.append(token)
            yield token_list
    
    elif nlp_type == ko_okt:
        token_list = []
        for text in data_df['text']:
            morphs = ko_okt.morphs(text)
            carry = []
            for token in morphs:
                if not token in stopwords:
                    carry.append(token)
            token_list.append(carry)
        yield token_list
    
    elif nlp_type == ko_soynlp:
        token_list = []
        ko_soynlp.train(data_df['text'])
        for text in data_df['text']:
            result = ko_soynlp.extract()
            score = {word:score.cohesion_forward for word, score in result.items()}
            tokenizer = soynlp.tokenizer.LTokenizer(score)
            carry = []
            for token in tokenizer.tokenize(text):
                if not token in stopwords:
                    carry.append(token)
            token_list.append(carry)
        yield token_list
        

# token frequency function
# -----------------------------------
# function name: token_frequency
# function parameter: data_df
# function return: token_freq

def token_frequence(data_df):
    """
     token frequency

    Args:
        data_df (DataFarme): text data

    Returns:
        dict: token dict
    """
    token_freq = {}
    for token_list in data_df['del_stop']:
        for token in token_list:
            if token not in token_freq:
                token_freq[token] = 0
            token_freq[token] += 1
    return token_freq


# count token frequency function
# -------------------------------------------
# function name: count_token_frequency
# function parameter: token_freq
# function return: cnt_token_freq

def count_token_frequency(token_freq):
    """
    count token frequency

    Args:
        token_freq (dict): token frequency dict

    Returns:
        dict: count frequency
    """
    cnt_token_dict = {}
    for key, value in token_freq.items():
        if value not in cnt_token_dict:
            cnt_token_dict[value] = [1, [key]]
        else:
            cnt_token_dict[value][1].append(key)
            cnt_token_dict[value][0] += 1
    return cnt_token_dict


# create data vacob function
# ------------------------------------------------
# function name: create_vocab
# function parameter: token_freq
# function return: data_vocab

def create_vocab(token_freq):
    """
    create data vocab

    Args:
        token_freq (dict): token frequency dict

    Returns:
        dict: data vocab dict
    """
    sorted_list = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    
    PAD_TOKEN, OOV_TOKEN = 'PAD', 'OOV'
    data_vocab = {PAD_TOKEN:0, OOV_TOKEN:1}
    
    for idx, tk in enumerate(sorted_list, 2):
        # if idx > 5000:
        #     break
        data_vocab[tk[0]] = idx
        
    vocab_df = pd.DataFrame(columns=['key', 'value'])
    vocab_df['key'] = data_vocab.keys()
    vocab_df['value'] = data_vocab.values()
    vocab_df.to_csv('./psychiatry_vocab.csv', encoding='utf-8', index=False)
    return data_vocab


# encoding function
# -----------------------------------------
# function name: encoding
# function parameter: data_df, data_vocab
# function return: encoding_data

def encoding(data_df, data_vocab):
    """
    token encoding

    Args:
        data_df (DataFrame): text data
        data_vocab (dict): data vocab dict

    Yields:
        list: encoding token data
    """
    encoding_data = []
    for token_list in data_df['del_stop']:
        sent = []
        for token in token_list:
            try:
                sent.append(data_vocab[token])
            except:
                sent.append(data_vocab['OOV'])
        encoding_data.append(sent)
    yield encoding_data
    

# show sent population function
# ---------------------------------------------------
# function name: show_sent_pop
# function parameter: data_df
# function return: None

def show_sent_pop(data_df):
    """
    show histogram sent population

    Args:
        data_df (DataFrame): encoding text data
    """
    encoding_data = data_df['encoding'].to_list()
    
    data_len_list = [len(sent) for sent in encoding_data]
    
    plt.hist(data_len_list)
    plt.title('sent population')
    plt.xlabel('sentence length')
    plt.ylabel('sentence count')
    plt.show()
    
    
# padding function
# -----------------------------------
# function name: padding
# function parame: data_df, len_data
# function return: padding_data

def padding(data_df, len_data=10):
    """
    sent padding len_data

    Args:
        data_df (DataFrame): encoding text data
        len_data (int, optional): want to padding length. Defaults to 10.

    Returns:
        list: padded text data
    """
    padding_data = data_df['encoding'].to_list()
    for idx, sent in enumerate(padding_data):
        current_len = len(sent)
        if current_len < len_data:
            sent.extend([0]*(len_data-current_len))
            padding_data[idx] = sent
        else:
            # sent = sent[current_len-len_data:]
            sent = sent[:len_data]
            padding_data[idx] = sent
    return padding_data


# data binary classification model
# --------------------------------------------------------------
# class perpose: data binary classification
# class name: BCRNNModels
# parameters: n_vocab, hidden_dim, embdding_dim, n_layers, dropout, bidirectional, model_type
# attribute field: embedding, model, classifier, dropout
# class function: create model structure, forward learning model
# class structure
# - embedding
# - model
# - classifier
# - dropout

class BCRNNModels(nn.Module):
    def __init__(self, n_vocab, hidden_dim, embedding_dim, n_layers, dropout=0.5, bidirectional=True, model_type='lstm'):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        if model_type == 'rnn':
            self.model = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
        elif model_type == 'lstm':
            self.model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
                proj_size=0
            )
        if bidirectional:
            self.classifier = nn.Linear(hidden_dim*2, 1)
        else:
            self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.model(embeddings)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)
        logits = F.sigmoid(logits)
        return logits


# model class
class SentenceClassifier(nn.Module):
    def __init__(self, n_vocab, hidden_dim, embedding_dim, n_layers, dropout=0.5, bidirectional=True, model_type='lstm'):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        if model_type == 'rnn':
            self.model = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
        elif model_type == 'lstm':
            self.model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
        if bidirectional:
            self.classifier = nn.Linear(hidden_dim*2, 1)
        else:
            self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.model(embeddings)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)
        return logits



# predict function
# -----------------------------------------------------
# - function name: predict
# - parameter: model, X_data
# - function return: loss, score_accuracy, score_f1score, proba
# -----------------------------------------------------
# must not update weight & bais

def predict_web(model, X_data, type_='others'):
    X_data = X_data.unsqueeze(1)
    with torch.no_grad():
        if type_=='me':
            pred = model(X_data)
            # pred = torch.max(pred)
            # pred = torch.argmax(pred)
        elif type_=='others':
            pred = model(X_data)
            pred = torch.sigmoid(pred)
            # pred = torch.max(pred)
            # pred = torch.argmax(pred)
        # pred_label = torch.argmax((pred > 0.5).int())
        # pred_label = int(pred_label.flatten())
        # if pred_label > 1:
        #     pred_label = 0
        # pred_label = lable_translate[pred_label]
        pred = torch.max(pred)
        
    return pred.item()

def predict(model, x_data, y_data):
    # 텐서 하나
    # y_data의 차원을 확인하고 unsqueeze가 필요한지 확인
    if len(y_data.shape) == 1:  # 1D 텐서인 경우
        y_data = y_data.unsqueeze(1)  # (N,) -> (N, 1)
    
    y_data = y_data.float()  # float형으로 변환
    x_data = x_data.unsqueeze(1)
    with torch.no_grad():
        # 예측 수행
        pred = model(x_data)
        # 0.5 이상이면 1, 미만이면 0으로 변환
        pred_labels = torch.argmax((pred > 0.5).int())
        pred_labels = [LABEL_TRANSLATE[int(label)] for label in pred_labels.flatten()]
        real_labels = [LABEL_TRANSLATE[int(label)] for label in y_data.flatten()]

    return pred_labels, real_labels

#python -m http.server --cgi 8080
#터미널에 입력해서 실행

