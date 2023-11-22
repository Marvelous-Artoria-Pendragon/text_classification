# 模块导入
import pandas as pd
import torch
import os
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
from tqdm import tqdm
from wordcloud import WordCloud
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler
from torch import nn, optim

def clean_data(x):
    x = str(x)
    x = x.replace('\n', ' ')
    x = x.replace('\r', '')
    x = x.replace('\t', ' ')
    x = x.replace('展开全文', '')
    return x

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_train = pd.read_csv(r"D:\Programming\python\Text_Classification\Emotional Recognition of netizens during the epidemic period\nCoV_100k_train_labled.csv")
    data_test = pd.read_csv(r"D:\Programming\python\Text_Classification\Emotional Recognition of netizens during the epidemic period\nCov_10k_test.csv")

    # print(data_train.info())
    # print(data_test.info())

    data_train = data_train[['微博id', '微博中文内容', '情感倾向']]
    data_test = data_test[['微博id', '微博中文内容']]
    data_train.columns = ['id', 'content', 'y']
    data_test.columns = ['id', 'content']


    data_train['content'] = data_train['content'].apply(clean_data)
    data_test['content'] = data_test['content'].apply(clean_data)
    data_train['text_len'] = data_train['content'].apply(len)
    data_test['text_len'] = data_test['content'].apply(len)


    x_range = [i for i in range(256)]
    train_distribution = np.zeros(256)
    test_distribution = np.zeros(256)

    data_train['text_len'].apply(lambda x: train_distribution.__setitem__(x - 1, train_distribution[x -1] + 1))
    data_test['text_len'].apply(lambda x: test_distribution.__setitem__(x - 1, test_distribution[x -1] + 1))

    # 清理缺失、错误标签
    # 这里为方便直接删除，整体上不影响
    nul_index =  data_train.loc[~data_train['y'].isin(['0', '-1', '1'])].index
    data_train.drop(nul_index, inplace=True)
    data_train['y'] = data_train['y'].astype(int)
    data_train['y'] += 1
    return data_train, data_test

class Config:
    num_classes = 3
    lr = 5e-4
    batch_size = 16
    shuffle = True
    epoch = 5
    num_workers = -1
    scheduler_step_size = 1
    scheduler_gamma = 0.9
    max_seq_len = 256
    model_name = 'D:\\Programming\\python\\Text_Classification\\bert-base-chinese'
    # model_name = 'hfl/chinese-bert-wwm-ext'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_save_dir = './model' 
    model_save_name = 'bert.pt'
    
    # training parameters
    num_epochs = 10
    snapshots = 2
    useTest = True
    test_interval = 1

class TextDataset(Dataset):
    def __init__(self, texts: list | np.ndarray, labels: list | np.ndarray):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx: int) -> tuple[str, int]:
        text_i = self.texts[idx]
        label_i = self.labels[idx]
        return text_i, label_i

    def __len__(self) -> int:
        assert len(self.texts) == len(self.labels)
        return len(self.labels)

def collate_fn(text_label: tuple[str, int]):
    tokenizer = BertTokenizer.from_pretrained(Config.model_name)
    texts = [text for text, _ in text_label]
    labels = [label for _, label in text_label]
    # The padding length can't surpass the max sequence length
    max_len = min(max([len(text) for text in texts]) + 2, Config.max_seq_len)

    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs = texts, 
                                       add_special_tokens = True,       # add [CLS], [SEP], [EOP]
                                       padding = 'max_length', 
                                       truncation = True, 
                                       max_length = max_len,
                                       return_tensors = 'pt',
                                       return_token_type_ids = True,
                                       return_attention_mask = True,)
    inputs_ids = data['input_ids']
    attention_mask = data['attention_mask']     # mask the padding value
    token_type_ids = data['token_type_ids']
    labels = torch.FloatTensor(labels)
    return inputs_ids, attention_mask, token_type_ids, labels

class BertClassifier(nn.Module):
    def __init__(self, hiddent_size: int, num_classes: int, model_name: str = Config.model_name) -> None:
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(hiddent_size, num_classes)
        # self.drop = nn.Dropout(p=0.3)
        # self.out = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.bert(input_ids = input_ids, 
                            attention_mask = attention_mask, 
                            token_type_ids = token_type_ids)
            # out->(batch_size, seq_len, hidden_size)
            # we just need the first row, which is stand for the label
        prediction = self.fc(out.last_hidden_state[:, 0, :])
        return prediction

def train():
    
    model = BertClassifier(hiddent_size = 768, num_classes = Config.num_classes)
    optimizer = optim.AdamW(model.parameters(), lr = Config.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = Config.scheduler_step_size, gamma = Config.scheduler_gamma)
    
    model.to(Config.device)
    criterion.to(Config.device)

    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)

    for epoch in range(Config.num_epochs):
        running_loss = 0.0
        running_corrects = 0.0

        start_time = timeit.default_timer()
        model.train()
        for input_ids, attention_mask, token_type_ids, labels in tqdm(train_loader):
            input_ids = input_ids.to(Config.device)
            attention_mask = attention_mask.to(Config.device)
            token_type_ids = token_type_ids.to(Config.device)
            labels = labels.to(Config.device).requires_grad_()
            optimizer.zero_grad()
            outputs = model(input_ids = input_ids, 
                               attention_mask = attention_mask, 
                               token_type_ids = token_type_ids)
            prediction = torch.max(outputs, 1)[1]
            loss = criterion(prediction.float(), labels.float())
            loss.backward()
            optimizer.step()

            scheduler.step()
            running_loss += loss.item()
            running_corrects += torch.sum(prediction == labels.data)


        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print("[train] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, Config.num_epochs, epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

        if (epoch + 1) % Config.snapshots == 0:
            if not os.path.exists(Config.model_save_dir):
                os.makedirs(Config.model_save_dir)
            save_path = os.path.join(Config.model_save_dir, Config.model_save_name + '_epoch-' + str(epoch) + '.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_acc,
            }, save_path)
            print("Save model at {}\n".format(save_path))

        if Config.useTest and epoch % Config.test_interval == (Config.test_interval - 1):
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            model.eval()
            for input_ids, attention_mask, token_type_ids, labels in tqdm(test_loader):
                input_ids = input_ids.to(Config.device)
                attention_mask = attention_mask.to(Config.device)
                token_type_ids = token_type_ids.to(Config.device)
                labels = labels.to(Config.device)

                with torch.no_grad():
                    outputs = model(input_ids = input_ids, 
                                        attention_mask = attention_mask, 
                                        token_type_ids = token_type_ids)

                prediction = torch.max(outputs, 1)[1]
                loss = criterion(prediction.float(), labels.float())


                running_loss += loss.item()
                running_corrects += torch.sum(prediction == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, Config.num_epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")




if __name__ == "__main__":
    data_train, data_test = load_data()
    # 导入模型
    X_train, X_test, y_train, y_test = train_test_split(data_train['content'].tolist(), data_train['y'].tolist(), test_size = 0.2, random_state = 42)
    train_loader = DataLoader(TextDataset(X_train, labels = y_train), 
                            batch_size = Config.batch_size, collate_fn = collate_fn, shuffle = Config.shuffle)
    test_loader = DataLoader(TextDataset(X_test, labels = y_test), 
                            batch_size = Config.batch_size, collate_fn = collate_fn, shuffle = Config.shuffle)
    train()