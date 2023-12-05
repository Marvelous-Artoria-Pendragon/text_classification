import numpy as np
import pandas as pd
import torch
import warnings
import pandas as pd
import pickle as pk
import timeit
import os
import logging
from transformers import BertConfig, BertModel, BertTokenizer
from tqdm import tqdm 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler
from sklearn.linear_model import LogisticRegression as LR
from torch import nn, optim

warnings.filterwarnings('ignore')
bert_models = {0: 'bert-base-uncased',
               1: 'bert-large-uncased',
               2: 'bert-base-cased',
               3: 'bert-large-cased',
               4: 'bert-base-multilingual-uncased',
               5: 'bert-base-multilingual-cased',
               6: 'bert-base-chinese',
               7: 'bert-base-german-cased',
               8: 'bert-large-uncased-whole-word-masking',
               9: 'bert-large-cased-whole-word-masking',
               10: 'bert-large-uncased-whole-word-masking-finetuned-squad',
               11: 'bert-large-cased-whole-word-masking-finetuned-squad',
               12: 'bert-base-cased-finetuned-mrpc',
               13: 'bert-base-german-dbmdz-cased',
               14: 'bert-base-german-dbmdz-uncased',
               15: 'cl-tohoku/bert-base-japanese',
               16: 'cl-tohoku/bert-base-japanese-whole-word-masking',
               17: 'cl-tohoku/bert-base-japanese-char',
               18: 'cl-tohoku/bert-base-japanese-char-whole-word-masking',
               19: 'TurkuNLP/bert-base-finnish-cased-v1',
               20: 'TurkuNLP/bert-base-finnish-uncased-v1',
               21: 'wietsedv/bert-base-dutch-cased',
               22: 'bert-base-uncased-hm',
               23: 'bert-base-cased-hm',
               24: 'bert-large-uncased-hm',
               25: 'bert-large-cased-hm',
               26: 'albert-base-v1',
               27: 'albert-large-v1',
               28: 'albert-xlarge-v1',
               29: 'albert-xxlarge-v1',
               30: 'albert-base-v2',
               31: 'albert-large-v2',
               32: 'albert-xlarge-v2',
               33: 'voidful/albert_chinese_small',
               34: 'roberta-base',
               35: 'roberta-large',
               36: 'roberta-large-mnli',
               37: 'roberta-base-mnli',
               38: 'roberta-base-openai-detector',
               39: 'roberta-large-openai-detector',
               40: 'distilbert-base-uncased',
               42: 'distilbert-base-cased',
               43: 'distilbert-base-multilingual-cased',
               44: 'distilbert-base-multilingual-uncased',
               45: 'distilbert-base-uncased-distilled-squad',
               46: 'distilbert-base-cased-distilled-squad',
               47: 'distilbert-base-cased-finetuned-mrpc',
               48: 'distilbert-base-uncased-finetuned-mrpc',
               49: 'distilbert-base-cased-finetuned-sst-2-english',
               50: 'distilbert-base-uncased-finetuned-sst-2-english',
               51: 'distilbert-base-cased-finetuned-qqp',
               52: 'distilbert-base-uncased-finetuned-qqp',
               53: 'distilbert-base-cased-finetuned-qnli',
               54: 'distilbert-base-uncased-finetuned-qnli',
               55: 'distilbert-base-cased-finetuned-mnli',
               56: 'distilbert-base-uncased-finetuned-mnli',
               57: 'distilbert-base-cased-finetuned-rte',
               58: 'distilbert-base-uncased-finetuned-rte',
               59: 'distilbert-base-cased-finetuned-wnli',
               60: 'distilbert-base-uncased-finetuned-wnli',
               61: 'distilbert-base-cased-whole-word-masking',
               62: 'distilbert-base-uncased-whole-word-masking',
               63: 'distilbert-base-cased-finetuned-adapter',
               64: 'distilbert-base-uncased-finetuned-adapter',
               65: 'distilbert-base-cased-finetuned-pos-squad',
               66: 'distilbert-base-uncased-finetuned-pos-squad',
               67: 'distilbert-base-cased-finetuned-ner-conll03',
               68: 'distilbert-base-uncased-finetuned-ner-conll03',
               69: 'distilbert-base-cased-finetuned-ner-msra',
               70: 'distilbert-base-uncased-finetuned-ner-msra',
               71: 'distilbert-base-cased-finetuned-ner-weibo',
               72: 'distilbert-base-uncased-finetuned-ner-weibo',
               73: 'distilbert-base-cased-finetuned-stsb-200',
               74: 'distilbert-base-uncased-finetuned-stsb-200',
               75: 'distilbert-base-cased-finetuned-qa-squad1',
               76: 'distilbert-base-uncased-finetuned-qa-squad1',
               77: 'distilbert-base-cased-finetuned-qa-squad2',
               78: 'distilbert-base-uncased-finetuned-qa-squad2',
               79: 'distilbert-base-cased-finetuned-adapter-extended-bert-base-case',
               80: 'distilbert-base-uncased-finetuned-adapter-extended-bert-base-uncase',
               81: 'distilbert-base-cased-finetuned-conll03-english',
               82: 'distilbert-base-uncased-finetuned-conll03-english',
               83: 't5-small',
               84: 't5-base',
               85: 't5-large',
               86: 't5-3b',
               87: 't5-11b',
               88: 'xlm-roberta-base',
               89: 'xlm-roberta-large',
               90: 'gpt2',
               91: 'gpt2-xl',
               92: 'transfo-xl-wt103',
               93: 'xlnet-base-cased',
               94: 'xlnet-large-cased',
               95: 'xlm-mlm-en-2048',
               96: 'xlm-mlm-ende-1024',
               }

class Config:
    
    # model name or the model's local path
    model_name = bert_models[6]
    model_save_name = model_name + '_fine-tuning.pt'
    output_dir = './output' 
    log_path = os.path.join(output_dir, model_name.replace('/', '-') + "_fine-tuning.log")
    dataset_path = r'C:\Artoria\Code\Text_Classification\NLP_Corpus_ZH\datasets\online_shopping_10_cats\online_shopping_10_cats.csv'
    
    # training parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    hidden_size = 768
    lr = 5e-4
    batch_size = 64
    shuffle = True
    num_workers = -1
    weight_decay = 0.001
    scheduler_step_size = 1
    scheduler_gamma = 0.9
    scheduler_warmup_steps = -1             # default from 0
    max_seq_len = 256
    num_epochs = 5
    snapshots = 1
    useTest = True
    max_grad_norm = 1.0
    test_interval = 1

    # visualization parameters
    displayModel = True
    plot_loss = True

if not os.path.exists(Config.output_dir):
    os.makedirs(Config.output_dir)

def getLogger(log_path: str = 'traininglog.log') -> logging.Logger:
    # logger config
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename = log_path, encoding = 'utf-8')
    sh = logging.StreamHandler()
    fh.setLevel(logging.DEBUG)
    sh.setLevel(logging.DEBUG)
    # file formatter
    ffmt = logging.Formatter(fmt = "%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                            datefmt = "%d-%M-%Y %H:%M:%S")
    # stream formatter
    sfmt = logging.Formatter(fmt = "%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                            datefmt = "%d-%M-%Y %H:%M:%S")
    fh.setFormatter(ffmt)
    sh.setFormatter(sfmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str:int], dict[int:str]]:
    """
    data preprocess

    return (data_train, data_test, cla2id, id2cla)
    """
    # rewrite this method to adapt your data
    df = pd.read_csv(Config.dataset_path)
    cla2id = {}
    id2cla = {}
    for i, cat in enumerate(df['cat'].unique()):
        cla2id[cat] = i
        id2cla[i] = cat

    data_train, data_test = train_test_split(df, test_size=0.2, random_state=42)
    ## data preprocessing
    # type here
    data_train['cat'] = data_train['cat'].apply(lambda x: cla2id[x])
    data_test['cat'] = data_test['cat'].apply(lambda x: cla2id[x])
    
    data_train = data_train[['review', 'cat']]
    data_test = data_test[['review']]
    
    data_train.columns = ['text', 'y']
    data_test.columns = ['text']


    return data_train, data_test, cla2id, id2cla

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

def collate_fn(text_label: tuple[str, int]) -> tuple[torch.IntTensor, torch.FloatTensor, torch.IntTensor, torch.FloatTensor]:
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
        
        # fetch the features here
        # features = out.last_hidden_states[:, 0, :].numpy() or
        # features = out[0][:, 0, :].numpy()
        logits = self.fc(out.last_hidden_state[:, 0, :])
        return logits

def displayBertModel(model: nn.Module):
    # convert all model's parameters to a list
    params = list(model.named_parameters())

    logger.info('\nThe BERT model has {:} different named parameters.\n'.format(len(params)))
    logger.info('\n==== Embedding Layer ====\n')
    for p in params[0:5]:
        logger.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    logger.info('\n==== First Transformer ====\n')
    for p in params[5:-4]:
        logger.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    logger.info('\n==== Output Layer ====\n')
    for p in params[-3:]:
        logger.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

def train(cla2id: dict | None = None, id2cla: dict | None = None) -> pd.DataFrame:
    
    model = BertClassifier(hiddent_size = Config.hidden_size, num_classes = Config.num_classes)

    if Config.displayModel:
        displayBertModel(model)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'
        {'params': (p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)),
         'weight_decay': Config.weight_decay},
        
        # Filter for parameters which *do* include those
        {'params': (p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)), 
         'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr = Config.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size = Config.scheduler_step_size,
                                          last_epoch = Config.scheduler_warmup_steps,
                                          gamma = Config.scheduler_gamma)
    
    model.to(Config.device)
    criterion.to(Config.device)

    train_size = len(train_dataloader.dataset)
    test_size = len(test_dataloader.dataset)
    total_steps = len(train_dataloader) * Config.num_epochs
    train_stats = []
    
    for epoch in range(Config.num_epochs):
        running_loss = 0.0
        running_corrects = 0.0

        train_start_time = timeit.default_timer()
        # training
        model.train()
        logger.info("===========start training============")
        train_loop = tqdm(train_dataloader, desc = 'Train')
        for input_ids, attention_mask, token_type_ids, labels in train_loop:
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
            # gradient clip, avoid gradient explosion, must be between backward and step
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = Config.max_grad_norm, norm_type = 2)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            corrects = torch.sum(prediction == labels.data).item()
            running_corrects += corrects
            # update train info
            train_loop.set_description(f'Epoch [{epoch+1}/{Config.num_epochs}]')
            train_loop.set_postfix(loss = loss.item() / len(labels), acc = corrects / len(labels), label_size = len(labels))


        train_loss = running_loss / train_size
        train_acc = running_corrects / train_size
        logger.info("[train] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, Config.num_epochs, train_loss, train_acc))
        train_stop_time = timeit.default_timer()
        logger.info("Execution time: " + str(train_stop_time - train_start_time) + "\n")
        
        # save model per snapshots
        if (epoch + 1) % Config.snapshots == 0:
            save_path = os.path.join(Config.output_dir, Config.model_save_name + '_epoch-' + str(epoch) + '.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': train_acc,
                'training_time': train_stop_time - train_start_time,
                'cla2id': cla2id,
                'id2cla': id2cla,
            }, save_path)
            logger.info("Save model at {}\n".format(save_path))

        # do test per test_interval
        logger.info("===========start validating============")
        if Config.useTest and epoch % Config.test_interval == (Config.test_interval - 1):

            running_loss = 0.0
            running_corrects = 0.0

            test_start_time = timeit.default_timer()
            model.eval()
            test_loop = tqdm(test_dataloader, desc = 'Test')
            for input_ids, attention_mask, token_type_ids, labels in test_loop:
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
                corrects = torch.sum(prediction == labels.data).item()
                running_corrects += corrects
                # update test info
                test_loop.set_postfix(loss = loss.item() / len(labels), acc = corrects / len(labels))

            test_loss = running_loss / test_size
            test_acc = running_corrects / test_size

            logger.info("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, Config.num_epochs, test_loss, test_acc))
            test_stop_time = timeit.default_timer()
            logger.info("Execution time: " + str(test_stop_time - test_start_time) + "\n")
        
        # save train&test message
        train_stats.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'training_time': train_stop_time - train_start_time,
            'test_time': test_stop_time - test_start_time
        })

    # save model and tokenizer parameters
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(Config.output_dir)
    # tokenizer.save_pretrained(Config.output_dir)

    df_stats = pd.DataFrame(data = train_stats)
    df_stats = df_stats.set_index('epoch')
    return df_stats

def plotTrainStats(df_stats: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style = 'darkgrid')         # grid style
    sns.set(font_scale = 1.5)           # increase the plot size and font size
    plt.rcParams["figure.figsize"] = (12, 6)
    # plt learning curve
    plt.plot(df_stats['train_loss'], 'b-o', label = 'train_loss', linewidth = 2)
    plt.plot(df_stats['test_loss'], 'g-o', label = 'test_loss', linewidth = 2)
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1,2,3,4])
    plt.plot

if __name__ == "__main__":
    logger = getLogger(Config.log_path)
    data_train, data_test, cla2id, id2cla = load_data(Config.dataset_path)
    # 导入模型
    X_train, X_test, y_train, y_test = train_test_split(data_train['text'].tolist(), data_train['y'].tolist(), test_size = 0.2, random_state = 42)
    train_dataloader = DataLoader(TextDataset(X_train, labels = y_train), 
                            batch_size = Config.batch_size, collate_fn = collate_fn, shuffle = Config.shuffle)
    test_dataloader = DataLoader(TextDataset(X_test, labels = y_test), 
                            batch_size = Config.batch_size, collate_fn = collate_fn, shuffle = Config.shuffle)
    
    train_stats = train(cla2id, id2cla)
    print(train_stats)