<a href="https://colab.research.google.com/github/Ankur3107/colab_notebooks/blob/master/Generic_Transformer_Classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```
!pip install transformers
```

    Collecting transformers
    [?25l  Downloading https://files.pythonhosted.org/packages/27/3c/91ed8f5c4e7ef3227b4119200fc0ed4b4fd965b1f0172021c25701087825/transformers-3.0.2-py3-none-any.whl (769kB)
    [K     |████████████████████████████████| 778kB 2.8MB/s 
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from transformers) (1.18.5)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.6/dist-packages (from transformers) (4.41.1)
    Collecting sentencepiece!=0.1.92
    [?25l  Downloading https://files.pythonhosted.org/packages/d4/a4/d0a884c4300004a78cca907a6ff9a5e9fe4f090f5d95ab341c53d28cbc58/sentencepiece-0.1.91-cp36-cp36m-manylinux1_x86_64.whl (1.1MB)
    [K     |████████████████████████████████| 1.1MB 13.3MB/s 
    [?25hRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.6/dist-packages (from transformers) (2019.12.20)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from transformers) (2.23.0)
    Requirement already satisfied: dataclasses; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from transformers) (0.7)
    Collecting sacremoses
    [?25l  Downloading https://files.pythonhosted.org/packages/7d/34/09d19aff26edcc8eb2a01bed8e98f13a1537005d31e95233fd48216eed10/sacremoses-0.0.43.tar.gz (883kB)
    [K     |████████████████████████████████| 890kB 19.7MB/s 
    [?25hRequirement already satisfied: filelock in /usr/local/lib/python3.6/dist-packages (from transformers) (3.0.12)
    Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from transformers) (20.4)
    Collecting tokenizers==0.8.1.rc1
    [?25l  Downloading https://files.pythonhosted.org/packages/40/d0/30d5f8d221a0ed981a186c8eb986ce1c94e3a6e87f994eae9f4aa5250217/tokenizers-0.8.1rc1-cp36-cp36m-manylinux1_x86_64.whl (3.0MB)
    [K     |████████████████████████████████| 3.0MB 20.6MB/s 
    [?25hRequirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2020.6.20)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (1.15.0)
    Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (7.1.2)
    Requirement already satisfied: joblib in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (0.16.0)
    Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from packaging->transformers) (2.4.7)
    Building wheels for collected packages: sacremoses
      Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
      Created wheel for sacremoses: filename=sacremoses-0.0.43-cp36-none-any.whl size=893257 sha256=6bda503b3bbdbf7626ff38a3e3235c3a20e948d97c40c78a681f5c87b90ed237
      Stored in directory: /root/.cache/pip/wheels/29/3c/fd/7ce5c3f0666dab31a50123635e6fb5e19ceb42ce38d4e58f45
    Successfully built sacremoses
    Installing collected packages: sentencepiece, sacremoses, tokenizers, transformers
    Successfully installed sacremoses-0.0.43 sentencepiece-0.1.91 tokenizers-0.8.1rc1 transformers-3.0.2



```
import os, pandas as pd
from sklearn.model_selection import train_test_split
import logging
from transformers import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm.autonotebook import tqdm
```

# 1. Set Configuration


```
class Config:
  train_file = './data.csv'
  eval_file = './eval.csv'
  max_seq_len = 128
  batch_size = 32
  epochs = 5
  model_name = 'bert-base-uncased'
  learning_rate = 2e-5
  n_classes = 3
  device = 'cpu'
  


flags = Config
```

# 2. Build Dataset Pipeline


```
class TextLabelDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
  
    def __len__(self):
        return len(self.texts)
  
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
        )

        return {
          'texts': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size, is_prediction=False):

  if isinstance(df, str):
    df = pd.read_csv(df)
  else:
    pass

  if is_prediction:
    ds = TextLabelDataset(
        texts=df.text.to_numpy(),
        labels=np.array([-1]*len(df.text.values)),
        tokenizer=tokenizer,
        max_len=max_len
        )
  else:
    ds = TextLabelDataset(
        texts=df.text.to_numpy(),
        labels=df.labels.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
        )
```

# 3. Build Model 


```
class Classifier(nn.Module):

  def __init__(self, model_name, n_classes):
      super(Classifier, self).__init__()
      self.bert = AutoModel.from_pretrained(model_name)
      self.drop = nn.Dropout(p=0.3)
      self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
      _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      output = self.drop(pooled_output)
      return self.out(output)
```


```
class ClassificationModel:

  def __init__(self, flags):
    self.flags = flags
    self.tokenizer = BertTokenizer.from_pretrained(self.flags.model_name)
    self.model = Classifier(self.flags.model_name, self.flags.n_classes)
    self.model = self.model.to(self.flags.device)

  def train(self):

    train_data_loader = create_data_loader(self.flags.train_file, self.tokenizer, self.flags.max_seq_len, self.flags.batch_size)
    val_data_loader = create_data_loader(self.flags.eval_file, self.tokenizer, self.flags.max_seq_len, self.flags.batch_size)

    optimizer = AdamW(self.model.parameters(), lr=self.flags.learning_rate, correct_bias=False)
    total_steps = len(train_data_loader) * self.flags.epochs

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(self.flags.device)

    history = defaultdict(list)
    best_accuracy = 0

    if isinstance(self.flags.train_file, str):
      train_df = pd.read_csv(self.flags.train_file)

    if isinstance(self.flags.eval_file, str):
      eval_df = pd.read_csv(self.flags.eval_file)

    for epoch in range(self.flags.epochs):

      print(f'Epoch {epoch + 1}/{self.flags.epochs}')
      print('-' * 10)

      train_acc, train_loss = self.train_epoch(
        self.model,
        train_data_loader,    
        loss_fn, 
        optimizer, 
        self.flags.device, 
        scheduler, 
        len(train_df)
      )

      print(f'Train loss {train_loss} accuracy {train_acc}')

      val_acc, val_loss = self.eval_model(
        self.model,
        val_data_loader,
        loss_fn, 
        self.flags.device, 
        len(eval_df)
      )

      print(f'Val   loss {val_loss} accuracy {val_acc}')
      print()

      history['train_acc'].append(train_acc)
      history['train_loss'].append(train_loss)
      history['val_acc'].append(val_acc)
      history['val_loss'].append(val_loss)

      if val_acc > best_accuracy:
        torch.save(self.model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc


  def train_epoch(self, model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for bi, d in enumerate(tk0):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )

      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

  def eval_model(self, model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
      tk0 = tqdm(data_loader, total=len(data_loader), desc="Evaluating")
      for bi, d in enumerate(tk0):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


    
```

### Download Data and Preparation


```
!wget https://raw.githubusercontent.com/SrinidhiRaghavan/AI-Sentiment-Analysis-on-IMDB-Dataset/master/imdb_tr.csv
```

    --2020-08-26 15:57:14--  https://raw.githubusercontent.com/SrinidhiRaghavan/AI-Sentiment-Analysis-on-IMDB-Dataset/master/imdb_tr.csv
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 23677025 (23M) [text/plain]
    Saving to: ‘imdb_tr.csv’
    
    imdb_tr.csv         100%[===================>]  22.58M  39.9MB/s    in 0.6s    
    
    2020-08-26 15:57:16 (39.9 MB/s) - ‘imdb_tr.csv’ saved [23677025/23677025]
    



```
data = pd.read_csv('imdb_tr.csv', encoding = "ISO-8859-1")
```


```
data.columns = ['row_Number', 'text', 'labels']
```


```
train_data = data.sample(1000)
test_data = data.sample(100)

```


```
train_data.to_csv('data.csv', index=False)
test_data.to_csv('eval.csv', index=False)
```

# Training


```
from collections import defaultdict
import numpy as np
```


```
class Config:
  train_file = './data.csv'
  eval_file = './eval.csv'
  max_seq_len = 128
  batch_size = 32
  epochs = 5
  model_name = 'bert-base-uncased'
  learning_rate = 2e-5
  n_classes = 2
  device = 'cuda'
  
flags = Config
```


```
classification = ClassificationModel(flags)
```


```
classification.train()
```

    Epoch 1/5
    ----------



    HBox(children=(FloatProgress(value=0.0, description='Training', max=32.0, style=ProgressStyle(description_widt…


    
    Train loss 0.6540139000862837 accuracy 0.622



    HBox(children=(FloatProgress(value=0.0, description='Evaluating', max=4.0, style=ProgressStyle(description_wid…


    
    Val   loss 0.4141501262784004 accuracy 0.78
    
    Epoch 2/5
    ----------



    HBox(children=(FloatProgress(value=0.0, description='Training', max=32.0, style=ProgressStyle(description_widt…


    
    Train loss 0.3276493112789467 accuracy 0.864



    HBox(children=(FloatProgress(value=0.0, description='Evaluating', max=4.0, style=ProgressStyle(description_wid…


    
    Val   loss 0.3254726273007691 accuracy 0.87
    
    Epoch 3/5
    ----------



    HBox(children=(FloatProgress(value=0.0, description='Training', max=32.0, style=ProgressStyle(description_widt…


    
    Train loss 0.12970392164424993 accuracy 0.9530000000000001



    HBox(children=(FloatProgress(value=0.0, description='Evaluating', max=4.0, style=ProgressStyle(description_wid…


    
    Val   loss 0.4319960339926183 accuracy 0.8300000000000001
    
    Epoch 4/5
    ----------



    HBox(children=(FloatProgress(value=0.0, description='Training', max=32.0, style=ProgressStyle(description_widt…


    
    Train loss 0.0639086696319282 accuracy 0.982



    HBox(children=(FloatProgress(value=0.0, description='Evaluating', max=4.0, style=ProgressStyle(description_wid…


    
    Val   loss 0.5208611574489623 accuracy 0.8300000000000001
    
    Epoch 5/5
    ----------



    HBox(children=(FloatProgress(value=0.0, description='Training', max=32.0, style=ProgressStyle(description_widt…


    
    Train loss 0.01748604617023375 accuracy 0.996



    HBox(children=(FloatProgress(value=0.0, description='Evaluating', max=4.0, style=ProgressStyle(description_wid…


    
    Val   loss 0.4388579736405518 accuracy 0.9
    



```

```
