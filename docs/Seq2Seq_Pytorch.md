<a href="https://colab.research.google.com/github/Ankur3107/colab_notebooks/blob/master/Seq2Seq_Pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```
!pip install spacy --upgrade
```

    Collecting spacy
    [?25l  Downloading https://files.pythonhosted.org/packages/10/b5/c7a92c7ce5d4b353b70b4b5b4385687206c8b230ddfe08746ab0fd310a3a/spacy-2.3.2-cp36-cp36m-manylinux1_x86_64.whl (9.9MB)
    [K     |████████████████████████████████| 10.0MB 3.9MB/s 
    [?25hRequirement already satisfied, skipping upgrade: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (1.0.2)
    Requirement already satisfied, skipping upgrade: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (0.7.1)
    Requirement already satisfied, skipping upgrade: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.6/dist-packages (from spacy) (1.0.0)
    Requirement already satisfied, skipping upgrade: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy) (2.0.3)
    Requirement already satisfied, skipping upgrade: setuptools in /usr/local/lib/python3.6/dist-packages (from spacy) (49.6.0)
    Collecting thinc==7.4.1
    [?25l  Downloading https://files.pythonhosted.org/packages/10/ae/ef3ae5e93639c0ef8e3eb32e3c18341e511b3c515fcfc603f4b808087651/thinc-7.4.1-cp36-cp36m-manylinux1_x86_64.whl (2.1MB)
    [K     |████████████████████████████████| 2.1MB 18.3MB/s 
    [?25hRequirement already satisfied, skipping upgrade: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (2.23.0)
    Requirement already satisfied, skipping upgrade: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (4.41.1)
    Requirement already satisfied, skipping upgrade: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.6/dist-packages (from spacy) (1.1.3)
    Requirement already satisfied, skipping upgrade: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (0.4.1)
    Requirement already satisfied, skipping upgrade: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy) (1.0.2)
    Requirement already satisfied, skipping upgrade: numpy>=1.15.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (1.18.5)
    Requirement already satisfied, skipping upgrade: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy) (3.0.2)
    Requirement already satisfied, skipping upgrade: importlib-metadata>=0.20; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from catalogue<1.1.0,>=0.0.7->spacy) (1.7.0)
    Requirement already satisfied, skipping upgrade: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (3.0.4)
    Requirement already satisfied, skipping upgrade: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2.10)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (1.24.3)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2020.6.20)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy) (3.1.0)
    Installing collected packages: thinc, spacy
      Found existing installation: thinc 7.4.0
        Uninstalling thinc-7.4.0:
          Successfully uninstalled thinc-7.4.0
      Found existing installation: spacy 2.2.4
        Uninstalling spacy-2.2.4:
          Successfully uninstalled spacy-2.2.4
    Successfully installed spacy-2.3.2 thinc-7.4.1



```
!python -m spacy download en
!python -m spacy download de
!python -m spacy download hi
```

    Collecting en_core_web_sm==2.3.1
    [?25l  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0MB)
    [K     |████████████████████████████████| 12.1MB 803kB/s 
    [?25hRequirement already satisfied: spacy<2.4.0,>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from en_core_web_sm==2.3.1) (2.3.2)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
    Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
    Requirement already satisfied: thinc==7.4.1 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
    Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.18.5)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.41.1)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.23.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (49.6.0)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
    Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
    Building wheels for collected packages: en-core-web-sm
      Building wheel for en-core-web-sm (setup.py) ... [?25l[?25hdone
      Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-cp36-none-any.whl size=12047109 sha256=c566e8eddcd63bc8259a784a72fae15d14da6e23d183b7a16bf6032b9aeaeed2
      Stored in directory: /tmp/pip-ephem-wheel-cache-76hzqxdi/wheels/2b/3f/41/f0b92863355c3ba34bb32b37d8a0c662959da0058202094f46
    Successfully built en-core-web-sm
    Installing collected packages: en-core-web-sm
      Found existing installation: en-core-web-sm 2.2.5
        Uninstalling en-core-web-sm-2.2.5:
          Successfully uninstalled en-core-web-sm-2.2.5
    Successfully installed en-core-web-sm-2.3.1
    [38;5;2m✔ Download and installation successful[0m
    You can now load the model via spacy.load('en_core_web_sm')
    [38;5;2m✔ Linking successful[0m
    /usr/local/lib/python3.6/dist-packages/en_core_web_sm -->
    /usr/local/lib/python3.6/dist-packages/spacy/data/en
    You can now load the model via spacy.load('en')
    Collecting de_core_news_sm==2.3.0
    [?25l  Downloading https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-2.3.0/de_core_news_sm-2.3.0.tar.gz (14.9MB)
    [K     |████████████████████████████████| 14.9MB 833kB/s 
    [?25hRequirement already satisfied: spacy<2.4.0,>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from de_core_news_sm==2.3.0) (2.3.2)
    Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (1.18.5)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (4.41.1)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (2.23.0)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (1.0.0)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (1.0.2)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (1.1.3)
    Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (0.4.1)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (1.0.2)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (49.6.0)
    Requirement already satisfied: thinc==7.4.1 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (7.4.1)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (3.0.2)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (0.7.1)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (2.0.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (2020.6.20)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (1.24.3)
    Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (1.7.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->de_core_news_sm==2.3.0) (3.1.0)
    Building wheels for collected packages: de-core-news-sm
      Building wheel for de-core-news-sm (setup.py) ... [?25l[?25hdone
      Created wheel for de-core-news-sm: filename=de_core_news_sm-2.3.0-cp36-none-any.whl size=14907580 sha256=959d3b6df2936d8e86bbc601dbd80eddfb310d3af7aca8895529c319cb58b539
      Stored in directory: /tmp/pip-ephem-wheel-cache-59e5ledg/wheels/db/f3/1e/0df0f27eee12bd1aaa94bcfef11b01eca62f90b9b9a0ce08fd
    Successfully built de-core-news-sm
    Installing collected packages: de-core-news-sm
      Found existing installation: de-core-news-sm 2.2.5
        Uninstalling de-core-news-sm-2.2.5:
          Successfully uninstalled de-core-news-sm-2.2.5
    Successfully installed de-core-news-sm-2.3.0
    [38;5;2m✔ Download and installation successful[0m
    You can now load the model via spacy.load('de_core_news_sm')
    [38;5;2m✔ Linking successful[0m
    /usr/local/lib/python3.6/dist-packages/de_core_news_sm -->
    /usr/local/lib/python3.6/dist-packages/spacy/data/de
    You can now load the model via spacy.load('de')
    
    [38;5;1m✘ No compatible model found for 'hi' (spaCy v2.3.2).[0m
    



```
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from spacy.lang.hi import Hindi
```


```
spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")

spacy_hi = Hindi()
def tokenize_hi(text):
    return [tok.text for tok in spacy_hi.tokenizer(text)]

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)
```


```
class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

```


```
# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = False
save_model = True

# Training hyperparameters
num_epochs = 10000
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0
```


```
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

```


```
model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)
```


```
def translate_sentence(model, sentence, german, english, device, max_length=50):
    # Load german tokenizer
    spacy_ger = spacy.load("de")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "ein pferd geht unter einer brücke neben einem boot."
```


```
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target[:-1, :])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

# running on entire test data takes a while
score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score * 100:.2f}")
```

    [Epoch 0 / 10000]
    => Saving checkpoint
    Translated example sentence: 
     ['.', 'secures', 'secures', 'half', 'secures', 'secures', 'half', '.', 'secures', 'toddler', 'secures', 'secures', 'half', '.', '.', 'olympians', 'half', '.', 'olympians', 'helmet', '.', 'secures', 'toddler', 'secures', 'secures', 'toddler', 'secures', 'secures', '.', 'secures', 'secures', '.', 'secures', '.', 'secures', 'secures', 'secures', 'secures', 'half', 'half', '.', 'secures', 'toddler', 'secures', 'secures', 'secures', 'mosaic', 'secures', 'secures', 'toddler']
    [Epoch 1 / 10000]
    => Saving checkpoint
    Translated example sentence: 
     ['a', 'horse', 'walking', 'under', 'a', 'boat', 'next', 'to', 'a', 'boat', '.', '<eos>']
    [Epoch 2 / 10000]
    => Saving checkpoint
    Translated example sentence: 
     ['a', 'horse', 'is', 'walking', 'under', 'a', 'bridge', 'next', 'to', 'a', 'boat', '.', '<eos>']
    [Epoch 3 / 10000]
    => Saving checkpoint
    Translated example sentence: 
     ['a', 'horse', 'is', 'walking', 'under', 'a', 'bridge', 'next', 'to', 'a', 'boat', '.', '<eos>']
    [Epoch 4 / 10000]
    => Saving checkpoint
    Translated example sentence: 
     ['a', 'horse', 'walks', 'under', 'a', 'bridge', 'next', 'to', 'a', 'boat', '.', '<eos>']
    [Epoch 5 / 10000]
    => Saving checkpoint
    Translated example sentence: 
     ['a', 'horse', 'walks', 'under', 'a', 'bridge', 'next', 'to', 'a', 'boat', '.', '<eos>']
    [Epoch 6 / 10000]
    => Saving checkpoint
    Translated example sentence: 
     ['a', 'horse', 'walks', 'underneath', 'a', 'bridge', 'next', 'to', 'a', 'boat', '.', '<eos>']
    [Epoch 7 / 10000]
    => Saving checkpoint
    Translated example sentence: 
     ['a', 'horse', 'is', 'walking', 'under', 'a', 'bridge', 'beside', 'a', 'boat', '.', '<eos>']



```

```
