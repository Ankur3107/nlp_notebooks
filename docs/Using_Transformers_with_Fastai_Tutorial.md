# Package Installation


```
!git clone https://github.com/devkosal/fastai_roberta.git
```

    Cloning into 'fastai_roberta'...
    remote: Enumerating objects: 171, done.[K
    remote: Counting objects: 100% (171/171), done.[K
    remote: Compressing objects: 100% (121/121), done.[K
    remote: Total 171 (delta 91), reused 111 (delta 44), pack-reused 0[K
    Receiving objects: 100% (171/171), 25.46 MiB | 18.58 MiB/s, done.
    Resolving deltas: 100% (91/91), done.



```
!pip install fastai==1.0.60 transformers==2.3.0
```

    Collecting fastai==1.0.60
    [?25l  Downloading https://files.pythonhosted.org/packages/f5/e4/a7025bf28f303dbda0f862c09a7f957476fa92c9271643b4061a81bb595f/fastai-1.0.60-py3-none-any.whl (237kB)
    [K     |████████████████████████████████| 245kB 4.7MB/s 
    [?25hCollecting transformers==2.3.0
    [?25l  Downloading https://files.pythonhosted.org/packages/50/10/aeefced99c8a59d828a92cc11d213e2743212d3641c87c82d61b035a7d5c/transformers-2.3.0-py3-none-any.whl (447kB)
    [K     |████████████████████████████████| 450kB 7.4MB/s 
    [?25hRequirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (3.2.2)
    Requirement already satisfied: bottleneck in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (1.3.2)
    Requirement already satisfied: numpy>=1.15 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (1.18.5)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (7.0.0)
    Requirement already satisfied: nvidia-ml-py3 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (7.352.0)
    Requirement already satisfied: dataclasses; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (0.7)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (2.23.0)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (3.13)
    Requirement already satisfied: numexpr in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (2.7.1)
    Requirement already satisfied: fastprogress>=0.2.1 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (1.0.0)
    Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (20.4)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (0.7.0+cu101)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (4.6.3)
    Requirement already satisfied: spacy>=2.0.18 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (2.2.4)
    Requirement already satisfied: torch>=1.0.0 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (1.6.0+cu101)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (1.4.1)
    Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.60) (1.0.5)
    Collecting sentencepiece
    [?25l  Downloading https://files.pythonhosted.org/packages/d4/a4/d0a884c4300004a78cca907a6ff9a5e9fe4f090f5d95ab341c53d28cbc58/sentencepiece-0.1.91-cp36-cp36m-manylinux1_x86_64.whl (1.1MB)
    [K     |████████████████████████████████| 1.1MB 7.0MB/s 
    [?25hRequirement already satisfied: boto3 in /usr/local/lib/python3.6/dist-packages (from transformers==2.3.0) (1.14.48)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from transformers==2.3.0) (4.41.1)
    Collecting sacremoses
    [?25l  Downloading https://files.pythonhosted.org/packages/7d/34/09d19aff26edcc8eb2a01bed8e98f13a1537005d31e95233fd48216eed10/sacremoses-0.0.43.tar.gz (883kB)
    [K     |████████████████████████████████| 890kB 26.1MB/s 
    [?25hRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.6/dist-packages (from transformers==2.3.0) (2019.12.20)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->fastai==1.0.60) (2.4.7)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->fastai==1.0.60) (2.8.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->fastai==1.0.60) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->fastai==1.0.60) (1.2.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->fastai==1.0.60) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->fastai==1.0.60) (2020.6.20)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->fastai==1.0.60) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->fastai==1.0.60) (2.10)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from packaging->fastai==1.0.60) (1.15.0)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.60) (1.0.2)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.60) (0.7.1)
    Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.60) (0.4.1)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.60) (1.1.3)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.60) (3.0.2)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.60) (1.0.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.60) (49.6.0)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.60) (2.0.3)
    Requirement already satisfied: thinc==7.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.60) (7.4.0)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.60) (1.0.2)
    Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from torch>=1.0.0->fastai==1.0.60) (0.16.0)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas->fastai==1.0.60) (2018.9)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.6/dist-packages (from boto3->transformers==2.3.0) (0.10.0)
    Requirement already satisfied: s3transfer<0.4.0,>=0.3.0 in /usr/local/lib/python3.6/dist-packages (from boto3->transformers==2.3.0) (0.3.3)
    Requirement already satisfied: botocore<1.18.0,>=1.17.48 in /usr/local/lib/python3.6/dist-packages (from boto3->transformers==2.3.0) (1.17.48)
    Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers==2.3.0) (7.1.2)
    Requirement already satisfied: joblib in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers==2.3.0) (0.16.0)
    Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.0.18->fastai==1.0.60) (1.7.0)
    Requirement already satisfied: docutils<0.16,>=0.10 in /usr/local/lib/python3.6/dist-packages (from botocore<1.18.0,>=1.17.48->boto3->transformers==2.3.0) (0.15.2)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.0.18->fastai==1.0.60) (3.1.0)
    Building wheels for collected packages: sacremoses
      Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
      Created wheel for sacremoses: filename=sacremoses-0.0.43-cp36-none-any.whl size=893257 sha256=aef96c81bb474827b2b34caa9690e68458a664de90731bcc580754ce52b74dd7
      Stored in directory: /root/.cache/pip/wheels/29/3c/fd/7ce5c3f0666dab31a50123635e6fb5e19ceb42ce38d4e58f45
    Successfully built sacremoses
    Installing collected packages: fastai, sentencepiece, sacremoses, transformers
      Found existing installation: fastai 1.0.61
        Uninstalling fastai-1.0.61:
          Successfully uninstalled fastai-1.0.61
    Successfully installed fastai-1.0.60 sacremoses-0.0.43 sentencepiece-0.1.91 transformers-2.3.0


# Load And Set Configuration


```
from fastai.text import *
from fastai.metrics import *
from transformers import RobertaTokenizer
```


```
# Creating a config object to store task specific information
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        
config = Config(
    testing=True,
    seed = 2019,
    roberta_model_name='roberta-base', # can also be exchnaged with roberta-large 
    max_lr=1e-5,
    epochs=1,
    use_fp16=False,
    bs=4, 
    max_seq_len=256, 
    num_labels = 2,
    hidden_dropout_prob=.05,
    hidden_size=768, # 1024 for roberta-large
    start_tok = "<s>",
    end_tok = "</s>",
)
```


```
df = pd.read_csv("fastai_roberta/fastai_roberta_imdb/imdb_dataset.csv")
```


```
if config.testing: df = df[:5000]
print(df.shape)
```

    (5000, 2)



```
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I thought this was a wonderful way to spend ti...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basically there's a family where a little boy ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Petter Mattei's "Love in the Time of Money" is...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```
feat_cols = "review"
label_cols = "sentiment"
```

# Setting Up the Tokenizer


```
class FastAiRobertaTokenizer(BaseTokenizer):
    """Wrapper around RobertaTokenizer to be compatible with fastai"""
    def __init__(self, tokenizer: RobertaTokenizer, max_seq_len: int=128, **kwargs): 
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len 
    def __call__(self, *args, **kwargs): 
        return self 
    def tokenizer(self, t:str) -> List[str]: 
        """Adds Roberta bos and eos tokens and limits the maximum sequence length""" 
        return [config.start_tok] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + [config.end_tok]
```


```
# create fastai tokenizer for roberta
roberta_tok = RobertaTokenizer.from_pretrained("roberta-base")

fastai_tokenizer = Tokenizer(tok_func=FastAiRobertaTokenizer(roberta_tok, max_seq_len=config.max_seq_len), 
                             pre_rules=[], post_rules=[])
```


```
# create fastai vocabulary for roberta
path = Path()
roberta_tok.save_vocabulary(path)

with open('vocab.json', 'r') as f:
    roberta_vocab_dict = json.load(f)
    
fastai_roberta_vocab = Vocab(list(roberta_vocab_dict.keys()))
```


```
# Setting up pre-processors
class RobertaTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
         super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)

class RobertaNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_roberta_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):
    """
    Constructing preprocessors for Roberta
    We remove sos and eos tokens since we add that ourselves in the tokenizer.
    We also use a custom vocabulary to match the numericalization with the original Roberta model.
    """
    return [RobertaTokenizeProcessor(tokenizer=tokenizer), RobertaNumericalizeProcessor(vocab=vocab)]
```

# Setting up the DataBunch


```
# Creating a Roberta specific DataBunch class
class RobertaDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training Roberta"
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=64, val_bs:int=None, pad_idx=1,
               pad_first=True, device:torch.device=None, no_check:bool=False, backwards:bool=False, 
               dl_tfms:Optional[Collection[Callable]]=None, **dl_kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)
```


```
class RobertaTextList(TextList):
    _bunch = RobertaDataBunch
    _label_cls = TextList
```


```
# loading the tokenizer and vocab processors
processor = get_roberta_processor(tokenizer=fastai_tokenizer, vocab=fastai_roberta_vocab)
```


```
# creating our databunch 
data = RobertaTextList.from_df(df, ".", cols=feat_cols, processor=processor) \
    .split_by_rand_pct(seed=config.seed) \
    .label_from_df(cols=label_cols,label_cls=CategoryList) \
    .databunch(bs=config.bs, pad_first=False, pad_idx=0)
```










```
data
```




    RobertaDataBunch;
    
    Train: LabelList (4000 items)
    x: RobertaTextList
    <s> One Ġof Ġthe Ġother Ġreviewers Ġhas Ġmentioned Ġthat Ġafter Ġwatching Ġjust Ġ1 ĠOz Ġepisode Ġyou 'll Ġbe Ġhooked . ĠThey Ġare Ġright , Ġas Ġthis Ġis Ġexactly Ġwhat Ġhappened Ġwith Ġme .< br Ġ/ >< br Ġ/> The Ġfirst Ġthing Ġthat Ġstruck Ġme Ġabout ĠOz Ġwas Ġits Ġbrutality Ġand Ġunfl inch ing Ġscenes Ġof Ġviolence , Ġwhich Ġset Ġin Ġright Ġfrom Ġthe Ġword ĠGO . ĠTrust Ġme , Ġthis Ġis Ġnot Ġa Ġshow Ġfor Ġthe Ġfaint Ġheart ed Ġor Ġtimid . ĠThis Ġshow Ġpulls Ġno Ġpunches Ġwith Ġregards Ġto Ġdrugs , Ġsex Ġor Ġviolence . ĠIts Ġis Ġhardcore , Ġin Ġthe Ġclassic Ġuse Ġof Ġthe Ġword .< br Ġ/ >< br Ġ/> It Ġis Ġcalled ĠO Z Ġas Ġthat Ġis Ġthe Ġnickname Ġgiven Ġto Ġthe ĠOswald ĠMaximum ĠSecurity ĠState ĠPen itent ary . ĠIt Ġfocuses Ġmainly Ġon ĠEmerald ĠCity , Ġan Ġexperimental Ġsection Ġof Ġthe Ġprison Ġwhere Ġall Ġthe Ġcells Ġhave Ġglass Ġfronts Ġand Ġface Ġin wards , Ġso Ġprivacy Ġis Ġnot Ġhigh Ġon Ġthe Ġagenda . ĠEm ĠCity Ġis Ġhome Ġto Ġmany .. A ry ans , ĠMuslims , Ġgang st as , ĠLatinos , ĠChristians , ĠItalians , ĠIrish Ġand Ġmore .... so Ġsc uff les , Ġdeath Ġstares , Ġdod gy Ġdealings Ġand Ġshady Ġagreements Ġare Ġnever Ġfar Ġaway .< br Ġ/ >< br Ġ/> I Ġwould Ġsay Ġthe Ġmain Ġappeal Ġof Ġthe Ġshow Ġis Ġdue Ġto Ġthe Ġfact Ġthat Ġit Ġgoes Ġwhere Ġother Ġshows Ġwouldn 't Ġdare . ĠForget Ġpretty Ġpictures Ġpainted Ġfor Ġmainstream Ġaudiences , Ġforget Ġcharm , Ġforget </s>,<s> A Ġwonderful Ġlittle Ġproduction . Ġ< br Ġ/ >< br Ġ/> The Ġfilming Ġtechnique Ġis Ġvery Ġun assuming - Ġvery Ġold - time - BBC Ġfashion Ġand Ġgives Ġa Ġcomforting , Ġand Ġsometimes Ġdiscomfort ing , Ġsense Ġof Ġrealism Ġto Ġthe Ġentire Ġpiece . Ġ< br Ġ/ >< br Ġ/> The Ġactors Ġare Ġextremely Ġwell Ġchosen - ĠMichael ĠSheen Ġnot Ġonly Ġ" has Ġgot Ġall Ġthe Ġpol ari " Ġbut Ġhe Ġhas Ġall Ġthe Ġvoices Ġdown Ġpat Ġtoo ! ĠYou Ġcan Ġtruly Ġsee Ġthe Ġseamless Ġediting Ġguided Ġby Ġthe Ġreferences Ġto ĠWilliams ' Ġdiary Ġentries , Ġnot Ġonly Ġis Ġit Ġwell Ġworth Ġthe Ġwatching Ġbut Ġit Ġis Ġa Ġterrific ly Ġwritten Ġand Ġperformed Ġpiece . ĠA Ġmaster ful Ġproduction Ġabout Ġone Ġof Ġthe Ġgreat Ġmaster 's Ġof Ġcomedy Ġand Ġhis Ġlife . Ġ< br Ġ/ >< br Ġ/> The Ġrealism Ġreally Ġcomes Ġhome Ġwith Ġthe Ġlittle Ġthings : Ġthe Ġfantasy Ġof Ġthe Ġguard Ġwhich , Ġrather Ġthan Ġuse Ġthe Ġtraditional Ġ' dream ' Ġtechniques Ġremains Ġsolid Ġthen Ġdisappears . ĠIt Ġplays Ġon Ġour Ġknowledge Ġand Ġour Ġsenses , Ġparticularly Ġwith Ġthe Ġscenes Ġconcerning ĠOr ton Ġand ĠHall i well Ġand Ġthe Ġsets Ġ( particularly Ġof Ġtheir Ġflat Ġwith ĠHall i well 's Ġmur als Ġdecor ating Ġevery Ġsurface ) Ġare Ġterribly Ġwell Ġdone . </s>,<s> Basically Ġthere 's Ġa Ġfamily Ġwhere Ġa Ġlittle Ġboy Ġ( Jake ) Ġthinks Ġthere 's Ġa Ġzombie Ġin Ġhis Ġcloset Ġ& Ġhis Ġparents Ġare Ġfighting Ġall Ġthe Ġtime .< br Ġ/ >< br Ġ/> This Ġmovie Ġis Ġslower Ġthan Ġa Ġsoap Ġopera ... Ġand Ġsuddenly , ĠJake Ġdecides Ġto Ġbecome ĠRam bo Ġand Ġkill Ġthe Ġzombie .< br Ġ/ >< br Ġ/> OK , Ġfirst Ġof Ġall Ġwhen Ġyou 're Ġgoing Ġto Ġmake Ġa Ġfilm Ġyou Ġmust ĠDec ide Ġif Ġits Ġa Ġthriller Ġor Ġa Ġdrama ! ĠAs Ġa Ġdrama Ġthe Ġmovie Ġis Ġwatch able . ĠParents Ġare Ġdivor cing Ġ& Ġarguing Ġlike Ġin Ġreal Ġlife . ĠAnd Ġthen Ġwe Ġhave ĠJake Ġwith Ġhis Ġcloset Ġwhich Ġtotally Ġruins Ġall Ġthe Ġfilm ! ĠI Ġexpected Ġto Ġsee Ġa ĠB OO GE Y MAN Ġsimilar Ġmovie , Ġand Ġinstead Ġi Ġwatched Ġa Ġdrama Ġwith Ġsome Ġmeaningless Ġthriller Ġspots .< br Ġ/ >< br Ġ/> 3 Ġout Ġof Ġ10 Ġjust Ġfor Ġthe Ġwell Ġplaying Ġparents Ġ& Ġdescent Ġdialog s . ĠAs Ġfor Ġthe Ġshots Ġwith ĠJake : Ġjust Ġignore Ġthem . </s>,<s> Pet ter ĠMatte i 's Ġ" Love Ġin Ġthe ĠTime Ġof ĠMoney " Ġis Ġa Ġvisually Ġstunning Ġfilm Ġto Ġwatch . ĠMr . ĠMatte i Ġoffers Ġus Ġa Ġvivid Ġportrait Ġabout Ġhuman Ġrelations . ĠThis Ġis Ġa Ġmovie Ġthat Ġseems Ġto Ġbe Ġtelling Ġus Ġwhat Ġmoney , Ġpower Ġand Ġsuccess Ġdo Ġto Ġpeople Ġin Ġthe Ġdifferent Ġsituations Ġwe Ġencounter . Ġ< br Ġ/ >< br Ġ/> This Ġbeing Ġa Ġvariation Ġon Ġthe ĠArthur ĠSchn itz ler 's Ġplay Ġabout Ġthe Ġsame Ġtheme , Ġthe Ġdirector Ġtransfers Ġthe Ġaction Ġto Ġthe Ġpresent Ġtime ĠNew ĠYork Ġwhere Ġall Ġthese Ġdifferent Ġcharacters Ġmeet Ġand Ġconnect . ĠEach Ġone Ġis Ġconnected Ġin Ġone Ġway , Ġor Ġanother Ġto Ġthe Ġnext Ġperson , Ġbut Ġno Ġone Ġseems Ġto Ġknow Ġthe Ġprevious Ġpoint Ġof Ġcontact . ĠSty lish ly , Ġthe Ġfilm Ġhas Ġa Ġsophisticated Ġluxurious Ġlook . ĠWe Ġare Ġtaken Ġto Ġsee Ġhow Ġthese Ġpeople Ġlive Ġand Ġthe Ġworld Ġthey Ġlive Ġin Ġtheir Ġown Ġhabitat .< br Ġ/ >< br Ġ/> The Ġonly Ġthing Ġone Ġgets Ġout Ġof Ġall Ġthese Ġsouls Ġin Ġthe Ġpicture Ġis Ġthe Ġdifferent Ġstages Ġof Ġloneliness Ġeach Ġone Ġinhab its . ĠA Ġbig Ġcity Ġis Ġnot Ġexactly Ġthe Ġbest Ġplace Ġin Ġwhich Ġhuman Ġrelations Ġfind Ġsincere Ġfulfillment , Ġas Ġone Ġdiscern s Ġis Ġthe Ġcase Ġwith Ġmost Ġof Ġthe Ġpeople Ġwe Ġencounter .< br Ġ/ >< br Ġ/> The Ġacting Ġis Ġgood Ġunder ĠMr . ĠMatte i 's Ġdirection . ĠSteve ĠBus ce mi , ĠRos ario ĠDawson , ĠCarol ĠKane , ĠMichael ĠImper iol </s>,<s> Probably Ġmy Ġall - time Ġfavorite Ġmovie , Ġa Ġstory Ġof Ġself lessness , Ġsacrifice Ġand Ġdedication Ġto Ġa Ġnoble Ġcause , Ġbut Ġit 's Ġnot Ġpreach y Ġor Ġboring . ĠIt Ġjust Ġnever Ġgets Ġold , Ġdespite Ġmy Ġhaving Ġseen Ġit Ġsome Ġ15 Ġor Ġmore Ġtimes Ġin Ġthe Ġlast Ġ25 Ġyears . ĠPaul ĠLuk as ' Ġperformance Ġbrings Ġtears Ġto Ġmy Ġeyes , Ġand ĠBet te ĠDavis , Ġin Ġone Ġof Ġher Ġvery Ġfew Ġtruly Ġsympathetic Ġroles , Ġis Ġa Ġdelight . ĠThe Ġkids Ġare , Ġas Ġgrandma Ġsays , Ġmore Ġlike Ġ" d ressed - up Ġmid gets " Ġthan Ġchildren , Ġbut Ġthat Ġonly Ġmakes Ġthem Ġmore Ġfun Ġto Ġwatch . ĠAnd Ġthe Ġmother 's Ġslow Ġawakening Ġto Ġwhat 's Ġhappening Ġin Ġthe Ġworld Ġand Ġunder Ġher Ġown Ġroof Ġis Ġbelievable Ġand Ġstartling . ĠIf ĠI Ġhad Ġa Ġdozen Ġthumbs , Ġthey 'd Ġall Ġbe Ġ" up " Ġfor Ġthis Ġmovie . </s>
    y: CategoryList
    positive,positive,negative,positive,positive
    Path: .;
    
    Valid: LabelList (1000 items)
    x: RobertaTextList
    <s> Apparently , ĠThe ĠMut ilation ĠMan Ġis Ġabout Ġa Ġguy Ġwho Ġwand ers Ġthe Ġland Ġperforming Ġshows Ġof Ġself - mut ilation Ġas Ġa Ġway Ġof Ġcoping Ġwith Ġhis Ġabusive Ġchildhood . ĠI Ġuse Ġthe Ġword Ġ' app arently ' Ġbecause Ġwithout Ġlistening Ġto Ġa Ġdirector ĠAndy ĠCo pp 's Ġcommentary Ġ( which ĠI Ġdidn 't Ġhave Ġavailable Ġto Ġme ) Ġor Ġreading Ġup Ġon Ġthe Ġfilm Ġprior Ġto Ġwatching , Ġviewers Ġwon 't Ġhave Ġa Ġclue Ġwhat Ġit Ġis Ġabout .< br Ġ/ >< br Ġ/> G ore h ounds Ġand Ġfans Ġof Ġextreme Ġmovies Ġmay Ġbe Ġlured Ġinto Ġwatching ĠThe ĠMut ilation ĠMan Ġwith Ġthe Ġpromise Ġof Ġsome Ġharsh Ġscenes Ġof Ġspl atter Ġand Ġunsettling Ġreal - life Ġfootage , Ġbut Ġunless Ġthey 're Ġalso Ġfond Ġof Ġpret entious , Ġheadache - inducing , Ġexperimental Ġart - house Ġcinema , Ġthey 'll Ġfind Ġthis Ġone Ġa Ġreal Ġchore Ġto Ġsit Ġthrough .< br Ġ/ >< br Ġ/> 82 Ġminutes Ġof Ġugly Ġimagery Ġaccompanied Ġby Ġdis - ch ord ant Ġsound , Ġterrible Ġmusic Ġand Ġincomprehensible Ġdialogue , Ġthis Ġmind - n umb ingly Ġawful Ġdri vel Ġis Ġthe Ġperfect Ġway Ġto Ġtest Ġone 's Ġsanity : Ġif Ġyou 've Ġstill Ġgot Ġall Ġyour Ġmar bles , Ġyou 'll Ġswitch Ġthis Ġrubbish Ġoff Ġand Ġwatch Ġsomething Ġdecent Ġinstead Ġ( I Ġwatched Ġthe Ġwhole Ġthing , Ġbut Ġam Ġwell Ġaware Ġthat ĠI 'm Ġcompletely Ġbarking !). </s>,<s> Peter ĠC ushing Ġand ĠDonald ĠPle as ance Ġare Ġlegendary Ġactors , Ġand Ġdirector ĠK ost as ĠKar ag ian nis Ġwas Ġthe Ġman Ġbehind Ġthe Ġsuccessful ĠGreek ĠGi allo - es quire Ġthriller ĠDeath ĠKiss Ġin Ġ1974 ; Ġand Ġyet Ġwhen Ġyou Ġcombine Ġthe Ġthree Ġtalents , Ġall Ġyou Ġget Ġis Ġthis Ġcomplete Ġload Ġof Ġdri vel ! ĠGod Ġonly Ġknows Ġwhat Ġdrove Ġthe Ġlikes Ġof ĠPeter ĠC ushing Ġand ĠDonald ĠPle as ance Ġto Ġstar Ġin Ġthis Ġcheap ie Ġdevil Ġworship Ġflick , Ġbut ĠI Ġreally Ġdo Ġhope Ġthey Ġwere Ġwell Ġpaid Ġas Ġneither Ġone Ġdeserves Ġsomething Ġas Ġamateur ish Ġas Ġthis Ġon Ġtheir Ġresumes . ĠThe Ġstory Ġfocuses Ġon Ġa Ġgroup Ġof Ġdevil Ġworsh ippers Ġthat Ġkidnap Ġsome Ġkids , Ġleading Ġanother Ġgroup Ġto Ġgo Ġafter Ġthem . ĠThe Ġpace Ġof Ġthe Ġplot Ġis Ġvery Ġslow Ġand Ġthis Ġensures Ġthat Ġthe Ġfilm Ġis Ġvery Ġboring . ĠThe Ġplot Ġis Ġalso Ġa Ġlong Ġway Ġfrom Ġbeing Ġoriginal Ġand Ġanyone Ġwith Ġeven Ġa Ġpassing Ġinterest Ġin Ġthe Ġhorror Ġgenre Ġwill Ġhave Ġseen Ġsomething Ġa Ġbit Ġlike Ġthis , Ġand Ġno Ġdoubt Ġdone Ġmuch Ġbetter . ĠThe Ġobvious Ġlack Ġof Ġbudget Ġis Ġfelt Ġthroughout Ġand Ġthe Ġfilm Ġdoesn 't Ġmanage Ġto Ġovercome Ġthis Ġat Ġany Ġpoint . ĠThis Ġreally Ġis Ġa Ġdepressing Ġand Ġmiserable Ġwatch Ġand Ġnot Ġeven Ġa Ġslightly Ġdecent Ġending Ġmanages Ġto Ġup Ġthe Ġante Ġenough Ġto Ġlift Ġthis Ġfilm Ġout Ġof Ġthe Ġvery Ġbottom Ġof Ġthe Ġbarrel . ĠExtreme ly Ġpoor Ġstuff Ġand Ġdefinitely Ġnot Ġrecommended ! </s>,<s> Back Ġin Ġthe Ġ1970 s , ĠWP IX Ġran Ġ" The ĠAdventures Ġof ĠSuperman " Ġevery Ġweekday Ġafternoon Ġfor Ġquite Ġa Ġfew Ġyears . ĠEvery Ġonce Ġin Ġa Ġwhile , Ġwe 'd Ġget Ġa Ġtreat Ġwhen Ġthey Ġwould Ġpreempt Ġneighboring Ġshows Ġto Ġair Ġ" Super man Ġand Ġthe ĠMole ĠMen ." ĠI Ġalways Ġlooked Ġforward Ġto Ġthose Ġdays . ĠWatching Ġit Ġrecently , ĠI Ġwas Ġsurprised Ġat Ġjust Ġhow Ġbad Ġit Ġreally Ġwas .< br Ġ/ >< br Ġ/> It Ġwasn 't Ġbad Ġbecause Ġof Ġthe Ġspecial Ġeffects , Ġor Ġlack Ġthereof . ĠTrue , ĠGeorge ĠReeves ' ĠSuperman Ġcostume Ġwas Ġpretty Ġbad , Ġthe Ġedges Ġof Ġthe Ġfoam Ġpadding Ġused Ġto Ġmake Ġhim Ġlook Ġmore Ġimposing Ġbeing Ġplainly Ġvisible . ĠAnd Ġtrue , Ġthe ĠMole ĠMen 's Ġcostumes Ġwere Ġeven Ġworse . ĠWhat Ġwas Ġsupposed Ġto Ġbe Ġa Ġfurry Ġcovering Ġwouldn 't Ġhave Ġfooled Ġa Ġten Ġyear - old , Ġsince Ġthe Ġz ippers , Ġsleeve Ġhe ms Ġand Ġbadly Ġp illing Ġfabric Ġbadly Ġtailored Ġinto Ġbag gy Ġcostumes Ġwere Ġall Ġpainfully Ġobvious . ĠBut Ġthese Ġwere Ġforg ivable Ġshortcomings .< br Ġ/ >< br Ġ/> No , Ġwhat Ġmade Ġit Ġbad Ġwere Ġthe Ġcont rived Ġplot Ġdevices . ĠTime Ġand Ġagain , ĠSuperman Ġfailed Ġto Ġdo Ġanything Ġto Ġkeep Ġthe Ġsituation Ġfrom Ġdeteriorating . ĠA Ġlyn ch Ġmob Ġis Ġsearching Ġfor Ġthe Ġcreatures ? ĠRather Ġthan Ġround Ġup Ġthe Ġhysterical Ġcrowd Ġor Ġsearch Ġfor Ġthe Ġcreatures Ġhimself , Ġhe Ġstands Ġaround Ġexplaining Ġthe Ġdangers Ġof Ġthe Ġsituation Ġto ĠLois Ġand Ġthe ĠPR </s>,<s> Sat an 's ĠLittle ĠHel per Ġis Ġone Ġof Ġthe Ġbetter ĠB ĠHorror Ġmovies ĠI Ġhave Ġseen . ĠWhen ĠI Ġsay Ġbetter ĠI Ġmean Ġthe Ġstory . ĠThe Ġfilm Ġhat ches Ġa Ġnew Ġplot , Ġsomething Ġthat 's Ġnot Ġso ĠclichÃ© Ġin Ġthe ĠHorror Ġgenre Ġ- Ġsomething Ġfresh . ĠBut Ġthere Ġare Ġalso Ġsome Ġridiculous Ġquestions Ġthat Ġcome Ġalong Ġwith Ġit . ĠQuestions Ġyou Ġwill Ġbe Ġasking Ġyourself Ġthroughout Ġthe Ġmovie .< br Ġ/ >< br Ġ/> The Ġfilm Ġfirst Ġcaught Ġmy Ġattention Ġwhile ĠI Ġwas Ġcruising Ġthe ĠHorror Ġsection Ġin ĠHM V . ĠI Ġwas Ġtired Ġof Ġall Ġthe Ġso Ġcalled Ġ" ter r ifi ying " ĠHollywood Ġblock busters Ġand Ġwanted Ġsomething Ġdifferent . ĠThe Ġcover Ġart Ġfor ĠSatan 's ĠLittle ĠHel per Ġimmediately Ġcaught Ġmy Ġattention . ĠAs Ġyou Ġcan Ġsee , Ġthe Ġimage Ġdraws Ġyou Ġin Ġ- Ġit 's Ġchilling ! ĠI Ġknew Ġit Ġwas Ġa Ġstraight Ġto ĠDVD Ġrelease Ġ- Ġbut ĠI Ġtook Ġa Ġchance . ĠI Ġmean , ĠI Ġjust Ġseen Ġ" Boo gey ĠMan " Ġthe Ġnight Ġbefore Ġ- Ġso ĠIt Ġcouldn 't Ġget Ġany Ġworse ! ĠAfter ĠI Ġwatched Ġthe Ġmovie , ĠI Ġwas Ġsemi - s atisf ied . ĠI Ġloved Ġthe Ġplot Ġof Ġthe Ġmovie . ĠIt Ġwas Ġreally Ġcreepy Ġhow Ġthe Ġkiller Ġwas Ġpretending Ġto Ġbe Ġthe Ġlittle Ġboys Ġfriend , Ġso Ġhe Ġcould Ġkill . ĠIn Ġsome Ġsick Ġder anged Ġway , Ġhe Ġactually Ġthought Ġhe Ġand Ġthe Ġlittle Ġboy Ġwould Ġbecome Ġpartners Ġ- Ġa Ġduo Ġof Ġterror . ĠIt Ġwas Ġa </s>,<s> I Ġsaw Ġthis Ġgem Ġof Ġa Ġfilm Ġat ĠCannes Ġwhere Ġit Ġwas Ġpart Ġof Ġthe Ġdirectors Ġfortnight .< br Ġ/ >< br Ġ/> Welcome Ġto ĠColl in wood Ġis Ġnothing Ġshort Ġof Ġsuperb . ĠGreat Ġfun Ġthroughout , Ġwith Ġall Ġmembers Ġof Ġa Ġstrong Ġcast Ġacting Ġtheir Ġsocks Ġoff . ĠIt 's Ġa Ġsometimes Ġlaugh Ġout Ġloud Ġcomedy Ġabout Ġa Ġpetty Ġcro ok Ġ( Cos imo , Ġplayed Ġby ĠLuis ĠGu zman ) Ġwho Ġgets Ġcaught Ġtrying Ġto Ġsteal Ġa Ġcar Ġand Ġsent Ġto Ġprison . ĠWhile Ġin Ġprison Ġhe Ġmeets Ġa Ġ` l ifer ' Ġwho Ġtells Ġhim Ġof Ġ` the Ġultimate Ġbell ini ' ĠÂ ĸ Ġwhich Ġto Ġyou Ġand Ġme ĠÂ ĸ Ġis Ġa Ġsure - fire Ġget Ġrich Ġquick Ġscheme . ĠIt Ġturns Ġout Ġthat Ġthere Ġis Ġa Ġway Ġthrough Ġfrom Ġa Ġdeserted Ġbuilding Ġinto Ġthe Ġtowns Ġjew ell ers Ġshop ĠÂ ĸ Ġwhich Ġcould Ġnet Ġmillions . ĠSounds Ġsimple ? ĠÂ ĸ Ġwell Ġthrow Ġin Ġall Ġkinds Ġof Ġw acky Ġcharacters Ġand Ġincidents Ġalong Ġthe Ġway Ġand Ġyou Ġhave Ġgot Ġthe Ġingredients Ġfor Ġa Ġone Ġwild Ġride !! ĠÂ ĸ Ġword Ġpasses Ġfrom Ġone Ġlow Ġlife Ġloser Ġto Ġthe Ġnext Ġand Ġsoon Ġa Ġteam Ġof Ġthem Ġare Ġassembled Ġto Ġtry Ġand Ġcash Ġin Ġon ĠCos im os Ġ` bell ini ' Ġlead Ġby Ġfailed Ġboxer ĠPer o Ġ( Super bly Ġplayed Ġby ĠSam ĠRock well ĠÂ ĸ Ġsurely Ġa Ġstar Ġin Ġthe Ġmaking ) Ġand Ġreluctant Ġcro ok ĠRiley Ġ( William ĠH . ĠMacy ) Ġwho Ġis Ġforced Ġto </s>
    y: CategoryList
    negative,negative,negative,negative,positive
    Path: .;
    
    Test: None



# Building the Model


```
import torch
import torch.nn as nn
from transformers import RobertaModel

# defining our model architecture 
class CustomRobertaModel(nn.Module):
    def __init__(self,num_labels=2):
        super(CustomRobertaModel,self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(config.roberta_model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels) # defining final output layer
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _ , pooled_output = self.roberta(input_ids, token_type_ids, attention_mask) # 
        logits = self.classifier(pooled_output)        
        return logits
```


```
roberta_model = CustomRobertaModel(num_labels=config.num_labels)

learn = Learner(data, roberta_model, metrics=[accuracy])
```


```
learn.model.roberta.train() # setting roberta to train as it is in eval mode by default
learn.fit_one_cycle(config.epochs, max_lr=config.max_lr)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.184614</td>
      <td>0.180552</td>
      <td>0.929000</td>
      <td>02:17</td>
    </tr>
  </tbody>
</table>


# Getting Predictions


```
def get_preds_as_nparray(ds_type) -> np.ndarray:
    learn.model.roberta.eval()
    preds = learn.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in data.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    ordered_preds = preds[reverse_sampler, :]
    pred_values = np.argmax(ordered_preds, axis=1)
    return ordered_preds, pred_values
```


```
preds, pred_values = get_preds_as_nparray(DatasetType.Valid)
```






```
# accuracy on valid
(pred_values == data.valid_ds.y.items).mean()
```




    0.929



# Saving/Loading the model weights


```
def save_model(learner, file_name):
    st = learner.model.state_dict()
    torch.save(st, file_name) # will save model in current dir # backend is pickle 

def load_model(learner, file_name):
    st = torch.load(file_name)
    learner.model.load_state_dict(st)
```


```
# monkey patching Learner methods to save and load model file
Learner.save_model = save_model
Learner.load_model = load_model
```


```
learn.save_model("my_model.bin")
learn.load_model("my_model.bin")
```
