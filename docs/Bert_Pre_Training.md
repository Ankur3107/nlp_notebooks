<a href="https://colab.research.google.com/github/Ankur3107/colab_notebooks/blob/master/Bert_Pre_Training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```
!pip install transformers
```

    Collecting transformers
    [?25l  Downloading https://files.pythonhosted.org/packages/ae/05/c8c55b600308dc04e95100dc8ad8a244dd800fe75dfafcf1d6348c6f6209/transformers-3.1.0-py3-none-any.whl (884kB)
    [K     |████████████████████████████████| 890kB 3.4MB/s 
    [?25hCollecting sacremoses
    [?25l  Downloading https://files.pythonhosted.org/packages/7d/34/09d19aff26edcc8eb2a01bed8e98f13a1537005d31e95233fd48216eed10/sacremoses-0.0.43.tar.gz (883kB)
    [K     |████████████████████████████████| 890kB 16.3MB/s 
    [?25hRequirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.6/dist-packages (from transformers) (4.41.1)
    Requirement already satisfied: dataclasses; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from transformers) (0.7)
    Collecting tokenizers==0.8.1.rc2
    [?25l  Downloading https://files.pythonhosted.org/packages/80/83/8b9fccb9e48eeb575ee19179e2bdde0ee9a1904f97de5f02d19016b8804f/tokenizers-0.8.1rc2-cp36-cp36m-manylinux1_x86_64.whl (3.0MB)
    [K     |████████████████████████████████| 3.0MB 25.3MB/s 
    [?25hRequirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from transformers) (2.23.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from transformers) (1.18.5)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.6/dist-packages (from transformers) (2019.12.20)
    Collecting sentencepiece!=0.1.92
    [?25l  Downloading https://files.pythonhosted.org/packages/d4/a4/d0a884c4300004a78cca907a6ff9a5e9fe4f090f5d95ab341c53d28cbc58/sentencepiece-0.1.91-cp36-cp36m-manylinux1_x86_64.whl (1.1MB)
    [K     |████████████████████████████████| 1.1MB 44.6MB/s 
    [?25hRequirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from transformers) (20.4)
    Requirement already satisfied: filelock in /usr/local/lib/python3.6/dist-packages (from transformers) (3.0.12)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (1.15.0)
    Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (7.1.2)
    Requirement already satisfied: joblib in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (0.16.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (3.0.4)
    Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from packaging->transformers) (2.4.7)
    Building wheels for collected packages: sacremoses
      Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
      Created wheel for sacremoses: filename=sacremoses-0.0.43-cp36-none-any.whl size=893257 sha256=b4d9f604b99e77f4dc2b8892460fc931726fa7d37c4fc51dfcb98c01d6d08797
      Stored in directory: /root/.cache/pip/wheels/29/3c/fd/7ce5c3f0666dab31a50123635e6fb5e19ceb42ce38d4e58f45
    Successfully built sacremoses
    Installing collected packages: sacremoses, tokenizers, sentencepiece, transformers
    Successfully installed sacremoses-0.0.43 sentencepiece-0.1.91 tokenizers-0.8.1rc2 transformers-3.1.0



```
MAX_LEN = 128
BATCH_SIZE = 16 # per TPU core
TOTAL_STEPS = 2000  # thats approx 4 epochs
EVALUATE_EVERY = 200
LR =  1e-5

PRETRAINED_MODEL = 'bert-base-uncased'


import os
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.optimizers import Adam
import transformers
from transformers import TFAutoModelWithLMHead, AutoTokenizer
import logging

AUTO = tf.data.experimental.AUTOTUNE
```

    2.3.0



```
def connect_to_TPU():
    """Detect hardware, return appropriate distribution strategy"""
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

    return tpu, strategy, global_batch_size


tpu, strategy, global_batch_size = connect_to_TPU()
print("REPLICAS: ", strategy.num_replicas_in_sync)
```

    INFO:absl:Entering into master device scope: /job:worker/replica:0/task:0/device:CPU:0


    Running on TPU  grpc://10.19.232.114:8470
    INFO:tensorflow:Initializing the TPU system: grpc://10.19.232.114:8470


    INFO:tensorflow:Initializing the TPU system: grpc://10.19.232.114:8470


    INFO:tensorflow:Clearing out eager caches


    INFO:tensorflow:Clearing out eager caches


    INFO:tensorflow:Finished initializing TPU system.


    INFO:tensorflow:Finished initializing TPU system.
    WARNING:absl:`tf.distribute.experimental.TPUStrategy` is deprecated, please use  the non experimental symbol `tf.distribute.TPUStrategy` instead.


    INFO:tensorflow:Found TPU system:


    INFO:tensorflow:Found TPU system:


    INFO:tensorflow:*** Num TPU Cores: 8


    INFO:tensorflow:*** Num TPU Cores: 8


    INFO:tensorflow:*** Num TPU Workers: 1


    INFO:tensorflow:*** Num TPU Workers: 1


    INFO:tensorflow:*** Num TPU Cores Per Worker: 8


    INFO:tensorflow:*** Num TPU Cores Per Worker: 8


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)


    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)


    REPLICAS:  8



```
!wget https://raw.githubusercontent.com/SrinidhiRaghavan/AI-Sentiment-Analysis-on-IMDB-Dataset/master/imdb_tr.csv
```

    --2020-09-02 10:33:57--  https://raw.githubusercontent.com/SrinidhiRaghavan/AI-Sentiment-Analysis-on-IMDB-Dataset/master/imdb_tr.csv
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 23677025 (23M) [text/plain]
    Saving to: ‘imdb_tr.csv’
    
    imdb_tr.csv         100%[===================>]  22.58M  49.2MB/s    in 0.5s    
    
    2020-09-02 10:33:58 (49.2 MB/s) - ‘imdb_tr.csv’ saved [23677025/23677025]
    



```
data = pd.read_csv('imdb_tr.csv', encoding = "ISO-8859-1")
```


```
data.head()
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
      <th>row_Number</th>
      <th>text</th>
      <th>polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2148</td>
      <td>first think another Disney movie, might good, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23577</td>
      <td>Put aside Dr. House repeat missed, Desperate H...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1319</td>
      <td>big fan Stephen King's work, film made even gr...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13358</td>
      <td>watched horrid thing TV. Needless say one movi...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9495</td>
      <td>truly enjoyed film. acting terrific plot. Jeff...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```
#data = data.sample(1000)
```


```
%%time

def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_mask=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen,
        truncation=True
    )
    
    return np.array(enc_di['input_ids'])
    

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
X_data = regular_encode(data.text.values, tokenizer, maxlen=MAX_LEN)
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=433.0, style=ProgressStyle(description_…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=231508.0, style=ProgressStyle(descripti…


    


    /usr/local/lib/python3.6/dist-packages/transformers/tokenization_utils_base.py:1770: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
      FutureWarning,


    CPU times: user 1min 4s, sys: 233 ms, total: 1min 4s
    Wall time: 1min 5s



```
def prepare_mlm_input_and_labels(X):
    # 15% BERT masking
    inp_mask = np.random.rand(*X.shape)<0.15 
    # do not mask special tokens
    inp_mask[X<=2] = False
    # set targets to -1 by default, it means ignore
    labels =  -1 * np.ones(X.shape, dtype=int)
    # set labels for masked tokens
    labels[inp_mask] = X[inp_mask]
    
    # prepare input
    X_mlm = np.copy(X)
    # set input to [MASK] which is the last token for the 90% of tokens
    # this means leaving 10% unchanged
    inp_mask_2mask = inp_mask  & (np.random.rand(*X.shape)<0.90)
    X_mlm[inp_mask_2mask] = tokenizer.mask_token_id  # mask token is the last in the dict

    # set 10% to a random token
    inp_mask_2random = inp_mask_2mask  & (np.random.rand(*X.shape) < 1/9)
    X_mlm[inp_mask_2random] = np.random.randint(3, tokenizer.mask_token_id, inp_mask_2random.sum())
    
    return X_mlm, labels


# use validation and test data for mlm
X_train_mlm = np.vstack(X_data)
# masks and labels
X_train_mlm, y_train_mlm = prepare_mlm_input_and_labels(X_train_mlm)
```


```
def create_dist_dataset(X, y=None, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(X)

    ### Add y if present ###
    if y is not None:
        dataset_y = tf.data.Dataset.from_tensor_slices(y)
        dataset = tf.data.Dataset.zip((dataset, dataset_y))
        
    ### Repeat if training ###
    if training:
        dataset = dataset.shuffle(len(X)).repeat()

    dataset = dataset.batch(global_batch_size).prefetch(AUTO)

    ### make it distributed  ###
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    return dist_dataset
    
    
train_dist_dataset = create_dist_dataset(X_train_mlm, y_train_mlm, True)

```


```
%%time

def create_mlm_model_and_optimizer():
    with strategy.scope():
        model = TFAutoModelWithLMHead.from_pretrained(PRETRAINED_MODEL)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    return model, optimizer


mlm_model, optimizer = create_mlm_model_and_optimizer()
mlm_model.summary()
```

    /usr/local/lib/python3.6/dist-packages/transformers/modeling_tf_auto.py:788: FutureWarning: The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use `TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models and `TFAutoModelForSeq2SeqLM` for encoder-decoder models.
      FutureWarning,



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=536063208.0, style=ProgressStyle(descri…


    


    Some weights of the model checkpoint at bert-base-uncased were not used when initializing TFBertForMaskedLM: ['nsp___cls']
    - This IS expected if you are initializing TFBertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing TFBertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    All the weights of TFBertForMaskedLM were initialized from the model checkpoint at bert-base-uncased.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertForMaskedLM for predictions without further training.


    Model: "tf_bert_for_masked_lm"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    bert (TFBertMainLayer)       multiple                  109482240 
    _________________________________________________________________
    mlm___cls (TFBertMLMHead)    multiple                  24459834  
    =================================================================
    Total params: 110,104,890
    Trainable params: 110,104,890
    Non-trainable params: 0
    _________________________________________________________________
    CPU times: user 14.5 s, sys: 15 s, total: 29.5 s
    Wall time: 58.1 s



```
def define_mlm_loss_and_metrics():
    with strategy.scope():
        mlm_loss_object = masked_sparse_categorical_crossentropy

        def compute_mlm_loss(labels, predictions):
            per_example_loss = mlm_loss_object(labels, predictions)
            loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size = global_batch_size)
            return loss

        train_mlm_loss_metric = tf.keras.metrics.Mean()
        
    return compute_mlm_loss, train_mlm_loss_metric


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked,
                                                          y_pred_masked,
                                                          from_logits=True)
    return loss

            
            
def train_mlm(train_dist_dataset, total_steps=2000, evaluate_every=200):
    step = 0
    ### Training lopp ###
    for tensor in train_dist_dataset:
        distributed_mlm_train_step(tensor) 
        step+=1

        if (step % evaluate_every == 0):   
            ### Print train metrics ###  
            train_metric = train_mlm_loss_metric.result().numpy()
            print("Step %d, train loss: %.2f" % (step, train_metric))     

            ### Reset  metrics ###
            train_mlm_loss_metric.reset_states()
            
        if step  == total_steps:
            break


@tf.function
def distributed_mlm_train_step(data):
    strategy.experimental_run_v2(mlm_train_step, args=(data,))


@tf.function
def mlm_train_step(inputs):
    features, labels = inputs

    with tf.GradientTape() as tape:
        predictions = mlm_model(features, training=True)[0]
        loss = compute_mlm_loss(labels, predictions)

    gradients = tape.gradient(loss, mlm_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mlm_model.trainable_variables))

    train_mlm_loss_metric.update_state(loss)
    

compute_mlm_loss, train_mlm_loss_metric = define_mlm_loss_and_metrics()
```


```
%%time
train_mlm(train_dist_dataset, TOTAL_STEPS, EVALUATE_EVERY)
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/data/ops/multi_device_iterator_ops.py:601: get_next_as_optional (from tensorflow.python.data.ops.iterator_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Iterator.get_next_as_optional()` instead.


    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/data/ops/multi_device_iterator_ops.py:601: get_next_as_optional (from tensorflow.python.data.ops.iterator_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Iterator.get_next_as_optional()` instead.


    WARNING:tensorflow:From <ipython-input-12-d78fc23ea715>:47: StrategyBase.experimental_run_v2 (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
    Instructions for updating:
    renamed to `run`


    WARNING:tensorflow:From <ipython-input-12-d78fc23ea715>:47: StrategyBase.experimental_run_v2 (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
    Instructions for updating:
    renamed to `run`


    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_masked_lm/bert/pooler/dense/kernel:0', 'tf_bert_for_masked_lm/bert/pooler/dense/bias:0'] when minimizing the loss.


    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_masked_lm/bert/pooler/dense/kernel:0', 'tf_bert_for_masked_lm/bert/pooler/dense/bias:0'] when minimizing the loss.


    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_masked_lm/bert/pooler/dense/kernel:0', 'tf_bert_for_masked_lm/bert/pooler/dense/bias:0'] when minimizing the loss.


    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_for_masked_lm/bert/pooler/dense/kernel:0', 'tf_bert_for_masked_lm/bert/pooler/dense/bias:0'] when minimizing the loss.


    Step 200, train loss: 8.89
    Step 400, train loss: 8.03
    Step 600, train loss: 7.68
    Step 800, train loss: 7.43
    Step 1000, train loss: 7.22
    Step 1200, train loss: 7.00
    Step 1400, train loss: 6.86
    Step 1600, train loss: 6.68
    Step 1800, train loss: 6.54
    Step 2000, train loss: 6.38
    CPU times: user 1min 23s, sys: 13.4 s, total: 1min 37s
    Wall time: 9min 3s



```
mlm_model.save_pretrained('imdb_bert_uncased')
```

# Load and Test


```
from transformers import *
from pprint import pprint
```


```
pretrained_model = TFAutoModelWithLMHead.from_pretrained(PRETRAINED_MODEL)
nlp = pipeline("fill-mask",model=pretrained_model, tokenizer=tokenizer ,framework='tf')
pprint(nlp(f"I watched {nlp.tokenizer.mask_token} and that was awesome"))
```

    /usr/local/lib/python3.6/dist-packages/transformers/modeling_tf_auto.py:788: FutureWarning: The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use `TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models and `TFAutoModelForSeq2SeqLM` for encoder-decoder models.
      FutureWarning,
    Some weights of the model checkpoint at bert-base-uncased were not used when initializing TFBertForMaskedLM: ['nsp___cls']
    - This IS expected if you are initializing TFBertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing TFBertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    All the weights of TFBertForMaskedLM were initialized from the model checkpoint at bert-base-uncased.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertForMaskedLM for predictions without further training.


    [{'score': 0.31239137053489685,
      'sequence': '[CLS] i watched him and that was awesome [SEP]',
      'token': 2032,
      'token_str': 'him'},
     {'score': 0.1729636937379837,
      'sequence': '[CLS] i watched her and that was awesome [SEP]',
      'token': 2014,
      'token_str': 'her'},
     {'score': 0.13816313445568085,
      'sequence': '[CLS] i watched it and that was awesome [SEP]',
      'token': 2009,
      'token_str': 'it'},
     {'score': 0.08374697715044022,
      'sequence': '[CLS] i watched, and that was awesome [SEP]',
      'token': 1010,
      'token_str': ','},
     {'score': 0.06438492983579636,
      'sequence': '[CLS] i watched them and that was awesome [SEP]',
      'token': 2068,
      'token_str': 'them'}]



```
movie_mlm_model = TFAutoModelWithLMHead.from_pretrained('imdb_bert_uncased')
nlp = pipeline("fill-mask",model=movie_mlm_model, tokenizer=tokenizer ,framework='tf')
pprint(nlp(f"I watched {nlp.tokenizer.mask_token} and that was awesome"))
```

    /usr/local/lib/python3.6/dist-packages/transformers/modeling_tf_auto.py:788: FutureWarning: The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use `TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models and `TFAutoModelForSeq2SeqLM` for encoder-decoder models.
      FutureWarning,
    All model checkpoint weights were used when initializing TFBertForMaskedLM.
    
    All the weights of TFBertForMaskedLM were initialized from the model checkpoint at imdb_bert_uncased.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertForMaskedLM for predictions without further training.


    [{'score': 0.4467789828777313,
      'sequence': '[CLS] i watched it and that was awesome [SEP]',
      'token': 2009,
      'token_str': 'it'},
     {'score': 0.06318594515323639,
      'sequence': '[CLS] i watched movie and that was awesome [SEP]',
      'token': 3185,
      'token_str': 'movie'},
     {'score': 0.056345004588365555,
      'sequence': '[CLS] i watched, and that was awesome [SEP]',
      'token': 1010,
      'token_str': ','},
     {'score': 0.013144557364284992,
      'sequence': '[CLS] i watched this and that was awesome [SEP]',
      'token': 2023,
      'token_str': 'this'},
     {'score': 0.012886741198599339,
      'sequence': '[CLS] i watched one and that was awesome [SEP]',
      'token': 2028,
      'token_str': 'one'}]



```

```
