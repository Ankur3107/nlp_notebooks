<a href="https://colab.research.google.com/github/Ankur3107/large-scale-multi-label-classification/blob/master/simpletransformers_intro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```
!pip install simpletransformers
```

    Collecting simpletransformers
    [?25l  Downloading https://files.pythonhosted.org/packages/14/f2/f0e219441ba3705dcfc6a4552171e177fa0f6d20df9adb62d94f76ff9fe6/simpletransformers-0.47.3-py3-none-any.whl (208kB)
    [K     |████████████████████████████████| 215kB 2.7MB/s 
    [?25hRequirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (2.23.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (1.18.5)
    Requirement already satisfied: regex in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (2019.12.20)
    Collecting transformers>=3.0.2
    [?25l  Downloading https://files.pythonhosted.org/packages/27/3c/91ed8f5c4e7ef3227b4119200fc0ed4b4fd965b1f0172021c25701087825/transformers-3.0.2-py3-none-any.whl (769kB)
    [K     |████████████████████████████████| 778kB 8.3MB/s 
    [?25hCollecting streamlit
    [?25l  Downloading https://files.pythonhosted.org/packages/7a/95/c1f097bfd0ea06f97d02e09e6e0af9bfa4da2c1e761112d5916bfd3bf846/streamlit-0.65.2-py2.py3-none-any.whl (7.2MB)
    [K     |████████████████████████████████| 7.2MB 16.7MB/s 
    [?25hCollecting seqeval
      Downloading https://files.pythonhosted.org/packages/34/91/068aca8d60ce56dd9ba4506850e876aba5e66a6f2f29aa223224b50df0de/seqeval-0.0.12.tar.gz
    Collecting tqdm>=4.47.0
    [?25l  Downloading https://files.pythonhosted.org/packages/28/7e/281edb5bc3274dfb894d90f4dbacfceaca381c2435ec6187a2c6f329aed7/tqdm-4.48.2-py2.py3-none-any.whl (68kB)
    [K     |████████████████████████████████| 71kB 7.8MB/s 
    [?25hRequirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (1.4.1)
    Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (1.0.5)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (0.22.2.post1)
    Collecting wandb
    [?25l  Downloading https://files.pythonhosted.org/packages/65/14/e7988204e4d4c9a349e73362399263b1c17f2b4d8a753864444f9eac1c92/wandb-0.9.5-py2.py3-none-any.whl (1.4MB)
    [K     |████████████████████████████████| 1.4MB 40.7MB/s 
    [?25hCollecting tokenizers
    [?25l  Downloading https://files.pythonhosted.org/packages/e9/ee/fedc3509145ad60fe5b418783f4a4c1b5462a4f0e8c7bbdbda52bdcda486/tokenizers-0.8.1-cp36-cp36m-manylinux1_x86_64.whl (3.0MB)
    [K     |████████████████████████████████| 3.0MB 31.9MB/s 
    [?25hCollecting tensorboardx
    [?25l  Downloading https://files.pythonhosted.org/packages/af/0c/4f41bcd45db376e6fe5c619c01100e9b7531c55791b7244815bac6eac32c/tensorboardX-2.1-py2.py3-none-any.whl (308kB)
    [K     |████████████████████████████████| 317kB 39.9MB/s 
    [?25hRequirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->simpletransformers) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->simpletransformers) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->simpletransformers) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->simpletransformers) (2020.6.20)
    Collecting sentencepiece!=0.1.92
    [?25l  Downloading https://files.pythonhosted.org/packages/d4/a4/d0a884c4300004a78cca907a6ff9a5e9fe4f090f5d95ab341c53d28cbc58/sentencepiece-0.1.91-cp36-cp36m-manylinux1_x86_64.whl (1.1MB)
    [K     |████████████████████████████████| 1.1MB 41.2MB/s 
    [?25hRequirement already satisfied: dataclasses; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from transformers>=3.0.2->simpletransformers) (0.7)
    Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from transformers>=3.0.2->simpletransformers) (20.4)
    Requirement already satisfied: filelock in /usr/local/lib/python3.6/dist-packages (from transformers>=3.0.2->simpletransformers) (3.0.12)
    Collecting sacremoses
    [?25l  Downloading https://files.pythonhosted.org/packages/7d/34/09d19aff26edcc8eb2a01bed8e98f13a1537005d31e95233fd48216eed10/sacremoses-0.0.43.tar.gz (883kB)
    [K     |████████████████████████████████| 890kB 23.0MB/s 
    [?25hCollecting base58
      Downloading https://files.pythonhosted.org/packages/3c/03/58572025c77b9e6027155b272a1b96298e711cd4f95c24967f7137ab0c4b/base58-2.0.1-py3-none-any.whl
    Requirement already satisfied: altair>=3.2.0 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (4.1.0)
    Requirement already satisfied: cachetools>=4.0 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (4.1.1)
    Requirement already satisfied: boto3 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (1.14.37)
    Collecting enum-compat
      Downloading https://files.pythonhosted.org/packages/55/ae/467bc4509246283bb59746e21a1a2f5a8aecbef56b1fa6eaca78cd438c8b/enum_compat-0.0.3-py3-none-any.whl
    Collecting pydeck>=0.1.dev5
    [?25l  Downloading https://files.pythonhosted.org/packages/51/1e/296f4108bf357e684617a776ecaf06ee93b43e30c35996dfac1aa985aa6c/pydeck-0.5.0b1-py2.py3-none-any.whl (4.4MB)
    [K     |████████████████████████████████| 4.4MB 39.2MB/s 
    [?25hRequirement already satisfied: astor in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (0.8.1)
    Collecting watchdog
    [?25l  Downloading https://files.pythonhosted.org/packages/0e/06/121302598a4fc01aca942d937f4a2c33430b7181137b35758913a8db10ad/watchdog-0.10.3.tar.gz (94kB)
    [K     |████████████████████████████████| 102kB 11.3MB/s 
    [?25hRequirement already satisfied: python-dateutil in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (2.8.1)
    Collecting validators
      Downloading https://files.pythonhosted.org/packages/89/3b/23e14394d0a719d1a9f2e1944a1d227ac7107a3383aa7e8eba60003e7266/validators-0.18.0-py3-none-any.whl
    Requirement already satisfied: tornado>=5.0 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (5.1.1)
    Requirement already satisfied: click>=7.0 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (7.1.2)
    Requirement already satisfied: botocore>=1.13.44 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (1.17.37)
    Requirement already satisfied: pyarrow in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (0.14.1)
    Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (7.0.0)
    Requirement already satisfied: toml in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (0.10.1)
    Requirement already satisfied: protobuf>=3.6.0 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (3.12.4)
    Collecting blinker
    [?25l  Downloading https://files.pythonhosted.org/packages/1b/51/e2a9f3b757eb802f61dc1f2b09c8c99f6eb01cf06416c0671253536517b6/blinker-1.4.tar.gz (111kB)
    [K     |████████████████████████████████| 112kB 39.5MB/s 
    [?25hRequirement already satisfied: tzlocal in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (1.5.1)
    Requirement already satisfied: Keras>=2.2.4 in /usr/local/lib/python3.6/dist-packages (from seqeval->simpletransformers) (2.4.3)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas->simpletransformers) (2018.9)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->simpletransformers) (0.16.0)
    Collecting gql==0.2.0
      Downloading https://files.pythonhosted.org/packages/c4/6f/cf9a3056045518f06184e804bae89390eb706168349daa9dff8ac609962a/gql-0.2.0.tar.gz
    Requirement already satisfied: nvidia-ml-py3>=7.352.0 in /usr/local/lib/python3.6/dist-packages (from wandb->simpletransformers) (7.352.0)
    Collecting GitPython>=1.0.0
    [?25l  Downloading https://files.pythonhosted.org/packages/f9/1e/a45320cab182bf1c8656107b3d4c042e659742822fc6bff150d769a984dd/GitPython-3.1.7-py3-none-any.whl (158kB)
    [K     |████████████████████████████████| 163kB 41.6MB/s 
    [?25hCollecting docker-pycreds>=0.4.0
      Downloading https://files.pythonhosted.org/packages/f5/e8/f6bd1eee09314e7e6dee49cbe2c5e22314ccdb38db16c9fc72d2fa80d054/docker_pycreds-0.4.0-py2.py3-none-any.whl
    Collecting configparser>=3.8.1
      Downloading https://files.pythonhosted.org/packages/4b/6b/01baa293090240cf0562cc5eccb69c6f5006282127f2b846fad011305c79/configparser-5.0.0-py3-none-any.whl
    Requirement already satisfied: PyYAML>=3.10 in /usr/local/lib/python3.6/dist-packages (from wandb->simpletransformers) (3.13)
    Collecting shortuuid>=0.5.0
      Downloading https://files.pythonhosted.org/packages/25/a6/2ecc1daa6a304e7f1b216f0896b26156b78e7c38e1211e9b798b4716c53d/shortuuid-1.0.1-py3-none-any.whl
    Requirement already satisfied: psutil>=5.0.0 in /usr/local/lib/python3.6/dist-packages (from wandb->simpletransformers) (5.4.8)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from wandb->simpletransformers) (1.15.0)
    Collecting sentry-sdk>=0.4.0
    [?25l  Downloading https://files.pythonhosted.org/packages/f4/4c/49f899856e3a83e02bc88f2c4945aa0bda4f56b804baa9f71e6664a574a2/sentry_sdk-0.16.5-py2.py3-none-any.whl (113kB)
    [K     |████████████████████████████████| 122kB 45.3MB/s 
    [?25hCollecting subprocess32>=3.5.3
    [?25l  Downloading https://files.pythonhosted.org/packages/32/c8/564be4d12629b912ea431f1a50eb8b3b9d00f1a0b1ceff17f266be190007/subprocess32-3.5.4.tar.gz (97kB)
    [K     |████████████████████████████████| 102kB 10.6MB/s 
    [?25hRequirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from packaging->transformers>=3.0.2->simpletransformers) (2.4.7)
    Requirement already satisfied: jsonschema in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit->simpletransformers) (2.6.0)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit->simpletransformers) (2.11.2)
    Requirement already satisfied: entrypoints in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit->simpletransformers) (0.3)
    Requirement already satisfied: toolz in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit->simpletransformers) (0.10.0)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.6/dist-packages (from boto3->streamlit->simpletransformers) (0.10.0)
    Requirement already satisfied: s3transfer<0.4.0,>=0.3.0 in /usr/local/lib/python3.6/dist-packages (from boto3->streamlit->simpletransformers) (0.3.3)
    Requirement already satisfied: ipywidgets>=7.0.0 in /usr/local/lib/python3.6/dist-packages (from pydeck>=0.1.dev5->streamlit->simpletransformers) (7.5.1)
    Collecting ipykernel>=5.1.2; python_version >= "3.4"
    [?25l  Downloading https://files.pythonhosted.org/packages/52/19/c2812690d8b340987eecd2cbc18549b1d130b94c5d97fcbe49f5f8710edf/ipykernel-5.3.4-py3-none-any.whl (120kB)
    [K     |████████████████████████████████| 122kB 43.4MB/s 
    [?25hRequirement already satisfied: traitlets>=4.3.2 in /usr/local/lib/python3.6/dist-packages (from pydeck>=0.1.dev5->streamlit->simpletransformers) (4.3.3)
    Collecting pathtools>=0.1.1
      Downloading https://files.pythonhosted.org/packages/e7/7f/470d6fcdf23f9f3518f6b0b76be9df16dcc8630ad409947f8be2eb0ed13a/pathtools-0.1.2.tar.gz
    Requirement already satisfied: decorator>=3.4.0 in /usr/local/lib/python3.6/dist-packages (from validators->streamlit->simpletransformers) (4.4.2)
    Requirement already satisfied: docutils<0.16,>=0.10 in /usr/local/lib/python3.6/dist-packages (from botocore>=1.13.44->streamlit->simpletransformers) (0.15.2)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from protobuf>=3.6.0->streamlit->simpletransformers) (49.2.0)
    Requirement already satisfied: h5py in /usr/local/lib/python3.6/dist-packages (from Keras>=2.2.4->seqeval->simpletransformers) (2.10.0)
    Collecting graphql-core<2,>=0.5.0
    [?25l  Downloading https://files.pythonhosted.org/packages/b0/89/00ad5e07524d8c523b14d70c685e0299a8b0de6d0727e368c41b89b7ed0b/graphql-core-1.1.tar.gz (70kB)
    [K     |████████████████████████████████| 71kB 7.8MB/s 
    [?25hRequirement already satisfied: promise<3,>=2.0 in /usr/local/lib/python3.6/dist-packages (from gql==0.2.0->wandb->simpletransformers) (2.3)
    Collecting gitdb<5,>=4.0.1
    [?25l  Downloading https://files.pythonhosted.org/packages/48/11/d1800bca0a3bae820b84b7d813ad1eff15a48a64caea9c823fc8c1b119e8/gitdb-4.0.5-py3-none-any.whl (63kB)
    [K     |████████████████████████████████| 71kB 8.4MB/s 
    [?25hRequirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages (from jinja2->altair>=3.2.0->streamlit->simpletransformers) (1.1.1)
    Requirement already satisfied: widgetsnbextension~=3.5.0 in /usr/local/lib/python3.6/dist-packages (from ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (3.5.1)
    Requirement already satisfied: ipython>=4.0.0; python_version >= "3.3" in /usr/local/lib/python3.6/dist-packages (from ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (5.5.0)
    Requirement already satisfied: nbformat>=4.2.0 in /usr/local/lib/python3.6/dist-packages (from ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (5.0.7)
    Requirement already satisfied: jupyter-client in /usr/local/lib/python3.6/dist-packages (from ipykernel>=5.1.2; python_version >= "3.4"->pydeck>=0.1.dev5->streamlit->simpletransformers) (5.3.5)
    Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.6/dist-packages (from traitlets>=4.3.2->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.2.0)
    Collecting smmap<4,>=3.0.1
      Downloading https://files.pythonhosted.org/packages/b0/9a/4d409a6234eb940e6a78dfdfc66156e7522262f5f2fecca07dc55915952d/smmap-3.0.4-py2.py3-none-any.whl
    Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.6/dist-packages (from widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (5.3.1)
    Requirement already satisfied: pygments in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (2.1.3)
    Requirement already satisfied: simplegeneric>0.8 in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.8.1)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.7.5)
    Requirement already satisfied: pexpect; sys_platform != "win32" in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (4.8.0)
    Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.4 in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (1.0.18)
    Requirement already satisfied: jupyter-core in /usr/local/lib/python3.6/dist-packages (from nbformat>=4.2.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (4.6.3)
    Requirement already satisfied: pyzmq>=13 in /usr/local/lib/python3.6/dist-packages (from jupyter-client->ipykernel>=5.1.2; python_version >= "3.4"->pydeck>=0.1.dev5->streamlit->simpletransformers) (19.0.2)
    Requirement already satisfied: Send2Trash in /usr/local/lib/python3.6/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (1.5.0)
    Requirement already satisfied: terminado>=0.8.1 in /usr/local/lib/python3.6/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.8.3)
    Requirement already satisfied: nbconvert in /usr/local/lib/python3.6/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (5.6.1)
    Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.6/dist-packages (from pexpect; sys_platform != "win32"->ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.6.0)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.6/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.2.5)
    Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.8.4)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (1.4.2)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.6.0)
    Requirement already satisfied: bleach in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (3.1.5)
    Requirement already satisfied: testpath in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.4.4)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.6/dist-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.5.1)
    Building wheels for collected packages: seqeval, sacremoses, watchdog, blinker, gql, subprocess32, pathtools, graphql-core
      Building wheel for seqeval (setup.py) ... [?25l[?25hdone
      Created wheel for seqeval: filename=seqeval-0.0.12-cp36-none-any.whl size=7424 sha256=8936ba073f9d800c754ae615ca94b6fb2b154e45cc4a44bf9e97bc359a55a072
      Stored in directory: /root/.cache/pip/wheels/4f/32/0a/df3b340a82583566975377d65e724895b3fad101a3fb729f68
      Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
      Created wheel for sacremoses: filename=sacremoses-0.0.43-cp36-none-any.whl size=893260 sha256=d707599938de7b2db237bf993863d8ff60828b7582a957ca47f501b6a6f61f54
      Stored in directory: /root/.cache/pip/wheels/29/3c/fd/7ce5c3f0666dab31a50123635e6fb5e19ceb42ce38d4e58f45
      Building wheel for watchdog (setup.py) ... [?25l[?25hdone
      Created wheel for watchdog: filename=watchdog-0.10.3-cp36-none-any.whl size=73870 sha256=b865eb2d2f95d6acc508b20b7c45b772477e967f5b58b2f06a96925a2b32aa78
      Stored in directory: /root/.cache/pip/wheels/a8/1d/38/2c19bb311f67cc7b4d07a2ec5ea36ab1a0a0ea50db994a5bc7
      Building wheel for blinker (setup.py) ... [?25l[?25hdone
      Created wheel for blinker: filename=blinker-1.4-cp36-none-any.whl size=13449 sha256=12c388e54e1449336b720ff67b9a6753cea0b6f8326294e1ab6abc10d7db1bda
      Stored in directory: /root/.cache/pip/wheels/92/a0/00/8690a57883956a301d91cf4ec999cc0b258b01e3f548f86e89
      Building wheel for gql (setup.py) ... [?25l[?25hdone
      Created wheel for gql: filename=gql-0.2.0-cp36-none-any.whl size=7630 sha256=0296d94cc81bbf8601c1ce3281ad60c61917b53a08f5cedd736dbd8fd28a0945
      Stored in directory: /root/.cache/pip/wheels/ce/0e/7b/58a8a5268655b3ad74feef5aa97946f0addafb3cbb6bd2da23
      Building wheel for subprocess32 (setup.py) ... [?25l[?25hdone
      Created wheel for subprocess32: filename=subprocess32-3.5.4-cp36-none-any.whl size=6489 sha256=17c9493efecdcf44c19bfbb395d66820a88d6204eb60de33deac1708662f649f
      Stored in directory: /root/.cache/pip/wheels/68/39/1a/5e402bdfdf004af1786c8b853fd92f8c4a04f22aad179654d1
      Building wheel for pathtools (setup.py) ... [?25l[?25hdone
      Created wheel for pathtools: filename=pathtools-0.1.2-cp36-none-any.whl size=8784 sha256=4210ca989c3e797d25ac98ebad56dcfd9e3543c8a07d9a295fd72989770210dd
      Stored in directory: /root/.cache/pip/wheels/0b/04/79/c3b0c3a0266a3cb4376da31e5bfe8bba0c489246968a68e843
      Building wheel for graphql-core (setup.py) ... [?25l[?25hdone
      Created wheel for graphql-core: filename=graphql_core-1.1-cp36-none-any.whl size=104650 sha256=76facbd1af6b52ad86e249b6d07ee3f3a23be3abdd2ed115514c986c37931033
      Stored in directory: /root/.cache/pip/wheels/45/99/d7/c424029bb0fe910c63b68dbf2aa20d3283d023042521bcd7d5
    Successfully built seqeval sacremoses watchdog blinker gql subprocess32 pathtools graphql-core
    [31mERROR: google-colab 1.0.0 has requirement ipykernel~=4.10, but you'll have ipykernel 5.3.4 which is incompatible.[0m
    [31mERROR: transformers 3.0.2 has requirement tokenizers==0.8.1.rc1, but you'll have tokenizers 0.8.1 which is incompatible.[0m
    Installing collected packages: sentencepiece, tokenizers, tqdm, sacremoses, transformers, base58, enum-compat, ipykernel, pydeck, pathtools, watchdog, validators, blinker, streamlit, seqeval, graphql-core, gql, smmap, gitdb, GitPython, docker-pycreds, configparser, shortuuid, sentry-sdk, subprocess32, wandb, tensorboardx, simpletransformers
      Found existing installation: tqdm 4.41.1
        Uninstalling tqdm-4.41.1:
          Successfully uninstalled tqdm-4.41.1
      Found existing installation: ipykernel 4.10.1
        Uninstalling ipykernel-4.10.1:
          Successfully uninstalled ipykernel-4.10.1
    Successfully installed GitPython-3.1.7 base58-2.0.1 blinker-1.4 configparser-5.0.0 docker-pycreds-0.4.0 enum-compat-0.0.3 gitdb-4.0.5 gql-0.2.0 graphql-core-1.1 ipykernel-5.3.4 pathtools-0.1.2 pydeck-0.5.0b1 sacremoses-0.0.43 sentencepiece-0.1.91 sentry-sdk-0.16.5 seqeval-0.0.12 shortuuid-1.0.1 simpletransformers-0.47.3 smmap-3.0.4 streamlit-0.65.2 subprocess32-3.5.4 tensorboardx-2.1 tokenizers-0.8.1 tqdm-4.48.2 transformers-3.0.2 validators-0.18.0 wandb-0.9.5 watchdog-0.10.3





```
!wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
```

    --2020-08-24 15:02:27--  https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.216.80.139
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.216.80.139|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 4475746 (4.3M) [application/zip]
    Saving to: ‘wikitext-2-v1.zip’
    
    wikitext-2-v1.zip   100%[===================>]   4.27M  2.93MB/s    in 1.5s    
    
    2020-08-24 15:02:29 (2.93 MB/s) - ‘wikitext-2-v1.zip’ saved [4475746/4475746]
    



```
!unzip wikitext-2-v1.zip
```

    Archive:  wikitext-2-v1.zip
       creating: wikitext-2/
      inflating: wikitext-2/wiki.test.tokens  
      inflating: wikitext-2/wiki.valid.tokens  
      inflating: wikitext-2/wiki.train.tokens  



```
ls wikitext-2/
```

    wiki.test.tokens  wiki.train.tokens  wiki.valid.tokens



```
from simpletransformers.language_modeling import LanguageModelingModel
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
}

model = LanguageModelingModel('bert', 'bert-base-cased', args=train_args, use_cuda=True)
```

    [34m[1mwandb[0m: [33mWARNING[0m W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.
    INFO:filelock:Lock 139946366819016 acquired on cache_dir/5e8a2b4893d13790ed4150ca1906be5f7a03d6c4ddf62296c383f6db42814db2.e13dbb970cb325137104fb2e5f36fe865f27746c6b526f6352861b1980eb80b1.lock



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=213450.0, style=ProgressStyle(descripti…


    INFO:filelock:Lock 139946366819016 released on cache_dir/5e8a2b4893d13790ed4150ca1906be5f7a03d6c4ddf62296c383f6db42814db2.e13dbb970cb325137104fb2e5f36fe865f27746c6b526f6352861b1980eb80b1.lock


    


    INFO:filelock:Lock 139948275897176 acquired on cache_dir/b945b69218e98b3e2c95acf911789741307dec43c698d35fad11c1ae28bda352.9da767be51e1327499df13488672789394e2ca38b877837e52618a67d7002391.lock



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=433.0, style=ProgressStyle(description_…


    INFO:filelock:Lock 139948275897176 released on cache_dir/b945b69218e98b3e2c95acf911789741307dec43c698d35fad11c1ae28bda352.9da767be51e1327499df13488672789394e2ca38b877837e52618a67d7002391.lock


    


    INFO:filelock:Lock 139947842575384 acquired on cache_dir/d8f11f061e407be64c4d5d7867ee61d1465263e24085cfa26abf183fdc830569.3fadbea36527ae472139fe84cddaa65454d7429f12d543d80bfc3ad70de55ac2.lock



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=435779157.0, style=ProgressStyle(descri…


    INFO:filelock:Lock 139947842575384 released on cache_dir/d8f11f061e407be64c4d5d7867ee61d1465263e24085cfa26abf183fdc830569.3fadbea36527ae472139fe84cddaa65454d7429f12d543d80bfc3ad70de55ac2.lock


    


    WARNING:transformers.modeling_utils:Some weights of the model checkpoint at bert-base-cased were not used when initializing BertForMaskedLM: ['cls.seq_relationship.weight', 'cls.seq_relationship.bias']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    WARNING:transformers.modeling_utils:Some weights of BertForMaskedLM were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['cls.predictions.decoder.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```
model.train_model("wikitext-2/wiki.train.tokens", eval_file="wikitext-2/wiki.test.tokens", batch_size=32)
```

    INFO:simpletransformers.language_modeling.language_modeling_utils: Creating features from dataset file at cache_dir/



    HBox(children=(FloatProgress(value=0.0, max=23767.0), HTML(value='')))


    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (724 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (629 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (520 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (910 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (517 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (511 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (616 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (538 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (848 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (518 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (515 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (513 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (533 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (663 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (520 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (640 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (538 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (579 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (519 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (527 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (839 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (547 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (524 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (589 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (546 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (519 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (552 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (535 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (522 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (536 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (524 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (546 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (666 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (807 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (522 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (585 > 512). Running this sequence through the model will result in indexing errors
    WARNING:transformers.tokenization_utils_base:Token indices sequence length is longer than the specified maximum sequence length for this model (682 > 512). Running this sequence through the model will result in indexing errors


    



    HBox(children=(FloatProgress(value=0.0, max=19808.0), HTML(value='')))


    INFO:simpletransformers.language_modeling.language_modeling_utils: Saving features into cached file cache_dir/bert_cached_lm_126_wiki.train.tokens
    INFO:simpletransformers.language_modeling.language_modeling_model: Training started


    



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Running Epoch 0 of 1', max=2476.0, style=ProgressStyle(de…


    
    



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-6-6f25e41f04f9> in <module>()
    ----> 1 model.train_model("wikitext-2/wiki.train.tokens", eval_file="wikitext-2/wiki.test.tokens", batch_size=32)
    

    /usr/local/lib/python3.6/dist-packages/simpletransformers/language_modeling/language_modeling_model.py in train_model(self, train_file, output_dir, show_running_loss, args, eval_file, verbose, **kwargs)
        368             eval_file=eval_file,
        369             verbose=verbose,
    --> 370             **kwargs,
        371         )
        372 


    /usr/local/lib/python3.6/dist-packages/simpletransformers/language_modeling/language_modeling_model.py in train(self, train_dataset, output_dir, show_running_loss, eval_file, verbose, **kwargs)
        564                 if args.fp16:
        565                     with amp.autocast():
    --> 566                         outputs = model(**inputs)
        567                         # model outputs are always tuple in pytorch-transformers (see doc)
        568                         loss = outputs[0]


    TypeError: BertForMaskedLM object argument after ** must be a mapping, not Tensor



```

```
