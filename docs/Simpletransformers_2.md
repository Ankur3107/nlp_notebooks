<a href="https://colab.research.google.com/github/Ankur3107/colab_notebooks/blob/master/Simpletransformers_2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```
!pip install simpletransformers
```

    Collecting simpletransformers
    [?25l  Downloading https://files.pythonhosted.org/packages/14/f2/f0e219441ba3705dcfc6a4552171e177fa0f6d20df9adb62d94f76ff9fe6/simpletransformers-0.47.3-py3-none-any.whl (208kB)
    [K     |████████████████████████████████| 215kB 2.8MB/s 
    [?25hCollecting wandb
    [?25l  Downloading https://files.pythonhosted.org/packages/65/14/e7988204e4d4c9a349e73362399263b1c17f2b4d8a753864444f9eac1c92/wandb-0.9.5-py2.py3-none-any.whl (1.4MB)
    [K     |████████████████████████████████| 1.4MB 9.1MB/s 
    [?25hRequirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (2.23.0)
    Collecting tqdm>=4.47.0
    [?25l  Downloading https://files.pythonhosted.org/packages/28/7e/281edb5bc3274dfb894d90f4dbacfceaca381c2435ec6187a2c6f329aed7/tqdm-4.48.2-py2.py3-none-any.whl (68kB)
    [K     |████████████████████████████████| 71kB 7.1MB/s 
    [?25hRequirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (1.4.1)
    Collecting seqeval
      Downloading https://files.pythonhosted.org/packages/34/91/068aca8d60ce56dd9ba4506850e876aba5e66a6f2f29aa223224b50df0de/seqeval-0.0.12.tar.gz
    Collecting transformers>=3.0.2
    [?25l  Downloading https://files.pythonhosted.org/packages/27/3c/91ed8f5c4e7ef3227b4119200fc0ed4b4fd965b1f0172021c25701087825/transformers-3.0.2-py3-none-any.whl (769kB)
    [K     |████████████████████████████████| 778kB 17.5MB/s 
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (1.18.5)
    Collecting tensorboardx
    [?25l  Downloading https://files.pythonhosted.org/packages/af/0c/4f41bcd45db376e6fe5c619c01100e9b7531c55791b7244815bac6eac32c/tensorboardX-2.1-py2.py3-none-any.whl (308kB)
    [K     |████████████████████████████████| 317kB 24.5MB/s 
    [?25hRequirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (0.22.2.post1)
    Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (1.0.5)
    Collecting tokenizers
    [?25l  Downloading https://files.pythonhosted.org/packages/e9/ee/fedc3509145ad60fe5b418783f4a4c1b5462a4f0e8c7bbdbda52bdcda486/tokenizers-0.8.1-cp36-cp36m-manylinux1_x86_64.whl (3.0MB)
    [K     |████████████████████████████████| 3.0MB 26.4MB/s 
    [?25hCollecting streamlit
    [?25l  Downloading https://files.pythonhosted.org/packages/7a/95/c1f097bfd0ea06f97d02e09e6e0af9bfa4da2c1e761112d5916bfd3bf846/streamlit-0.65.2-py2.py3-none-any.whl (7.2MB)
    [K     |████████████████████████████████| 7.2MB 42.1MB/s 
    [?25hRequirement already satisfied: regex in /usr/local/lib/python3.6/dist-packages (from simpletransformers) (2019.12.20)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from wandb->simpletransformers) (1.15.0)
    Collecting GitPython>=1.0.0
    [?25l  Downloading https://files.pythonhosted.org/packages/f9/1e/a45320cab182bf1c8656107b3d4c042e659742822fc6bff150d769a984dd/GitPython-3.1.7-py3-none-any.whl (158kB)
    [K     |████████████████████████████████| 163kB 56.4MB/s 
    [?25hCollecting watchdog>=0.8.3
    [?25l  Downloading https://files.pythonhosted.org/packages/0e/06/121302598a4fc01aca942d937f4a2c33430b7181137b35758913a8db10ad/watchdog-0.10.3.tar.gz (94kB)
    [K     |████████████████████████████████| 102kB 12.1MB/s 
    [?25hCollecting subprocess32>=3.5.3
    [?25l  Downloading https://files.pythonhosted.org/packages/32/c8/564be4d12629b912ea431f1a50eb8b3b9d00f1a0b1ceff17f266be190007/subprocess32-3.5.4.tar.gz (97kB)
    [K     |████████████████████████████████| 102kB 9.9MB/s 
    [?25hCollecting shortuuid>=0.5.0
      Downloading https://files.pythonhosted.org/packages/25/a6/2ecc1daa6a304e7f1b216f0896b26156b78e7c38e1211e9b798b4716c53d/shortuuid-1.0.1-py3-none-any.whl
    Collecting configparser>=3.8.1
      Downloading https://files.pythonhosted.org/packages/4b/6b/01baa293090240cf0562cc5eccb69c6f5006282127f2b846fad011305c79/configparser-5.0.0-py3-none-any.whl
    Requirement already satisfied: psutil>=5.0.0 in /usr/local/lib/python3.6/dist-packages (from wandb->simpletransformers) (5.4.8)
    Requirement already satisfied: nvidia-ml-py3>=7.352.0 in /usr/local/lib/python3.6/dist-packages (from wandb->simpletransformers) (7.352.0)
    Requirement already satisfied: Click>=7.0 in /usr/local/lib/python3.6/dist-packages (from wandb->simpletransformers) (7.1.2)
    Collecting gql==0.2.0
      Downloading https://files.pythonhosted.org/packages/c4/6f/cf9a3056045518f06184e804bae89390eb706168349daa9dff8ac609962a/gql-0.2.0.tar.gz
    Collecting docker-pycreds>=0.4.0
      Downloading https://files.pythonhosted.org/packages/f5/e8/f6bd1eee09314e7e6dee49cbe2c5e22314ccdb38db16c9fc72d2fa80d054/docker_pycreds-0.4.0-py2.py3-none-any.whl
    Requirement already satisfied: PyYAML>=3.10 in /usr/local/lib/python3.6/dist-packages (from wandb->simpletransformers) (3.13)
    Requirement already satisfied: python-dateutil>=2.6.1 in /usr/local/lib/python3.6/dist-packages (from wandb->simpletransformers) (2.8.1)
    Collecting sentry-sdk>=0.4.0
    [?25l  Downloading https://files.pythonhosted.org/packages/8f/0f/e6ae366e926589878f2b1e41485473ee0368e5b1b62fd0f8c0bc8311eb75/sentry_sdk-0.17.0-py2.py3-none-any.whl (115kB)
    [K     |████████████████████████████████| 122kB 48.5MB/s 
    [?25hRequirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->simpletransformers) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->simpletransformers) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->simpletransformers) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->simpletransformers) (1.24.3)
    Requirement already satisfied: Keras>=2.2.4 in /usr/local/lib/python3.6/dist-packages (from seqeval->simpletransformers) (2.4.3)
    Collecting sentencepiece!=0.1.92
    [?25l  Downloading https://files.pythonhosted.org/packages/d4/a4/d0a884c4300004a78cca907a6ff9a5e9fe4f090f5d95ab341c53d28cbc58/sentencepiece-0.1.91-cp36-cp36m-manylinux1_x86_64.whl (1.1MB)
    [K     |████████████████████████████████| 1.1MB 48.3MB/s 
    [?25hCollecting sacremoses
    [?25l  Downloading https://files.pythonhosted.org/packages/7d/34/09d19aff26edcc8eb2a01bed8e98f13a1537005d31e95233fd48216eed10/sacremoses-0.0.43.tar.gz (883kB)
    [K     |████████████████████████████████| 890kB 46.8MB/s 
    [?25hRequirement already satisfied: filelock in /usr/local/lib/python3.6/dist-packages (from transformers>=3.0.2->simpletransformers) (3.0.12)
    Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from transformers>=3.0.2->simpletransformers) (20.4)
    Requirement already satisfied: dataclasses; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from transformers>=3.0.2->simpletransformers) (0.7)
    Requirement already satisfied: protobuf>=3.8.0 in /usr/local/lib/python3.6/dist-packages (from tensorboardx->simpletransformers) (3.12.4)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->simpletransformers) (0.16.0)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas->simpletransformers) (2018.9)
    Requirement already satisfied: tornado>=5.0 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (5.1.1)
    Requirement already satisfied: boto3 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (1.14.47)
    Requirement already satisfied: altair>=3.2.0 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (4.1.0)
    Requirement already satisfied: pyarrow in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (0.14.1)
    Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (7.0.0)
    Requirement already satisfied: cachetools>=4.0 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (4.1.1)
    Requirement already satisfied: tzlocal in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (1.5.1)
    Requirement already satisfied: toml in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (0.10.1)
    Requirement already satisfied: botocore>=1.13.44 in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (1.17.47)
    Collecting validators
      Downloading https://files.pythonhosted.org/packages/89/3b/23e14394d0a719d1a9f2e1944a1d227ac7107a3383aa7e8eba60003e7266/validators-0.18.0-py3-none-any.whl
    Requirement already satisfied: astor in /usr/local/lib/python3.6/dist-packages (from streamlit->simpletransformers) (0.8.1)
    Collecting enum-compat
      Downloading https://files.pythonhosted.org/packages/55/ae/467bc4509246283bb59746e21a1a2f5a8aecbef56b1fa6eaca78cd438c8b/enum_compat-0.0.3-py3-none-any.whl
    Collecting blinker
    [?25l  Downloading https://files.pythonhosted.org/packages/1b/51/e2a9f3b757eb802f61dc1f2b09c8c99f6eb01cf06416c0671253536517b6/blinker-1.4.tar.gz (111kB)
    [K     |████████████████████████████████| 112kB 50.3MB/s 
    [?25hCollecting pydeck>=0.1.dev5
    [?25l  Downloading https://files.pythonhosted.org/packages/51/1e/296f4108bf357e684617a776ecaf06ee93b43e30c35996dfac1aa985aa6c/pydeck-0.5.0b1-py2.py3-none-any.whl (4.4MB)
    [K     |████████████████████████████████| 4.4MB 48.5MB/s 
    [?25hCollecting base58
      Downloading https://files.pythonhosted.org/packages/3c/03/58572025c77b9e6027155b272a1b96298e711cd4f95c24967f7137ab0c4b/base58-2.0.1-py3-none-any.whl
    Collecting gitdb<5,>=4.0.1
    [?25l  Downloading https://files.pythonhosted.org/packages/48/11/d1800bca0a3bae820b84b7d813ad1eff15a48a64caea9c823fc8c1b119e8/gitdb-4.0.5-py3-none-any.whl (63kB)
    [K     |████████████████████████████████| 71kB 9.7MB/s 
    [?25hCollecting pathtools>=0.1.1
      Downloading https://files.pythonhosted.org/packages/e7/7f/470d6fcdf23f9f3518f6b0b76be9df16dcc8630ad409947f8be2eb0ed13a/pathtools-0.1.2.tar.gz
    Collecting graphql-core<2,>=0.5.0
    [?25l  Downloading https://files.pythonhosted.org/packages/b0/89/00ad5e07524d8c523b14d70c685e0299a8b0de6d0727e368c41b89b7ed0b/graphql-core-1.1.tar.gz (70kB)
    [K     |████████████████████████████████| 71kB 8.7MB/s 
    [?25hRequirement already satisfied: promise<3,>=2.0 in /usr/local/lib/python3.6/dist-packages (from gql==0.2.0->wandb->simpletransformers) (2.3)
    Requirement already satisfied: h5py in /usr/local/lib/python3.6/dist-packages (from Keras>=2.2.4->seqeval->simpletransformers) (2.10.0)
    Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from packaging->transformers>=3.0.2->simpletransformers) (2.4.7)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from protobuf>=3.8.0->tensorboardx->simpletransformers) (49.6.0)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.6/dist-packages (from boto3->streamlit->simpletransformers) (0.10.0)
    Requirement already satisfied: s3transfer<0.4.0,>=0.3.0 in /usr/local/lib/python3.6/dist-packages (from boto3->streamlit->simpletransformers) (0.3.3)
    Requirement already satisfied: entrypoints in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit->simpletransformers) (0.3)
    Requirement already satisfied: jsonschema in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit->simpletransformers) (2.6.0)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit->simpletransformers) (2.11.2)
    Requirement already satisfied: toolz in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit->simpletransformers) (0.10.0)
    Requirement already satisfied: docutils<0.16,>=0.10 in /usr/local/lib/python3.6/dist-packages (from botocore>=1.13.44->streamlit->simpletransformers) (0.15.2)
    Requirement already satisfied: decorator>=3.4.0 in /usr/local/lib/python3.6/dist-packages (from validators->streamlit->simpletransformers) (4.4.2)
    Requirement already satisfied: ipywidgets>=7.0.0 in /usr/local/lib/python3.6/dist-packages (from pydeck>=0.1.dev5->streamlit->simpletransformers) (7.5.1)
    Requirement already satisfied: traitlets>=4.3.2 in /usr/local/lib/python3.6/dist-packages (from pydeck>=0.1.dev5->streamlit->simpletransformers) (4.3.3)
    Collecting ipykernel>=5.1.2; python_version >= "3.4"
    [?25l  Downloading https://files.pythonhosted.org/packages/52/19/c2812690d8b340987eecd2cbc18549b1d130b94c5d97fcbe49f5f8710edf/ipykernel-5.3.4-py3-none-any.whl (120kB)
    [K     |████████████████████████████████| 122kB 47.7MB/s 
    [?25hCollecting smmap<4,>=3.0.1
      Downloading https://files.pythonhosted.org/packages/b0/9a/4d409a6234eb940e6a78dfdfc66156e7522262f5f2fecca07dc55915952d/smmap-3.0.4-py2.py3-none-any.whl
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages (from jinja2->altair>=3.2.0->streamlit->simpletransformers) (1.1.1)
    Requirement already satisfied: ipython>=4.0.0; python_version >= "3.3" in /usr/local/lib/python3.6/dist-packages (from ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (5.5.0)
    Requirement already satisfied: nbformat>=4.2.0 in /usr/local/lib/python3.6/dist-packages (from ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (5.0.7)
    Requirement already satisfied: widgetsnbextension~=3.5.0 in /usr/local/lib/python3.6/dist-packages (from ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (3.5.1)
    Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.6/dist-packages (from traitlets>=4.3.2->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.2.0)
    Requirement already satisfied: jupyter-client in /usr/local/lib/python3.6/dist-packages (from ipykernel>=5.1.2; python_version >= "3.4"->pydeck>=0.1.dev5->streamlit->simpletransformers) (5.3.5)
    Requirement already satisfied: pygments in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (2.1.3)
    Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.4 in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (1.0.18)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.7.5)
    Requirement already satisfied: simplegeneric>0.8 in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.8.1)
    Requirement already satisfied: pexpect; sys_platform != "win32" in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (4.8.0)
    Requirement already satisfied: jupyter-core in /usr/local/lib/python3.6/dist-packages (from nbformat>=4.2.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (4.6.3)
    Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.6/dist-packages (from widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (5.3.1)
    Requirement already satisfied: pyzmq>=13 in /usr/local/lib/python3.6/dist-packages (from jupyter-client->ipykernel>=5.1.2; python_version >= "3.4"->pydeck>=0.1.dev5->streamlit->simpletransformers) (19.0.2)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.6/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.2.5)
    Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.6/dist-packages (from pexpect; sys_platform != "win32"->ipython>=4.0.0; python_version >= "3.3"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.6.0)
    Requirement already satisfied: Send2Trash in /usr/local/lib/python3.6/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (1.5.0)
    Requirement already satisfied: terminado>=0.8.1 in /usr/local/lib/python3.6/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.8.3)
    Requirement already satisfied: nbconvert in /usr/local/lib/python3.6/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (5.6.1)
    Requirement already satisfied: testpath in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.4.4)
    Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.8.4)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (1.4.2)
    Requirement already satisfied: bleach in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (3.1.5)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.6.0)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.6/dist-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit->simpletransformers) (0.5.1)
    Building wheels for collected packages: seqeval, watchdog, subprocess32, gql, sacremoses, blinker, pathtools, graphql-core
      Building wheel for seqeval (setup.py) ... [?25l[?25hdone
      Created wheel for seqeval: filename=seqeval-0.0.12-cp36-none-any.whl size=7423 sha256=0c9e5b916f7c76a179fa625f0d8b93874fafd41ad962954fa5ab259a75769366
      Stored in directory: /root/.cache/pip/wheels/4f/32/0a/df3b340a82583566975377d65e724895b3fad101a3fb729f68
      Building wheel for watchdog (setup.py) ... [?25l[?25hdone
      Created wheel for watchdog: filename=watchdog-0.10.3-cp36-none-any.whl size=73873 sha256=e29159085739fda9cb7db78f7615b4d1de26b9b9c3bb0d9c1fa953d71ec41687
      Stored in directory: /root/.cache/pip/wheels/a8/1d/38/2c19bb311f67cc7b4d07a2ec5ea36ab1a0a0ea50db994a5bc7
      Building wheel for subprocess32 (setup.py) ... [?25l[?25hdone
      Created wheel for subprocess32: filename=subprocess32-3.5.4-cp36-none-any.whl size=6489 sha256=627bc96100c3c3a95698ad43676e7a58057d6923b578f50ea6b4a8eebeec6dcf
      Stored in directory: /root/.cache/pip/wheels/68/39/1a/5e402bdfdf004af1786c8b853fd92f8c4a04f22aad179654d1
      Building wheel for gql (setup.py) ... [?25l[?25hdone
      Created wheel for gql: filename=gql-0.2.0-cp36-none-any.whl size=7630 sha256=178c6159f4f70ea80a4fb088ac0c6a8bac758e56266ff085915176cd0f18466a
      Stored in directory: /root/.cache/pip/wheels/ce/0e/7b/58a8a5268655b3ad74feef5aa97946f0addafb3cbb6bd2da23
      Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
      Created wheel for sacremoses: filename=sacremoses-0.0.43-cp36-none-any.whl size=893257 sha256=85e8b314242d4c1921d352f3e6058ec6559835318e580eb8cda29ce54af540c8
      Stored in directory: /root/.cache/pip/wheels/29/3c/fd/7ce5c3f0666dab31a50123635e6fb5e19ceb42ce38d4e58f45
      Building wheel for blinker (setup.py) ... [?25l[?25hdone
      Created wheel for blinker: filename=blinker-1.4-cp36-none-any.whl size=13450 sha256=e5ebfa369442a0bc4bb96651d7cf1a5f71257f0fe410bb1c25309f31eef4bdeb
      Stored in directory: /root/.cache/pip/wheels/92/a0/00/8690a57883956a301d91cf4ec999cc0b258b01e3f548f86e89
      Building wheel for pathtools (setup.py) ... [?25l[?25hdone
      Created wheel for pathtools: filename=pathtools-0.1.2-cp36-none-any.whl size=8785 sha256=8521122511ac313d2288474c30aa87876f562f34b1f1a412ac1294050011427b
      Stored in directory: /root/.cache/pip/wheels/0b/04/79/c3b0c3a0266a3cb4376da31e5bfe8bba0c489246968a68e843
      Building wheel for graphql-core (setup.py) ... [?25l[?25hdone
      Created wheel for graphql-core: filename=graphql_core-1.1-cp36-none-any.whl size=104651 sha256=5647c6aa90c5c7b0472476dfa74acf6b1d42386915e37c50856c526b4d79c271
      Stored in directory: /root/.cache/pip/wheels/45/99/d7/c424029bb0fe910c63b68dbf2aa20d3283d023042521bcd7d5
    Successfully built seqeval watchdog subprocess32 gql sacremoses blinker pathtools graphql-core
    [31mERROR: google-colab 1.0.0 has requirement ipykernel~=4.10, but you'll have ipykernel 5.3.4 which is incompatible.[0m
    [31mERROR: transformers 3.0.2 has requirement tokenizers==0.8.1.rc1, but you'll have tokenizers 0.8.1 which is incompatible.[0m
    Installing collected packages: smmap, gitdb, GitPython, pathtools, watchdog, subprocess32, shortuuid, configparser, graphql-core, gql, docker-pycreds, sentry-sdk, wandb, tqdm, seqeval, tokenizers, sentencepiece, sacremoses, transformers, tensorboardx, validators, enum-compat, blinker, ipykernel, pydeck, base58, streamlit, simpletransformers
      Found existing installation: tqdm 4.41.1
        Uninstalling tqdm-4.41.1:
          Successfully uninstalled tqdm-4.41.1
      Found existing installation: ipykernel 4.10.1
        Uninstalling ipykernel-4.10.1:
          Successfully uninstalled ipykernel-4.10.1
    Successfully installed GitPython-3.1.7 base58-2.0.1 blinker-1.4 configparser-5.0.0 docker-pycreds-0.4.0 enum-compat-0.0.3 gitdb-4.0.5 gql-0.2.0 graphql-core-1.1 ipykernel-5.3.4 pathtools-0.1.2 pydeck-0.5.0b1 sacremoses-0.0.43 sentencepiece-0.1.91 sentry-sdk-0.17.0 seqeval-0.0.12 shortuuid-1.0.1 simpletransformers-0.47.3 smmap-3.0.4 streamlit-0.65.2 subprocess32-3.5.4 tensorboardx-2.1 tokenizers-0.8.1 tqdm-4.48.2 transformers-3.0.2 validators-0.18.0 wandb-0.9.5 watchdog-0.10.3





```
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", "true"],
    ["Frodo was the heir of Isildur", "false"],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", "true"],
    ["Merry was the king of Rohan", "false"],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Optional model configuration
model_args = ClassificationArgs()
model_args.num_train_epochs=1
model_args.labels_list = ["true", "false"]

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args,use_cuda=False
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])
```

    INFO:filelock:Lock 140155715063760 acquired on /root/.cache/torch/transformers/80b4a484eddeb259bec2f06a6f2f05d90934111628e0e1c09a33bd4a121358e1.49b88ba7ec2c26a7558dda98ca3884c3b80fa31cf43a1b1f23aef3ff81ba344e.lock



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=501200538.0, style=ProgressStyle(descri…


    INFO:filelock:Lock 140155715063760 released on /root/.cache/torch/transformers/80b4a484eddeb259bec2f06a6f2f05d90934111628e0e1c09a33bd4a121358e1.49b88ba7ec2c26a7558dda98ca3884c3b80fa31cf43a1b1f23aef3ff81ba344e.lock


    


    WARNING:transformers.modeling_utils:Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    WARNING:transformers.modeling_utils:Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    INFO:filelock:Lock 140155715068592 acquired on /root/.cache/torch/transformers/d0c5776499adc1ded22493fae699da0971c1ee4c2587111707a4d177d20257a2.ef00af9e673c7160b4d41cfda1f48c5f4cba57d5142754525572a846a1ab1b9b.lock



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=898823.0, style=ProgressStyle(descripti…


    INFO:filelock:Lock 140155715068592 released on /root/.cache/torch/transformers/d0c5776499adc1ded22493fae699da0971c1ee4c2587111707a4d177d20257a2.ef00af9e673c7160b4d41cfda1f48c5f4cba57d5142754525572a846a1ab1b9b.lock
    INFO:filelock:Lock 140155714555408 acquired on /root/.cache/torch/transformers/b35e7cd126cd4229a746b5d5c29a749e8e84438b14bcdb575950584fe33207e8.70bec105b4158ed9a1747fea67a43f5dee97855c64d62b6ec3742f4cfdb5feda.lock


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=456318.0, style=ProgressStyle(descripti…


    INFO:filelock:Lock 140155714555408 released on /root/.cache/torch/transformers/b35e7cd126cd4229a746b5d5c29a749e8e84438b14bcdb575950584fe33207e8.70bec105b4158ed9a1747fea67a43f5dee97855c64d62b6ec3742f4cfdb5feda.lock
    INFO:simpletransformers.classification.classification_model: Converting to features started. Cache is not used.


    



    HBox(children=(FloatProgress(value=0.0, max=2.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Running Epoch 0 of 1', max=1.0, style=ProgressStyle(descr…


    


    /usr/local/lib/python3.6/dist-packages/torch/optim/lr_scheduler.py:200: UserWarning: Please also save or load the state of the optimzer when saving or loading the scheduler.
      warnings.warn(SAVE_STATE_WARNING, UserWarning)


    


    INFO:simpletransformers.classification.classification_model: Training of roberta model complete. Saved to outputs/.
    INFO:simpletransformers.classification.classification_model: Converting to features started. Cache is not used.



    HBox(children=(FloatProgress(value=0.0, max=2.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, description='Running Evaluation', max=1.0, style=ProgressStyle(descrip…


    


    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:900: RuntimeWarning: invalid value encountered in double_scalars
      mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
    INFO:simpletransformers.classification.classification_model:{'mcc': 0.0, 'tp': 1, 'tn': 0, 'fp': 1, 'fn': 0, 'eval_loss': 0.696165144443512}
    INFO:simpletransformers.classification.classification_model: Converting to features started. Cache is not used.



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    



```
predictions
```




    ['false']




```
ls outputs/
```

    [0m[01;34mcheckpoint-1-epoch-1[0m/  merges.txt         special_tokens_map.json  vocab.json
    config.json            model_args.json    tokenizer_config.json
    eval_results.txt       pytorch_model.bin  training_args.bin



```
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", "true"],
    ["Frodo was the heir of Isildur", "false"],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", "true"],
    ["Merry was the king of Rohan", "false"],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Optional model configuration
model_args = ClassificationArgs()
model_args.num_train_epochs=1
model_args.labels_list = ["true", "false"]
model_args.overwrite_output_dir = True
# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "outputs/", args=model_args,use_cuda=False
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])
```

    INFO:simpletransformers.classification.classification_model: Converting to features started. Cache is not used.



    HBox(children=(FloatProgress(value=0.0, max=2.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=1.0, style=ProgressStyle(description_width='i…


    INFO:simpletransformers.classification.classification_model:   Starting fine-tuning.



    HBox(children=(FloatProgress(value=0.0, description='Running Epoch 0 of 1', max=1.0, style=ProgressStyle(descr…


    


    /usr/local/lib/python3.6/dist-packages/torch/optim/lr_scheduler.py:200: UserWarning: Please also save or load the state of the optimzer when saving or loading the scheduler.
      warnings.warn(SAVE_STATE_WARNING, UserWarning)


    


    INFO:simpletransformers.classification.classification_model: Training of roberta model complete. Saved to outputs/.
    INFO:simpletransformers.classification.classification_model: Converting to features started. Cache is not used.



    HBox(children=(FloatProgress(value=0.0, max=2.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, description='Running Evaluation', max=1.0, style=ProgressStyle(descrip…


    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:900: RuntimeWarning: invalid value encountered in double_scalars
      mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
    INFO:simpletransformers.classification.classification_model:{'mcc': 0.0, 'tp': 1, 'tn': 0, 'fp': 1, 'fn': 0, 'eval_loss': 0.696165144443512}
    INFO:simpletransformers.classification.classification_model: Converting to features started. Cache is not used.


    



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    



```
import logging

import pandas as pd
import sklearn

import wandb
from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel,
)

sweep_config = {
    "method": "bayes",  # grid, random
    "metric": {"name": "train_loss", "goal": "minimize"},
    "parameters": {
        "num_train_epochs": {"values": [2, 3, 5]},
        "learning_rate": {"min": 5e-5, "max": 4e-4},
    },
}

sweep_id = wandb.sweep(sweep_config, project="Simple Sweep")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", "true"],
    ["Frodo was the heir of Isildur", "false"],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", "true"],
    ["Merry was the king of Rohan", "false"],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

model_args = ClassificationArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.evaluate_during_training = True
model_args.manual_seed = 4
model_args.train_batch_size = 16
model_args.eval_batch_size = 8
model_args.labels_list = ["true", "false"]
model_args.wandb_project = "Simple Sweep"

def train():
    # Initialize a new wandb run
    wandb.init()

    # Create a TransformerModel
    model = ClassificationModel(
        "roberta",
        "roberta-base",
        use_cuda=False,
        args=model_args,
        sweep_config=wandb.config,
    )

    # Train the model
    model.train_model(train_df, eval_df=eval_df)

    # Evaluate the model
    model.eval_model(eval_df)

    # Sync wandb
    wandb.join()


wandb.agent(sweep_id, train)
```

    Create sweep with ID: 03byb6mp
    Sweep URL: https://app.wandb.ai/ankur310794/Simple%20Sweep/sweeps/03byb6mp


    INFO:wandb.wandb_agent:Running runs: []
    INFO:wandb.wandb_agent:Agent received command: run
    INFO:wandb.wandb_agent:Agent starting run with config:
    	learning_rate: 0.00015971839686415725
    	num_train_epochs: 3


    wandb: Agent Starting Run: zt8m9f5b with config:
    	learning_rate: 0.00015971839686415725
    	num_train_epochs: 3
    wandb: Agent Started Run: zt8m9f5b




                Logging results to <a href="https://wandb.com" target="_blank">Weights & Biases</a> <a href="https://docs.wandb.com/integrations/jupyter.html" target="_blank">(Documentation)</a>.<br/>
                Project page: <a href="https://app.wandb.ai/ankur310794/Simple%20Sweep" target="_blank">https://app.wandb.ai/ankur310794/Simple%20Sweep</a><br/>
                Sweep page: <a href="https://app.wandb.ai/ankur310794/Simple%20Sweep/sweeps/03byb6mp" target="_blank">https://app.wandb.ai/ankur310794/Simple%20Sweep/sweeps/03byb6mp</a><br/>
Run page: <a href="https://app.wandb.ai/ankur310794/Simple%20Sweep/runs/zt8m9f5b" target="_blank">https://app.wandb.ai/ankur310794/Simple%20Sweep/runs/zt8m9f5b</a><br/>



    INFO:wandb.run_manager:system metrics and metadata threads started
    INFO:wandb.run_manager:checking resume status, waiting at most 10 seconds
    INFO:wandb.run_manager:resuming run from id: UnVuOnYxOnp0OG05ZjViOlNpbXBsZSBTd2VlcDphbmt1cjMxMDc5NA==
    INFO:wandb.run_manager:upserting run before process can begin, waiting at most 10 seconds
    INFO:wandb.run_manager:saving pip packages
    INFO:wandb.run_manager:initializing streaming files api
    INFO:wandb.run_manager:unblocking file change observer, beginning sync with W&B servers
    INFO:wandb.run_manager:file/dir modified: /content/wandb/run-20200827_001734-zt8m9f5b/config.yaml
    INFO:wandb.run_manager:file/dir created: /content/wandb/run-20200827_001734-zt8m9f5b/wandb-history.jsonl
    INFO:wandb.run_manager:file/dir created: /content/wandb/run-20200827_001734-zt8m9f5b/requirements.txt
    INFO:wandb.run_manager:file/dir created: /content/wandb/run-20200827_001734-zt8m9f5b/wandb-summary.json
    INFO:wandb.run_manager:file/dir created: /content/wandb/run-20200827_001734-zt8m9f5b/wandb-events.jsonl
    INFO:wandb.run_manager:file/dir created: /content/wandb/run-20200827_001734-zt8m9f5b/wandb-metadata.json
    INFO:wandb.wandb_agent:Running runs: ['zt8m9f5b']
    WARNING:transformers.modeling_utils:Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    WARNING:transformers.modeling_utils:Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    INFO:simpletransformers.classification.classification_model: Converting to features started. Cache is not used.



    HBox(children=(FloatProgress(value=0.0, max=2.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=3.0, style=ProgressStyle(description_width='i…




Logging results to <a href="https://wandb.com" target="_blank">Weights & Biases</a> <a href="https://docs.wandb.com/integrations/jupyter.html" target="_blank">(Documentation)</a>.<br/>
Project page: <a href="https://app.wandb.ai/ankur310794/Simple%20Sweep" target="_blank">https://app.wandb.ai/ankur310794/Simple%20Sweep</a><br/>
Run page: <a href="https://app.wandb.ai/ankur310794/Simple%20Sweep/runs/s4yq3tsr" target="_blank">https://app.wandb.ai/ankur310794/Simple%20Sweep/runs/s4yq3tsr</a><br/>



    INFO:wandb.run_manager:system metrics and metadata threads started
    INFO:wandb.run_manager:checking resume status, waiting at most 10 seconds
    INFO:wandb.run_manager:resuming run from id: UnVuOnYxOnM0eXEzdHNyOlNpbXBsZSBTd2VlcDphbmt1cjMxMDc5NA==
    INFO:wandb.run_manager:upserting run before process can begin, waiting at most 10 seconds
    INFO:wandb.run_manager:saving pip packages
    INFO:wandb.run_manager:initializing streaming files api
    INFO:wandb.run_manager:unblocking file change observer, beginning sync with W&B servers



    HBox(children=(FloatProgress(value=0.0, description='Running Epoch 0 of 3', max=1.0, style=ProgressStyle(descr…


    INFO:wandb.run_manager:file/dir modified: /content/wandb/run-20200827_001741-s4yq3tsr/config.yaml
    INFO:wandb.run_manager:file/dir created: /content/wandb/run-20200827_001741-s4yq3tsr/wandb-history.jsonl
    INFO:wandb.run_manager:file/dir created: /content/wandb/run-20200827_001741-s4yq3tsr/wandb-metadata.json
    INFO:wandb.run_manager:file/dir created: /content/wandb/run-20200827_001741-s4yq3tsr/wandb-events.jsonl
    INFO:wandb.run_manager:file/dir created: /content/wandb/run-20200827_001741-s4yq3tsr/wandb-summary.json
    INFO:wandb.run_manager:file/dir created: /content/wandb/run-20200827_001741-s4yq3tsr/requirements.txt


    
    


    Process Process-11:
    Traceback (most recent call last):
      File "/usr/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
      File "/usr/lib/python3.6/multiprocessing/process.py", line 93, in run
        self._target(*self._args, **self._kwargs)
      File "/usr/local/lib/python3.6/dist-packages/wandb/wandb_agent.py", line 64, in _start
        function()
      File "<ipython-input-11-75bdbb0e00ff>", line 67, in train
        model.train_model(train_df, eval_df=eval_df)
      File "/usr/local/lib/python3.6/dist-packages/simpletransformers/classification/classification_model.py", line 306, in train_model
        **kwargs,
      File "/usr/local/lib/python3.6/dist-packages/simpletransformers/classification/classification_model.py", line 494, in train
        loss.backward()
      File "/usr/local/lib/python3.6/dist-packages/torch/tensor.py", line 185, in backward
        torch.autograd.backward(self, gradient, retain_graph, create_graph)
      File "/usr/local/lib/python3.6/dist-packages/torch/autograd/__init__.py", line 127, in backward
        allow_unreachable=True)  # allow_unreachable flag
    RuntimeError: Unable to handle autograd's threading in combination with fork-based multiprocessing. See https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
    [34m[1mwandb[0m: Ctrl-c pressed. Waiting for runs to end. Press ctrl-c again to terminate them.



```
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", 1],
    ["Frodo was the heir of Isildur", 0],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", 1],
    ["Merry was the king of Rohan", 0],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Train only the classifier layers
model_args = ClassificationArgs()
model_args.train_custom_parameters_only = True
model_args.custom_parameter_groups = [
    {
        "params": ["classifier.weight"],
        "lr": 1e-3,
    },
    {
        "params": ["classifier.bias"],
        "lr": 1e-3,
        "weight_decay": 0.0,
    },
]
model_args.overwrite_output_dir=True
# Create a ClassificationModel
model = ClassificationModel(
    "bert", "bert-base-cased", args=model_args,use_cuda=False
)

# Train the model
model.train_model(train_df)
```

    WARNING:transformers.modeling_utils:Some weights of the model checkpoint at bert-base-cased were not used when initializing BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    WARNING:transformers.modeling_utils:Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    INFO:simpletransformers.classification.classification_model: Converting to features started. Cache is not used.



    HBox(children=(FloatProgress(value=0.0, max=2.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Running Epoch 0 of 1', max=1.0, style=ProgressStyle(descr…


    


    /usr/local/lib/python3.6/dist-packages/torch/optim/lr_scheduler.py:200: UserWarning: Please also save or load the state of the optimzer when saving or loading the scheduler.
      warnings.warn(SAVE_STATE_WARNING, UserWarning)


    


    INFO:simpletransformers.classification.classification_model: Training of bert model complete. Saved to outputs/.



```

```
