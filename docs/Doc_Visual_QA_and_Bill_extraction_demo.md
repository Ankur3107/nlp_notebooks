<a href="https://colab.research.google.com/github/Ankur3107/nlp_notebooks/blob/master/OCR-Document-Processing/Doc_Visual_QA_and_Bill_extraction_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
!wget --no-check-certificate https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz
```

    --2022-05-01 13:20:26--  https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz
    Resolving datasets.cvc.uab.es (datasets.cvc.uab.es)... 158.109.8.18
    Connecting to datasets.cvc.uab.es (datasets.cvc.uab.es)|158.109.8.18|:443... connected.
    WARNING: cannot verify datasets.cvc.uab.es's certificate, issued by ‘CN=GEANT OV RSA CA 4,O=GEANT Vereniging,C=NL’:
      Unable to locally verify the issuer's authority.
    HTTP request sent, awaiting response... 200 OK
    Length: 7122739200 (6.6G) [application/x-gzip]
    Saving to: ‘train.tar.gz.1’
    
    train.tar.gz.1        0%[                    ]  39.45M   638KB/s    eta 2h 26m 


```python
!wget --no-check-certificate https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz
```


```python
!wget --no-check-certificate https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz
```

# Install Packages


```python
!pip install -q transformers
```

    [K     |████████████████████████████████| 4.0 MB 5.3 MB/s 
    [K     |████████████████████████████████| 895 kB 45.4 MB/s 
    [K     |████████████████████████████████| 6.6 MB 31.6 MB/s 
    [K     |████████████████████████████████| 77 kB 4.5 MB/s 
    [K     |████████████████████████████████| 596 kB 43.9 MB/s 
    [?25h


```python
!pip install pyyaml==5.1
# workaround: install old version of pytorch since detectron2 hasn't released packages for pytorch 1.9 (issue: https://github.com/facebookresearch/detectron2/issues/3158)
!pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install detectron2 that matches pytorch 1.8
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
!pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
# exit(0)  # After installation, you need to "restart runtime" in Colab. This line can also restart runtime
```

    Collecting pyyaml==5.1
      Downloading PyYAML-5.1.tar.gz (274 kB)
    [K     |████████████████████████████████| 274 kB 5.3 MB/s 
    [?25hBuilding wheels for collected packages: pyyaml
      Building wheel for pyyaml (setup.py) ... [?25l[?25hdone
      Created wheel for pyyaml: filename=PyYAML-5.1-cp37-cp37m-linux_x86_64.whl size=44092 sha256=923c6817c78b049bf1912c6f36e26c890af93770cb6627b50efc1b77ac4eeeae
      Stored in directory: /root/.cache/pip/wheels/77/f5/10/d00a2bd30928b972790053b5de0c703ca87324f3fead0f2fd9
    Successfully built pyyaml
    Installing collected packages: pyyaml
      Attempting uninstall: pyyaml
        Found existing installation: PyYAML 6.0
        Uninstalling PyYAML-6.0:
          Successfully uninstalled PyYAML-6.0
    Successfully installed pyyaml-5.1
    Looking in links: https://download.pytorch.org/whl/torch_stable.html
    Collecting torch==1.8.0+cu101
      Downloading https://download.pytorch.org/whl/cu101/torch-1.8.0%2Bcu101-cp37-cp37m-linux_x86_64.whl (763.5 MB)
    [K     |████████████████████████████████| 763.5 MB 16 kB/s 
    [?25hCollecting torchvision==0.9.0+cu101
      Downloading https://download.pytorch.org/whl/cu101/torchvision-0.9.0%2Bcu101-cp37-cp37m-linux_x86_64.whl (17.3 MB)
    [K     |████████████████████████████████| 17.3 MB 795 kB/s 
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from torch==1.8.0+cu101) (1.21.6)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch==1.8.0+cu101) (4.2.0)
    Requirement already satisfied: pillow>=4.1.1 in /usr/local/lib/python3.7/dist-packages (from torchvision==0.9.0+cu101) (7.1.2)
    Installing collected packages: torch, torchvision
      Attempting uninstall: torch
        Found existing installation: torch 1.11.0+cu113
        Uninstalling torch-1.11.0+cu113:
          Successfully uninstalled torch-1.11.0+cu113
      Attempting uninstall: torchvision
        Found existing installation: torchvision 0.12.0+cu113
        Uninstalling torchvision-0.12.0+cu113:
          Successfully uninstalled torchvision-0.12.0+cu113
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    torchtext 0.12.0 requires torch==1.11.0, but you have torch 1.8.0+cu101 which is incompatible.
    torchaudio 0.11.0+cu113 requires torch==1.11.0, but you have torch 1.8.0+cu101 which is incompatible.[0m
    Successfully installed torch-1.8.0+cu101 torchvision-0.9.0+cu101
    [K     |████████████████████████████████| 6.3 MB 897 kB/s 
    [K     |████████████████████████████████| 74 kB 2.1 MB/s 
    [K     |████████████████████████████████| 50 kB 5.2 MB/s 
    [K     |████████████████████████████████| 147 kB 10.7 MB/s 
    [K     |████████████████████████████████| 130 kB 33.3 MB/s 
    [K     |████████████████████████████████| 749 kB 42.3 MB/s 
    [K     |████████████████████████████████| 843 kB 34.7 MB/s 
    [K     |████████████████████████████████| 112 kB 46.3 MB/s 
    [?25h  Building wheel for fvcore (setup.py) ... [?25l[?25hdone
      Building wheel for antlr4-python3-runtime (setup.py) ... [?25l[?25hdone



```python
!pip install -q datasets
```

    [K     |████████████████████████████████| 325 kB 5.4 MB/s 
    [K     |████████████████████████████████| 212 kB 43.4 MB/s 
    [K     |████████████████████████████████| 136 kB 43.2 MB/s 
    [K     |████████████████████████████████| 1.1 MB 35.9 MB/s 
    [K     |████████████████████████████████| 127 kB 45.1 MB/s 
    [K     |████████████████████████████████| 144 kB 46.5 MB/s 
    [K     |████████████████████████████████| 94 kB 2.6 MB/s 
    [K     |████████████████████████████████| 271 kB 47.1 MB/s 
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    datascience 0.10.6 requires folium==0.2.1, but you have folium 0.8.3 which is incompatible.[0m
    [?25h


```python
!sudo apt install tesseract-ocr
!pip install -q pytesseract
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    The following packages were automatically installed and are no longer required:
      libnvidia-common-460 nsight-compute-2020.2.0
    Use 'sudo apt autoremove' to remove them.
    The following additional packages will be installed:
      tesseract-ocr-eng tesseract-ocr-osd
    The following NEW packages will be installed:
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd
    0 upgraded, 3 newly installed, 0 to remove and 42 not upgraded.
    Need to get 4,795 kB of archives.
    After this operation, 15.8 MB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu bionic/universe amd64 tesseract-ocr-eng all 4.00~git24-0e00fe6-1.2 [1,588 kB]
    Get:2 http://archive.ubuntu.com/ubuntu bionic/universe amd64 tesseract-ocr-osd all 4.00~git24-0e00fe6-1.2 [2,989 kB]
    Get:3 http://archive.ubuntu.com/ubuntu bionic/universe amd64 tesseract-ocr amd64 4.00~git2288-10f4998a-2 [218 kB]
    Fetched 4,795 kB in 1s (3,752 kB/s)
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 76, <> line 3.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Selecting previously unselected package tesseract-ocr-eng.
    (Reading database ... 155202 files and directories currently installed.)
    Preparing to unpack .../tesseract-ocr-eng_4.00~git24-0e00fe6-1.2_all.deb ...
    Unpacking tesseract-ocr-eng (4.00~git24-0e00fe6-1.2) ...
    Selecting previously unselected package tesseract-ocr-osd.
    Preparing to unpack .../tesseract-ocr-osd_4.00~git24-0e00fe6-1.2_all.deb ...
    Unpacking tesseract-ocr-osd (4.00~git24-0e00fe6-1.2) ...
    Selecting previously unselected package tesseract-ocr.
    Preparing to unpack .../tesseract-ocr_4.00~git2288-10f4998a-2_amd64.deb ...
    Unpacking tesseract-ocr (4.00~git2288-10f4998a-2) ...
    Setting up tesseract-ocr-osd (4.00~git24-0e00fe6-1.2) ...
    Setting up tesseract-ocr-eng (4.00~git24-0e00fe6-1.2) ...
    Setting up tesseract-ocr (4.00~git2288-10f4998a-2) ...
    Processing triggers for man-db (2.8.3-2ubuntu0.1) ...
    [K     |████████████████████████████████| 4.3 MB 5.4 MB/s 
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    albumentations 0.1.12 requires imgaug<0.2.7,>=0.2.5, but you have imgaug 0.2.9 which is incompatible.[0m
    [?25h


```python
!pip install Pillow==9.0.0
```

    Collecting Pillow==9.0.0
      Downloading Pillow-9.0.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.3 MB)
    [K     |████████████████████████████████| 4.3 MB 5.3 MB/s 
    [?25hInstalling collected packages: Pillow
      Attempting uninstall: Pillow
        Found existing installation: Pillow 9.1.0
        Uninstalling Pillow-9.1.0:
          Successfully uninstalled Pillow-9.1.0
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    albumentations 0.1.12 requires imgaug<0.2.7,>=0.2.5, but you have imgaug 0.2.9 which is incompatible.[0m
    Successfully installed Pillow-9.0.0




# DocVQA Demo


```python
from transformers import LayoutLMv2Processor
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```


    Downloading:   0%|          | 0.00/135 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/226k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/707 [00:00<?, ?B/s]



```python
from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained("tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa")
```


    Downloading:   0%|          | 0.00/2.69k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/765M [00:00<?, ?B/s]



```python
import torch
device = torch.device("cuda")
model.to(device)
```


```python
def run_qa(image, question):
  # step 1: encoding
  encoding = processor(image, question, return_tensors="pt", truncation=True)

  # step 2: forward pass

  for k,v in encoding.items():
    encoding[k] = v.to(model.device)

  outputs = model(**encoding)

  # step 3: get start_logits and end_logits
  start_logits = outputs.start_logits
  end_logits = outputs.end_logits

  # step 4: get largest logit for both
  predicted_start_idx = start_logits.argmax(-1).item()
  predicted_end_idx = end_logits.argmax(-1).item()
  print("Predicted start idx:", predicted_start_idx)
  print("Predicted end idx:", predicted_end_idx)

  # step 5: decode the predicted answer
  return processor.tokenizer.decode(encoding.input_ids.squeeze()[predicted_start_idx:predicted_end_idx+1])

```


```python
from PIL import Image
image = Image.open("image-2.jpeg").convert("RGB")
image
```




    
![png](Doc_Visual_QA_and_Bill_extraction_demo_files/Doc_Visual_QA_and_Bill_extraction_demo_15_0.png)
    




```python
question = "Where to call?"
run_qa(image, question)
```

    Predicted start idx: 77
    Predicted end idx: 82





    'ext. 7240.'



# Bill Information extraction Demo


```python
import numpy as np
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
```


```python
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForTokenClassification.from_pretrained("Theivaprakasham/layoutlmv2-finetuned-sroie")
```


    Downloading:   0%|          | 0.00/3.07k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/765M [00:00<?, ?B/s]



```python
dataset = load_dataset("darentang/sroie", split="test")
```


    Downloading builder script:   0%|          | 0.00/4.25k [00:00<?, ?B/s]


    Downloading and preparing dataset sroie/sroie to /root/.cache/huggingface/datasets/darentang___sroie/sroie/1.0.0/26ed9374c9a15a1d2f44fd8886f679076e1a1fd7da2d53726d6e58a99436c506...



    Downloading data files:   0%|          | 0/1 [00:00<?, ?it/s]



    Downloading data:   0%|          | 0.00/456M [00:00<?, ?B/s]



    Extracting data files:   0%|          | 0/1 [00:00<?, ?it/s]



    Generating train split: 0 examples [00:00, ? examples/s]



    Generating test split: 0 examples [00:00, ? examples/s]


    Dataset sroie downloaded and prepared to /root/.cache/huggingface/datasets/darentang___sroie/sroie/1.0.0/26ed9374c9a15a1d2f44fd8886f679076e1a1fd7da2d53726d6e58a99436c506. Subsequent calls will reuse this data.



```python
Image.open(dataset[50]["image_path"]).convert("RGB").save("example1.png")
```


```python
ls
```

    example1.png  image-2.jpeg  [0m[01;34msample_data[0m/  train.tar.gz  train.tar.gz.1



```python
Image.open(dataset[100]["image_path"]).convert("RGB").save("example2.png")
```


```python
# define id2label, label2color
labels = dataset.features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
label2color = {'B-ADDRESS': 'blue',
 'B-COMPANY': 'green',
 'B-DATE': 'red',
 'B-TOTAL': 'red',
 'I-ADDRESS': "blue",
 'I-COMPANY': 'green',
 'I-DATE': 'red',
 'I-TOTAL': 'red',
 'O': 'green'}

label2color = dict((k.lower(), v.lower()) for k,v in label2color.items())

```


```python
def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def iob_to_label(label):
    return label
```


```python
image = Image.open("example2.png").convert("RGB")
image
```




    
![png](Doc_Visual_QA_and_Bill_extraction_demo_files/Doc_Visual_QA_and_Bill_extraction_demo_26_0.png)
    




```python
width, height = image.size
width, height
```




    (932, 2216)




```python
encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
offset_mapping = encoding.pop('offset_mapping')
```


```python
outputs = model(**encoding)
```


```python
# get predictions
predictions = outputs.logits.argmax(-1).squeeze().tolist()
token_boxes = encoding.bbox.squeeze().tolist()
```


```python
# only keep non-subword predictions
is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

```


```python
true_boxes
```




    [[0.0, 0.0, 0.0, 0.0],
     [430.584, 26.592000000000002, 523.7840000000001, 90.85600000000001],
     [680.36, 33.24, 722.3000000000001, 79.776],
     [163.1, 259.272, 296.376, 294.728],
     [315.94800000000004, 261.488, 462.272, 294.728],
     [480.91200000000003, 261.488, 581.568, 296.944],
     [596.48, 261.488, 752.124, 296.944],
     [335.52, 301.37600000000003, 582.5, 334.616],
     [258.16400000000004, 352.344, 319.67600000000004, 378.93600000000004],
     [332.724, 352.344, 386.78, 378.93600000000004],
     [398.896, 354.56, 499.552, 378.93600000000004],
     [514.464, 354.56, 660.788, 381.152],
     [229.272, 398.88, 323.404, 427.688],
     [335.52, 398.88, 480.91200000000003, 427.688],
     [501.41600000000005, 401.096, 685.952, 429.904],
     [144.46, 474.224, 203.176, 500.81600000000003],
     [217.156, 480.872, 219.952, 483.088],
     [233.0, 474.224, 433.38, 503.03200000000004],
     [481.844, 476.44, 543.356, 503.03200000000004],
     [556.404, 500.81600000000003, 558.268, 500.81600000000003],
     [571.316, 476.44, 772.6279999999999, 505.248],
     [275.872, 525.192, 342.976, 551.784],
     [355.092, 525.192, 385.84799999999996, 554.0],
     [398.896, 531.84, 400.76, 551.784],
     [414.74, 525.192, 641.2159999999999, 554.0],
     [32.620000000000005, 582.808, 876.0799999999999, 622.696],
     [46.6, 624.9119999999999, 83.88, 651.504],
     [95.996, 627.1279999999999, 147.256, 653.7199999999999],
     [45.668, 675.88, 121.16000000000001, 704.688],
     [132.344, 678.096, 227.408, 704.688],
     [239.524, 678.096, 329.928, 706.904],
     [340.18, 678.096, 553.608, 706.904],
     [45.668, 720.2, 93.2, 749.008],
     [108.11200000000001, 722.416, 117.432, 749.008],
     [133.27599999999998, 722.416, 234.864, 749.008],
     [247.912, 722.416, 341.11199999999997, 751.224],
     [354.16, 724.6320000000001, 370.93600000000004, 749.008],
     [44.736000000000004, 764.52, 158.44, 791.112],
     [172.42, 764.52, 259.096, 791.112],
     [271.212, 766.736, 364.41200000000003, 793.328],
     [45.668, 813.2719999999999, 189.19600000000003, 839.864],
     [202.244, 813.2719999999999, 387.712, 842.08],
     [44.736000000000004, 859.808, 244.184, 888.6160000000001],
     [46.6, 899.696, 169.624, 932.9359999999999],
     [285.192, 901.9119999999999, 483.708, 932.9359999999999],
     [602.072, 908.56, 677.564, 935.1519999999999],
     [703.66, 904.1279999999999, 875.1479999999999, 935.1519999999999],
     [45.668, 952.88, 161.236, 981.688],
     [259.096, 979.472, 261.89200000000005, 981.688],
     [289.852, 957.312, 378.39200000000005, 983.904],
     [598.344, 957.312, 675.6999999999999, 983.904],
     [703.66, 959.528, 833.208, 986.12],
     [47.532, 1003.8480000000001, 230.204, 1037.088],
     [259.096, 1014.928, 261.89200000000005, 1032.656],
     [610.46, 1008.2800000000001, 677.564, 1034.872],
     [58.716, 1054.816, 131.41199999999998, 1059.248],
     [527.512, 1076.9759999999999, 607.664, 1108.0],
     [700.864, 1059.248, 832.2760000000001, 1108.0],
     [32.620000000000005, 1110.216, 188.264, 1143.4560000000001],
     [189.19600000000003, 1112.432, 367.208, 1145.672],
     [369.072, 1112.432, 481.844, 1145.672],
     [520.988, 1110.216, 615.12, 1136.808],
     [0.0, 0.0, 932.0, 2216.0],
     [674.768, 1112.432, 779.1519999999999, 1136.808],
     [830.412, 1116.864, 877.944, 1150.104],
     [47.532, 1150.104, 139.79999999999998, 1174.48],
     [305.696, 1156.752, 312.22, 1176.6960000000001],
     [399.828, 1154.536, 470.66, 1178.912],
     [531.24, 1154.536, 602.072, 1178.912],
     [711.116, 1156.752, 782.88, 1181.1280000000002],
     [841.596, 1156.752, 877.944, 1181.1280000000002],
     [46.6, 1198.856, 127.68400000000001, 1223.2320000000002],
     [140.732, 1198.856, 230.204, 1223.2320000000002],
     [241.388, 1198.856, 280.532, 1227.6640000000002],
     [49.396, 1252.04, 138.868, 1274.1999999999998],
     [303.832, 1252.04, 315.94800000000004, 1276.416],
     [408.216, 1254.2559999999999, 462.272, 1276.416],
     [540.56, 1254.2559999999999, 595.548, 1278.6319999999998],
     [711.116, 1256.472, 740.94, 1280.848],
     [752.124, 1256.472, 781.948, 1280.848],
     [842.528, 1258.6879999999999, 877.944, 1280.848],
     [45.668, 1298.576, 118.364, 1322.952],
     [122.092, 1298.576, 194.78799999999998, 1322.952],
     [207.836, 1300.792, 259.096, 1325.168],
     [31.688000000000002, 1338.464, 59.648, 1342.896],
     [62.444, 1340.68, 166.828, 1373.92],
     [165.896, 1340.68, 229.272, 1380.568],
     [238.592, 1325.168, 876.0799999999999, 1378.352],
     [165.896, 1424.8880000000001, 235.796, 1451.48],
     [247.912, 1427.104, 327.132, 1453.6960000000001],
     [339.248, 1427.104, 488.368, 1462.5600000000002],
     [500.48400000000004, 1429.32, 578.772, 1460.344],
     [607.664, 1435.968, 610.46, 1455.912],
     [703.66, 1429.32, 786.608, 1455.912],
     [451.08799999999997, 1486.9360000000001, 579.704, 1513.528],
     [607.664, 1493.584, 610.46, 1513.528],
     [722.3000000000001, 1486.9360000000001, 786.608, 1515.7440000000001],
     [431.516, 1544.552, 501.41600000000005, 1571.144],
     [513.532, 1546.7679999999998, 579.704, 1571.144],
     [607.664, 1553.416, 610.46, 1571.144],
     [722.3000000000001, 1548.984, 785.6759999999999, 1573.36],
     [442.7, 1599.952, 578.772, 1635.408],
     [607.664, 1608.816, 610.46, 1628.76],
     [722.3000000000001, 1602.168, 785.6759999999999, 1628.76],
     [99.724, 1657.568, 179.876, 1684.16],
     [192.92399999999998, 1657.568, 279.59999999999997, 1684.16],
     [290.784, 1659.784, 448.292, 1690.808],
     [459.476, 1662.0, 490.232, 1686.376],
     [501.41600000000005, 1662.0, 580.636, 1695.24],
     [607.664, 1668.648, 611.392, 1688.592],
     [702.728, 1662.0, 785.6759999999999, 1690.808],
     [484.64000000000004, 1719.616, 554.54, 1746.208],
     [558.268, 1719.616, 611.392, 1746.208],
     [702.728, 1721.832, 785.6759999999999, 1748.424],
     [453.884, 1763.9360000000001, 558.268, 1797.1760000000002],
     [562.928, 1770.584, 611.392, 1790.528],
     [721.368, 1763.9360000000001, 785.6759999999999, 1792.7440000000001],
     [101.588, 1854.792, 169.624, 1890.248],
     [181.74, 1863.656, 355.092, 1892.464],
     [528.444, 1859.224, 594.616, 1863.656],
     [107.18, 1919.056, 169.624, 1950.08],
     [174.284, 1919.056, 258.16400000000004, 1952.296],
     [355.092, 1921.272, 381.188, 1954.512],
     [471.592, 1923.488, 620.712, 1956.728],
     [0.0, 0.0, 932.0, 2216.0],
     [697.136, 1923.488, 839.732, 1958.944],
     [110.908, 1963.376, 153.78, 1989.968],
     [359.752, 1965.592, 375.596, 1989.968],
     [537.764, 1967.808, 619.7800000000001, 1992.184],
     [775.424, 1967.808, 838.8000000000001, 1994.4],
     [298.24, 2012.1280000000002, 378.39200000000005, 2040.9360000000001],
     [393.304, 2020.992, 397.964, 2040.9360000000001],
     [534.968, 2014.344, 616.984, 2040.9360000000001],
     [775.424, 2016.5600000000002, 839.732, 2043.152],
     [94.132, 2098.5519999999997, 202.244, 2122.928],
     [214.36, 2098.5519999999997, 290.784, 2122.928],
     [300.104, 2098.5519999999997, 358.82, 2122.928],
     [369.072, 2100.768, 427.788, 2122.928],
     [439.904, 2100.768, 633.76, 2127.36],
     [645.876, 2102.984, 738.144, 2125.144],
     [749.3280000000001, 2102.984, 816.432, 2125.144],
     [932.0, 2216.0, 932.0, 2216.0]]




```python
true_predictions
```




    ['O',
     'O',
     'O',
     'B-COMPANY',
     'I-COMPANY',
     'I-COMPANY',
     'I-COMPANY',
     'O',
     'B-ADDRESS',
     'I-ADDRESS',
     'I-ADDRESS',
     'I-ADDRESS',
     'I-ADDRESS',
     'I-ADDRESS',
     'I-ADDRESS',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'B-DATE',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'B-TOTAL',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O',
     'O']




```python
# draw predictions over the image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
for prediction, box in zip(true_predictions, true_boxes):
    predicted_label = iob_to_label(prediction).lower()
    draw.rectangle(box, outline=label2color[predicted_label])
    draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)

```


```python
image
```




    
![png](Doc_Visual_QA_and_Bill_extraction_demo_files/Doc_Visual_QA_and_Bill_extraction_demo_35_0.png)
    




```python

```
