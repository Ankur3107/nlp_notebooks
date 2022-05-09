<a href="https://colab.research.google.com/github/Ankur3107/colab_notebooks/blob/master/contextual_topic_modeling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```
!pip install contextualized_topic_models
```

    Collecting contextualized_topic_models
      Downloading https://files.pythonhosted.org/packages/e8/83/16ee302ae54b21692ac995d71ae90265473becfd73bcf942c41c0a9b5950/contextualized_topic_models-1.4.2-py2.py3-none-any.whl
    Requirement already satisfied: torch==1.6.0 in /usr/local/lib/python3.6/dist-packages (from contextualized_topic_models) (1.6.0+cu101)
    Collecting gensim==3.8.3
    [?25l  Downloading https://files.pythonhosted.org/packages/2b/e0/fa6326251692056dc880a64eb22117e03269906ba55a6864864d24ec8b4e/gensim-3.8.3-cp36-cp36m-manylinux1_x86_64.whl (24.2MB)
    [K     |████████████████████████████████| 24.2MB 130kB/s 
    [?25hCollecting wheel==0.33.6
      Downloading https://files.pythonhosted.org/packages/00/83/b4a77d044e78ad1a45610eb88f745be2fd2c6d658f9798a15e384b7d57c9/wheel-0.33.6-py2.py3-none-any.whl
    Collecting pytest-runner==5.1
      Downloading https://files.pythonhosted.org/packages/f8/31/f291d04843523406f242e63b5b90f7b204a756169b4250ff213e10326deb/pytest_runner-5.1-py2.py3-none-any.whl
    Requirement already satisfied: torchvision==0.7.0 in /usr/local/lib/python3.6/dist-packages (from contextualized_topic_models) (0.7.0+cu101)
    Collecting sentence-transformers==0.3.2
    [?25l  Downloading https://files.pythonhosted.org/packages/78/e0/65ad8fd86eba720412d9ff102c4a3540a113bbc6cd29b01e7ecc33ebb1fa/sentence-transformers-0.3.2.tar.gz (65kB)
    [K     |████████████████████████████████| 71kB 9.6MB/s 
    [?25hCollecting numpy==1.19.1
    [?25l  Downloading https://files.pythonhosted.org/packages/b1/9a/7d474ba0860a41f771c9523d8c4ea56b084840b5ca4092d96bdee8a3b684/numpy-1.19.1-cp36-cp36m-manylinux2010_x86_64.whl (14.5MB)
    [K     |████████████████████████████████| 14.5MB 238kB/s 
    [?25hCollecting pytest==4.6.5
    [?25l  Downloading https://files.pythonhosted.org/packages/97/72/d4d6d22ad216f149685f2e93ce9280df9f36cbbc87fa546641ac92f22766/pytest-4.6.5-py2.py3-none-any.whl (230kB)
    [K     |████████████████████████████████| 235kB 59.2MB/s 
    [?25hRequirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from torch==1.6.0->contextualized_topic_models) (0.16.0)
    Requirement already satisfied: smart-open>=1.8.1 in /usr/local/lib/python3.6/dist-packages (from gensim==3.8.3->contextualized_topic_models) (2.1.0)
    Requirement already satisfied: six>=1.5.0 in /usr/local/lib/python3.6/dist-packages (from gensim==3.8.3->contextualized_topic_models) (1.15.0)
    Requirement already satisfied: scipy>=0.18.1 in /usr/local/lib/python3.6/dist-packages (from gensim==3.8.3->contextualized_topic_models) (1.4.1)
    Requirement already satisfied: pillow>=4.1.1 in /usr/local/lib/python3.6/dist-packages (from torchvision==0.7.0->contextualized_topic_models) (7.0.0)
    Collecting transformers>=3.0.2
    [?25l  Downloading https://files.pythonhosted.org/packages/27/3c/91ed8f5c4e7ef3227b4119200fc0ed4b4fd965b1f0172021c25701087825/transformers-3.0.2-py3-none-any.whl (769kB)
    [K     |████████████████████████████████| 778kB 48.0MB/s 
    [?25hRequirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from sentence-transformers==0.3.2->contextualized_topic_models) (4.41.1)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from sentence-transformers==0.3.2->contextualized_topic_models) (0.22.2.post1)
    Requirement already satisfied: nltk in /usr/local/lib/python3.6/dist-packages (from sentence-transformers==0.3.2->contextualized_topic_models) (3.2.5)
    Requirement already satisfied: atomicwrites>=1.0 in /usr/local/lib/python3.6/dist-packages (from pytest==4.6.5->contextualized_topic_models) (1.4.0)
    Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.6/dist-packages (from pytest==4.6.5->contextualized_topic_models) (20.1.0)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.6/dist-packages (from pytest==4.6.5->contextualized_topic_models) (0.2.5)
    Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from pytest==4.6.5->contextualized_topic_models) (20.4)
    Requirement already satisfied: py>=1.5.0 in /usr/local/lib/python3.6/dist-packages (from pytest==4.6.5->contextualized_topic_models) (1.9.0)
    Collecting pluggy<1.0,>=0.12
      Downloading https://files.pythonhosted.org/packages/a0/28/85c7aa31b80d150b772fbe4a229487bc6644da9ccb7e427dd8cc60cb8a62/pluggy-0.13.1-py2.py3-none-any.whl
    Requirement already satisfied: importlib-metadata>=0.12 in /usr/local/lib/python3.6/dist-packages (from pytest==4.6.5->contextualized_topic_models) (1.7.0)
    Requirement already satisfied: more-itertools>=4.0.0; python_version > "2.7" in /usr/local/lib/python3.6/dist-packages (from pytest==4.6.5->contextualized_topic_models) (8.4.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (2.23.0)
    Requirement already satisfied: boto in /usr/local/lib/python3.6/dist-packages (from smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (2.49.0)
    Requirement already satisfied: boto3 in /usr/local/lib/python3.6/dist-packages (from smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (1.14.48)
    Requirement already satisfied: filelock in /usr/local/lib/python3.6/dist-packages (from transformers>=3.0.2->sentence-transformers==0.3.2->contextualized_topic_models) (3.0.12)
    Collecting sacremoses
    [?25l  Downloading https://files.pythonhosted.org/packages/7d/34/09d19aff26edcc8eb2a01bed8e98f13a1537005d31e95233fd48216eed10/sacremoses-0.0.43.tar.gz (883kB)
    [K     |████████████████████████████████| 890kB 49.3MB/s 
    [?25hRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.6/dist-packages (from transformers>=3.0.2->sentence-transformers==0.3.2->contextualized_topic_models) (2019.12.20)
    Requirement already satisfied: dataclasses; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from transformers>=3.0.2->sentence-transformers==0.3.2->contextualized_topic_models) (0.7)
    Collecting tokenizers==0.8.1.rc1
    [?25l  Downloading https://files.pythonhosted.org/packages/40/d0/30d5f8d221a0ed981a186c8eb986ce1c94e3a6e87f994eae9f4aa5250217/tokenizers-0.8.1rc1-cp36-cp36m-manylinux1_x86_64.whl (3.0MB)
    [K     |████████████████████████████████| 3.0MB 48.4MB/s 
    [?25hCollecting sentencepiece!=0.1.92
    [?25l  Downloading https://files.pythonhosted.org/packages/d4/a4/d0a884c4300004a78cca907a6ff9a5e9fe4f090f5d95ab341c53d28cbc58/sentencepiece-0.1.91-cp36-cp36m-manylinux1_x86_64.whl (1.1MB)
    [K     |████████████████████████████████| 1.1MB 48.1MB/s 
    [?25hRequirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->sentence-transformers==0.3.2->contextualized_topic_models) (0.16.0)
    Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from packaging->pytest==4.6.5->contextualized_topic_models) (2.4.7)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata>=0.12->pytest==4.6.5->contextualized_topic_models) (3.1.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (2020.6.20)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.6/dist-packages (from boto3->smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (0.10.0)
    Requirement already satisfied: s3transfer<0.4.0,>=0.3.0 in /usr/local/lib/python3.6/dist-packages (from boto3->smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (0.3.3)
    Requirement already satisfied: botocore<1.18.0,>=1.17.48 in /usr/local/lib/python3.6/dist-packages (from boto3->smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (1.17.48)
    Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers>=3.0.2->sentence-transformers==0.3.2->contextualized_topic_models) (7.1.2)
    Requirement already satisfied: docutils<0.16,>=0.10 in /usr/local/lib/python3.6/dist-packages (from botocore<1.18.0,>=1.17.48->boto3->smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (0.15.2)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /usr/local/lib/python3.6/dist-packages (from botocore<1.18.0,>=1.17.48->boto3->smart-open>=1.8.1->gensim==3.8.3->contextualized_topic_models) (2.8.1)
    Building wheels for collected packages: sentence-transformers, sacremoses
      Building wheel for sentence-transformers (setup.py) ... [?25l[?25hdone
      Created wheel for sentence-transformers: filename=sentence_transformers-0.3.2-cp36-none-any.whl size=93965 sha256=f16e69cf4e046025ab7a2d6862502cf2e431b7cfd90def10046f9bba45bf6abe
      Stored in directory: /root/.cache/pip/wheels/f7/06/a0/567f3651876165429f6510d3197b011652a25e547552816824
      Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
      Created wheel for sacremoses: filename=sacremoses-0.0.43-cp36-none-any.whl size=893257 sha256=936a3c608102bf3d7950dac3afb5db63605b8a340cd539e4390c9c3031cd29d8
      Stored in directory: /root/.cache/pip/wheels/29/3c/fd/7ce5c3f0666dab31a50123635e6fb5e19ceb42ce38d4e58f45
    Successfully built sentence-transformers sacremoses
    [31mERROR: tensorflow 2.3.0 has requirement numpy<1.19.0,>=1.16.0, but you'll have numpy 1.19.1 which is incompatible.[0m
    [31mERROR: datascience 0.10.6 has requirement folium==0.2.1, but you'll have folium 0.8.3 which is incompatible.[0m
    [31mERROR: albumentations 0.1.12 has requirement imgaug<0.2.7,>=0.2.5, but you'll have imgaug 0.2.9 which is incompatible.[0m
    Installing collected packages: numpy, gensim, wheel, pytest-runner, sacremoses, tokenizers, sentencepiece, transformers, sentence-transformers, pluggy, pytest, contextualized-topic-models
      Found existing installation: numpy 1.18.5
        Uninstalling numpy-1.18.5:
          Successfully uninstalled numpy-1.18.5
      Found existing installation: gensim 3.6.0
        Uninstalling gensim-3.6.0:
          Successfully uninstalled gensim-3.6.0
      Found existing installation: wheel 0.35.1
        Uninstalling wheel-0.35.1:
          Successfully uninstalled wheel-0.35.1
      Found existing installation: pluggy 0.7.1
        Uninstalling pluggy-0.7.1:
          Successfully uninstalled pluggy-0.7.1
      Found existing installation: pytest 3.6.4
        Uninstalling pytest-3.6.4:
          Successfully uninstalled pytest-3.6.4
    Successfully installed contextualized-topic-models-1.4.2 gensim-3.8.3 numpy-1.19.1 pluggy-0.13.1 pytest-4.6.5 pytest-runner-5.1 sacremoses-0.0.43 sentence-transformers-0.3.2 sentencepiece-0.1.91 tokenizers-0.8.1rc1 transformers-3.0.2 wheel-0.33.6





```
!git clone https://github.com/tdhopper/topic-modeling-datasets.git
```

    Cloning into 'topic-modeling-datasets'...
    remote: Enumerating objects: 18915, done.[K
    remote: Counting objects: 100% (18915/18915), done.[K
    remote: Compressing objects: 100% (18875/18875), done.[K
    remote: Total 18915 (delta 37), reused 18915 (delta 37), pack-reused 0[K
    Receiving objects: 100% (18915/18915), 29.69 MiB | 16.43 MiB/s, done.
    Resolving deltas: 100% (37/37), done.



```
from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.utils.data_preparation import TextHandler
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.datasets.dataset import CTMDataset
```


```
!head -n 10 topic-modeling-datasets/data/lda-c/blei-ap/ap.txt
```

    <DOC>
    <DOCNO> AP881218-0003 </DOCNO>
    <TEXT>
     A 16-year-old student at a private Baptist school who allegedly killed one teacher and wounded another before firing into a filled classroom apparently ``just snapped,'' the school's pastor said. ``I don't know how it could have happened,'' said George Sweet, pastor of Atlantic Shores Baptist Church. ``This is a good, Christian school. We pride ourselves on discipline. Our kids are good kids.'' The Atlantic Shores Christian School sophomore was arrested and charged with first-degree murder, attempted murder, malicious assault and related felony charges for the Friday morning shooting. Police would not release the boy's name because he is a juvenile, but neighbors and relatives identified him as Nicholas Elliott. Police said the student was tackled by a teacher and other students when his semiautomatic pistol jammed as he fired on the classroom as the students cowered on the floor crying ``Jesus save us! God save us!'' Friends and family said the boy apparently was troubled by his grandmother's death and the divorce of his parents and had been tormented by classmates. Nicholas' grandfather, Clarence Elliott Sr., said Saturday that the boy's parents separated about four years ago and his maternal grandmother, Channey Williams, died last year after a long illness. The grandfather also said his grandson was fascinated with guns. ``The boy was always talking about guns,'' he said. ``He knew a lot about them. He knew all the names of them _ none of those little guns like a .32 or a .22 or nothing like that. He liked the big ones.'' The slain teacher was identified as Karen H. Farley, 40. The wounded teacher, 37-year-old Sam Marino, was in serious condition Saturday with gunshot wounds in the shoulder. Police said the boy also shot at a third teacher, Susan Allen, 31, as she fled from the room where Marino was shot. He then shot Marino again before running to a third classroom where a Bible class was meeting. The youngster shot the glass out of a locked door before opening fire, police spokesman Lewis Thurston said. When the youth's pistol jammed, he was tackled by teacher Maurice Matteson, 24, and other students, Thurston said. ``Once you see what went on in there, it's a miracle that we didn't have more people killed,'' Police Chief Charles R. Wall said. Police didn't have a motive, Detective Tom Zucaro said, but believe the boy's primary target was not a teacher but a classmate. Officers found what appeared to be three Molotov cocktails in the boy's locker and confiscated the gun and several spent shell casings. Fourteen rounds were fired before the gun jammed, Thurston said. The gun, which the boy carried to school in his knapsack, was purchased by an adult at the youngster's request, Thurston said, adding that authorities have interviewed the adult, whose name is being withheld pending an investigation by the federal Bureau of Alcohol, Tobacco and Firearms. The shootings occurred in a complex of four portable classrooms for junior and senior high school students outside the main building of the 4-year-old school. The school has 500 students in kindergarten through 12th grade. Police said they were trying to reconstruct the sequence of events and had not resolved who was shot first. The body of Ms. Farley was found about an hour after the shootings behind a classroom door.
     </TEXT>
    </DOC>
    <DOC>
    <DOCNO> AP880224-0195 </DOCNO>
    <TEXT>
     The Bechtel Group Inc. offered in 1985 to sell oil to Israel at a discount of at least $650 million for 10 years if it promised not to bomb a proposed Iraqi pipeline, a Foreign Ministry official said Wednesday. But then-Prime Minister Shimon Peres said the offer from Bruce Rappaport, a partner in the San Francisco-based construction and engineering company, was ``unimportant,'' the senior official told The Associated Press. Peres, now foreign minister, never discussed the offer with other government ministers, said the official, who spoke on condition of anonymity. The comments marked the first time Israel has acknowledged any offer was made for assurances not to bomb the planned $1 billion pipeline, which was to have run near Israel's border with Jordan. The pipeline was never built. In San Francisco, Tom Flynn, vice president for public relations for the Bechtel Group, said the company did not make any offer to Peres but that Rappaport, a Swiss financier, made it without Bechtel's knowledge or consent. Another Bechtel spokesman, Al Donner, said Bechtel ``at no point'' in development of the pipeline project had anything to do with the handling of the oil. He said proposals submitted by the company ``did not include any specific arrangements for the handling of the oil or for the disposal of the oil once it reached the terminal.'' Asked about Bechtel's disclaimers after they were made in San Francisco, the Israeli Foreign Ministry official said Peres believed Rappaport made the offer for the company. ``Rappaport came to Peres as a representative of Bechtel and said he was speaking on behalf of Bechtel,'' the official said. ``If he was not, he misrepresented himself.'' The Jerusalem Post on Wednesday quoted sources close to Peres as saying that according to Rappaport, Bechtel had said the oil sales would have to be conducted through a third party to keep the sales secret from Iraq and Jordan. The Foreign Ministry official said Peres did not take the offer seriously. ``This is a man who sees 10 people every day,'' he said. ``Thirty percent of them come with crazy ideas. He just says, `Yes, yes. We'll think about it.' That's how things work in Israel.'' The offer appeared to be the one mentioned in a September 1985 memo to Attorney General Edwin Meese III. The memo referred to an arrangement between Peres and Rappaport ``to the effect that Israel will receive somewhere between $65 million and $70 million a year for 10 years.'' The memo from Meese friend E. Robert Wallach, Rappaport's attorney, also states, ``What was also indicated to me, and which would be denied everywhere, is that a portion of those funds will go directly to Labor,'' a reference to the political party Peres leads. The Wallach memo has become the focus of an investigation into whether Meese knew of a possibly improper payment. Peres has denied any wrongdoing and has denounced the memo as ``complete nonsense.'' The Israeli official said Rappaport, a native of Israel and a close friend of Peres, relayed the offer to Peres earlier in September. ``Peres thought the offer was unimportant. For him, the most important thing was to have an Iraqi oil port near Israel's border,'' the official said. ``The thinking was that this would put Iraq in a position where it would not be able to wage war with Israel, out of concern for its pipeline.'' A person answering the telephone at Rappaport's Swiss residence said he was out of town and could not be reached for comment.



```
input_file = 'topic-modeling-datasets/data/lda-c/blei-ap/ap.txt'

handler = TextHandler(input_file)
handler.prepare() # create vocabulary and training data
```


```
bert_embeddings_from_file?
```


```
training_bert = bert_embeddings_from_file(input_file, "distilbert-base-nli-mean-tokens")
```

    100%|██████████| 245M/245M [00:03<00:00, 69.3MB/s]



```
training_dataset = CTMDataset(handler.bow, training_bert, handler.idx2token)
```


```
ctm = CTM(input_size=len(handler.vocab), bert_input_size=768, num_epochs=5, inference_type="combined", n_components=50)
```


```
ctm.fit(training_dataset) # run the model
```

    Settings: 
                   N Components: 50
                   Topic Prior Mean: 0.0
                   Topic Prior Variance: 0.98
                   Model Type: prodLDA
                   Hidden Sizes: (100, 100)
                   Activation: softplus
                   Dropout: 0.2
                   Learn Priors: True
                   Learning Rate: 0.002
                   Momentum: 0.99
                   Reduce On Plateau: False
                   Save Dir: None
    Epoch: [1/5]	Samples: [13500/67500]	Train Loss: 675.9943767361111	Time: 0:00:14.059041
    Epoch: [2/5]	Samples: [27000/67500]	Train Loss: 650.9020456090857	Time: 0:00:14.110983
    Epoch: [3/5]	Samples: [40500/67500]	Train Loss: 647.2234633246528	Time: 0:00:14.079987
    Epoch: [4/5]	Samples: [54000/67500]	Train Loss: 645.6622060908564	Time: 0:00:14.050282
    Epoch: [5/5]	Samples: [67500/67500]	Train Loss: 643.8000949074074	Time: 0:00:14.067917



```
ctm.get_topic_lists(5)[0:5]
```




    [['younger', 'products.', 'Street,', 'pledged', 'claims.'],
     ['A', 'million', 'was', 'said.', 'The'],
     ['2,', 'section', 'Seattle', 'primary.', 'link'],
     ['to', 'that', 'the', 'is', 'be'],
     ['Association,', 'pledged', 'participate', '0.2', 'drugs,']]



# Prediction


```
!head -n 10 topic-modeling-datasets/data/lda-c/blei-ap/ap.txt
```

    <DOC>
    <DOCNO> AP881218-0003 </DOCNO>
    <TEXT>
     A 16-year-old student at a private Baptist school who allegedly killed one teacher and wounded another before firing into a filled classroom apparently ``just snapped,'' the school's pastor said. ``I don't know how it could have happened,'' said George Sweet, pastor of Atlantic Shores Baptist Church. ``This is a good, Christian school. We pride ourselves on discipline. Our kids are good kids.'' The Atlantic Shores Christian School sophomore was arrested and charged with first-degree murder, attempted murder, malicious assault and related felony charges for the Friday morning shooting. Police would not release the boy's name because he is a juvenile, but neighbors and relatives identified him as Nicholas Elliott. Police said the student was tackled by a teacher and other students when his semiautomatic pistol jammed as he fired on the classroom as the students cowered on the floor crying ``Jesus save us! God save us!'' Friends and family said the boy apparently was troubled by his grandmother's death and the divorce of his parents and had been tormented by classmates. Nicholas' grandfather, Clarence Elliott Sr., said Saturday that the boy's parents separated about four years ago and his maternal grandmother, Channey Williams, died last year after a long illness. The grandfather also said his grandson was fascinated with guns. ``The boy was always talking about guns,'' he said. ``He knew a lot about them. He knew all the names of them _ none of those little guns like a .32 or a .22 or nothing like that. He liked the big ones.'' The slain teacher was identified as Karen H. Farley, 40. The wounded teacher, 37-year-old Sam Marino, was in serious condition Saturday with gunshot wounds in the shoulder. Police said the boy also shot at a third teacher, Susan Allen, 31, as she fled from the room where Marino was shot. He then shot Marino again before running to a third classroom where a Bible class was meeting. The youngster shot the glass out of a locked door before opening fire, police spokesman Lewis Thurston said. When the youth's pistol jammed, he was tackled by teacher Maurice Matteson, 24, and other students, Thurston said. ``Once you see what went on in there, it's a miracle that we didn't have more people killed,'' Police Chief Charles R. Wall said. Police didn't have a motive, Detective Tom Zucaro said, but believe the boy's primary target was not a teacher but a classmate. Officers found what appeared to be three Molotov cocktails in the boy's locker and confiscated the gun and several spent shell casings. Fourteen rounds were fired before the gun jammed, Thurston said. The gun, which the boy carried to school in his knapsack, was purchased by an adult at the youngster's request, Thurston said, adding that authorities have interviewed the adult, whose name is being withheld pending an investigation by the federal Bureau of Alcohol, Tobacco and Firearms. The shootings occurred in a complex of four portable classrooms for junior and senior high school students outside the main building of the 4-year-old school. The school has 500 students in kindergarten through 12th grade. Police said they were trying to reconstruct the sequence of events and had not resolved who was shot first. The body of Ms. Farley was found about an hour after the shootings behind a classroom door.
     </TEXT>
    </DOC>
    <DOC>
    <DOCNO> AP880224-0195 </DOCNO>
    <TEXT>
     The Bechtel Group Inc. offered in 1985 to sell oil to Israel at a discount of at least $650 million for 10 years if it promised not to bomb a proposed Iraqi pipeline, a Foreign Ministry official said Wednesday. But then-Prime Minister Shimon Peres said the offer from Bruce Rappaport, a partner in the San Francisco-based construction and engineering company, was ``unimportant,'' the senior official told The Associated Press. Peres, now foreign minister, never discussed the offer with other government ministers, said the official, who spoke on condition of anonymity. The comments marked the first time Israel has acknowledged any offer was made for assurances not to bomb the planned $1 billion pipeline, which was to have run near Israel's border with Jordan. The pipeline was never built. In San Francisco, Tom Flynn, vice president for public relations for the Bechtel Group, said the company did not make any offer to Peres but that Rappaport, a Swiss financier, made it without Bechtel's knowledge or consent. Another Bechtel spokesman, Al Donner, said Bechtel ``at no point'' in development of the pipeline project had anything to do with the handling of the oil. He said proposals submitted by the company ``did not include any specific arrangements for the handling of the oil or for the disposal of the oil once it reached the terminal.'' Asked about Bechtel's disclaimers after they were made in San Francisco, the Israeli Foreign Ministry official said Peres believed Rappaport made the offer for the company. ``Rappaport came to Peres as a representative of Bechtel and said he was speaking on behalf of Bechtel,'' the official said. ``If he was not, he misrepresented himself.'' The Jerusalem Post on Wednesday quoted sources close to Peres as saying that according to Rappaport, Bechtel had said the oil sales would have to be conducted through a third party to keep the sales secret from Iraq and Jordan. The Foreign Ministry official said Peres did not take the offer seriously. ``This is a man who sees 10 people every day,'' he said. ``Thirty percent of them come with crazy ideas. He just says, `Yes, yes. We'll think about it.' That's how things work in Israel.'' The offer appeared to be the one mentioned in a September 1985 memo to Attorney General Edwin Meese III. The memo referred to an arrangement between Peres and Rappaport ``to the effect that Israel will receive somewhere between $65 million and $70 million a year for 10 years.'' The memo from Meese friend E. Robert Wallach, Rappaport's attorney, also states, ``What was also indicated to me, and which would be denied everywhere, is that a portion of those funds will go directly to Labor,'' a reference to the political party Peres leads. The Wallach memo has become the focus of an investigation into whether Meese knew of a possibly improper payment. Peres has denied any wrongdoing and has denounced the memo as ``complete nonsense.'' The Israeli official said Rappaport, a native of Israel and a close friend of Peres, relayed the offer to Peres earlier in September. ``Peres thought the offer was unimportant. For him, the most important thing was to have an Iraqi oil port near Israel's border,'' the official said. ``The thinking was that this would put Iraq in a position where it would not be able to wage war with Israel, out of concern for its pipeline.'' A person answering the telephone at Rappaport's Swiss residence said he was out of town and could not be reached for comment.



```
len(training_dataset)
```




    13500




```
import numpy as np

distribution = ctm.get_thetas(training_dataset)[100] # topic distribution for the eight document

print(distribution)

topic = np.argmax(distribution)

ctm.get_topic_lists(5)[topic]
```

    [0.01682434044778347, 0.010533287189900875, 0.0016027707606554031, 0.018512511625885963, 0.021795328706502914, 0.024250835180282593, 0.007017648313194513, 0.008357801474630833, 0.008402344770729542, 0.0027029572520405054, 0.003236175747588277, 0.01058495044708252, 0.0018672933802008629, 0.023581067100167274, 0.08781484514474869, 0.012609513476490974, 0.007289501838386059, 0.022212006151676178, 0.004904361441731453, 0.03551305830478668, 0.02438373491168022, 0.002901036525145173, 0.02808697521686554, 0.03660248965024948, 0.00808636099100113, 0.0047630425542593, 0.001469444832764566, 0.02342250943183899, 0.02564718760550022, 0.171001598238945, 0.024755068123340607, 0.004217991139739752, 0.03483238071203232, 0.013865076005458832, 0.013528377749025822, 0.06646153330802917, 0.007620108313858509, 0.020605480298399925, 0.009293865412473679, 0.00579161336645484, 0.001458485028706491, 0.003241141326725483, 0.010330692864954472, 0.013481805101037025, 0.01557138655334711, 0.026073671877384186, 0.003067850833758712, 0.018384194001555443, 0.048528604209423065, 0.0029137199744582176]





    ['agreements', 'heating', 'administrative', "governor's", 'lines.']




```

```
