<a href="https://colab.research.google.com/github/Ankur3107/nlp_notebooks/blob/master/opendomain-qa/Wikipedia_answer_retrieval_DPR.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
!pip install wikipedia transformers sentence_transformers faiss-cpu -q
```

    [K     |████████████████████████████████| 3.4 MB 36.1 MB/s 
    [K     |████████████████████████████████| 78 kB 5.8 MB/s 
    [K     |████████████████████████████████| 8.6 MB 60.3 MB/s 
    [K     |████████████████████████████████| 3.3 MB 54.2 MB/s 
    [K     |████████████████████████████████| 895 kB 66.2 MB/s 
    [K     |████████████████████████████████| 67 kB 4.6 MB/s 
    [K     |████████████████████████████████| 596 kB 64.9 MB/s 
    [K     |████████████████████████████████| 1.2 MB 54.0 MB/s 
    [?25h  Building wheel for wikipedia (setup.py) ... [?25l[?25hdone
      Building wheel for sentence-transformers (setup.py) ... [?25l[?25hdone



```python
import wikipedia
from wikipedia.exceptions import DisambiguationError
from transformers import pipeline
```


```python
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

def get_passages(text, k=100):
    tokens = text.split(" ")
    tokens_chunks = list(divide_chunks(tokens, k))
    passages = [" ".join(c) for c in tokens_chunks]
    return passages

def get_passage_for_question(question, wiki_hits=3, passage_len=100, debug=False):
  top_hits = wikipedia.search(question, wiki_hits)
  if debug:
    print("Top Wiki hits :", top_hits)
  passages = []
  for hit in top_hits:
    try:
      html_page = wikipedia.page(title = hit, auto_suggest = False)
    except DisambiguationError:
      continue
    hit_passages = get_passages(html_page.content, k=passage_len)
    passages.extend(hit_passages)

  return passages
```


```python
qa = pipeline("question-answering", model="ankur310794/roberta-base-squad2-nq")
```


    Downloading:   0%|          | 0.00/643 [00:00<?, ?B/s]


    The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can safely remove it from your `config.json` file.
    The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can safely remove it from your `config.json` file.



    Downloading:   0%|          | 0.00/473M [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/1.28k [00:00<?, ?B/s]


    The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can safely remove it from your `config.json` file.



    Downloading:   0%|          | 0.00/780k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/446k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/772 [00:00<?, ?B/s]


    The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can safely remove it from your `config.json` file.
    The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can safely remove it from your `config.json` file.



```python
from transformers import TFAutoModel, AutoTokenizer
```


```python
def combine_results(passages, k=4):
  passages_list = list(divide_chunks(passages, k))
  passages_str = [" ".join(p) for p in passages_list]
  return passages_str
```


```python
passage_encoder = TFAutoModel.from_pretrained("nlpconnect/dpr-ctx_encoder_bert_uncased_L-2_H-128_A-2")
query_encoder = TFAutoModel.from_pretrained("nlpconnect/dpr-question_encoder_bert_uncased_L-2_H-128_A-2")

p_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/dpr-ctx_encoder_bert_uncased_L-2_H-128_A-2")
q_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/dpr-question_encoder_bert_uncased_L-2_H-128_A-2")

```


    Downloading:   0%|          | 0.00/658 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/16.8M [00:00<?, ?B/s]


    All model checkpoint layers were used when initializing TFBertModel.
    
    All the layers of TFBertModel were initialized from the model checkpoint at nlpconnect/dpr-ctx_encoder_bert_uncased_L-2_H-128_A-2.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertModel for predictions without further training.



    Downloading:   0%|          | 0.00/656 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/16.8M [00:00<?, ?B/s]


    All model checkpoint layers were used when initializing TFBertModel.
    
    All the layers of TFBertModel were initialized from the model checkpoint at nlpconnect/dpr-question_encoder_bert_uncased_L-2_H-128_A-2.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertModel for predictions without further training.



    Downloading:   0%|          | 0.00/360 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/226k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/455k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/112 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/360 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/226k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/455k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/112 [00:00<?, ?B/s]



```python
import numpy as np
def extracted_passage_embeddings(processed_passages, max_length=156):
    passage_inputs = p_tokenizer.batch_encode_plus(
                    processed_passages,
                    add_special_tokens=True,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_token_type_ids=True
                )
    passage_embeddings = passage_encoder.predict([np.array(passage_inputs['input_ids']), 
                                                np.array(passage_inputs['attention_mask']), 
                                                np.array(passage_inputs['token_type_ids'])], 
                                                batch_size=1024, 
                                                verbose=1)
    return passage_embeddings

def extracted_query_embeddings(queries, max_length=64):
    query_inputs = q_tokenizer.batch_encode_plus(
                    queries,
                    add_special_tokens=True,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_token_type_ids=True
                )
    query_embeddings = query_encoder.predict([np.array(query_inputs['input_ids']), 
                                                np.array(query_inputs['attention_mask']), 
                                                np.array(query_inputs['token_type_ids'])], 
                                                batch_size=1, 
                                                verbose=1)
    return query_embeddings
```


```python
import faiss
import spacy
nlp = spacy.load("en")
```


```python
def get_answer_full_sent(m_passages, answer_dict):
  all_sents = list(nlp(m_passages).sents)
  all_sents = [s.text for s in all_sents]

  for i in range(len(all_sents)):
    if len("".join(all_sents[0:i])[answer_dict['start']:answer_dict['end']])>2:
      answer_dict['answer_sentence'] = all_sents[i-1]
      return answer_dict
  return answer_dict
```


```python
from sentence_transformers import CrossEncoder
ranking_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=196)
def get_reranked_passage(passages, question, top_rr):
  passage_question_pair = [(question, p) for p in passages]
  scores = ranking_model.predict(passage_question_pair)
  shorted_index = np.argpartition(scores, -top_rr)[::-1]
  shorted_scores = np.array([scores[i] for i in shorted_index])
  return [passages[i] for i in shorted_index[0:top_rr]]
```


```python
# end to end with dpr
topk_r=30
topk_rr=8
import pandas as pd


def get_answer_dpr(question):
  passages = get_passage_for_question(question, debug=True)
  print("Total passages: ", len(passages))
  passage_embeddings = extracted_passage_embeddings(passages)
  query_embeddings = extracted_query_embeddings([question])
  faiss_index = faiss.IndexFlatL2(128)
  faiss_index.add(passage_embeddings.pooler_output)
  prob, index = faiss_index.search(query_embeddings.pooler_output, k=topk_r)
  r_passages = [passages[i] for i in index[0]]
  print("Top k retrieved passages :", len(r_passages))
  rr_passages = get_reranked_passage(r_passages, question, topk_rr)
  print("Top k reranked passages :", len(rr_passages))
  m_passages = combine_results(rr_passages)
  print("Merged passages :", len(m_passages))
  results = qa(question=[question]*len(m_passages), context=m_passages, max_seq_len=512)
  if isinstance(results, dict):
    results = [results]
  output_results = [get_answer_full_sent(m_passages[i],results[i]) for i in range(len(results))]
  return pd.DataFrame(output_results)[['answer', 'answer_sentence', 'score']].sort_values("score", ascending=False)
```


```python
results= get_answer_dpr("where was tara located in gone with the wind?")
results
```

    Top Wiki hits : ['Tara (plantation)', 'Margaret Mitchell', 'RKO Forty Acres']
    Total passages:  95
    1/1 [==============================] - 0s 470ms/step
    1/1 [==============================] - 0s 21ms/step
    Top k retrieved passages : 30
    Top k reranked passages : 8
    Merged passages : 2


    /usr/local/lib/python3.7/dist-packages/numpy/core/_asarray.py:83: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      return array(a, dtype, copy=False, order=order)






  <div id="df-17c68291-5890-4bd1-924f-9423f8d0f7f7">
    <div class="colab-df-container">
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
      <th>answer</th>
      <th>answer_sentence</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Talmadge Farms</td>
      <td>Now the Tara facade is still located at Talmad...</td>
      <td>0.740677</td>
    </tr>
    <tr>
      <th>1</th>
      <td>virtually the same</td>
      <td>In the 2007 novel by Donald McCaig, Rhett Butl...</td>
      <td>0.159294</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-17c68291-5890-4bd1-924f-9423f8d0f7f7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-17c68291-5890-4bd1-924f-9423f8d0f7f7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-17c68291-5890-4bd1-924f-9423f8d0f7f7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
results.sort_values("score", ascending=False)
```





  <div id="df-e023e6a4-4c3f-46d9-9958-2b8b0e640726">
    <div class="colab-df-container">
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
      <th>answer</th>
      <th>answer_sentence</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Talmadge Farms</td>
      <td>Now the Tara facade is still located at Talmad...</td>
      <td>0.740677</td>
    </tr>
    <tr>
      <th>1</th>
      <td>virtually the same</td>
      <td>In the 2007 novel by Donald McCaig, Rhett Butl...</td>
      <td>0.159294</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e023e6a4-4c3f-46d9-9958-2b8b0e640726')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e023e6a4-4c3f-46d9-9958-2b8b0e640726 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e023e6a4-4c3f-46d9-9958-2b8b0e640726');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
!pip install gradio -q
```

    [K     |████████████████████████████████| 865 kB 21.2 MB/s 
    [K     |████████████████████████████████| 2.0 MB 37.0 MB/s 
    [K     |████████████████████████████████| 210 kB 50.3 MB/s 
    [K     |████████████████████████████████| 61 kB 345 kB/s 
    [K     |████████████████████████████████| 856 kB 42.0 MB/s 
    [K     |████████████████████████████████| 3.6 MB 54.9 MB/s 
    [?25h  Building wheel for ffmpy (setup.py) ... [?25l[?25hdone
      Building wheel for flask-cachebuster (setup.py) ... [?25l[?25hdone



```python
import gradio as gr
inp = gr.inputs.Textbox(lines=2, default='what is coronavirus?', label="Question")
out = gr.outputs.Dataframe(label="Answers")#gr.outputs.Textbox(label="Answers")
gr.Interface(fn=get_answer_dpr, inputs=inp, outputs=out).launch()
```

    Colab notebook detected. To show errors in colab notebook, set `debug=True` in `launch()`
    Running on public URL: https://21615.gradio.app
    
    This share link expires in 72 hours. For free permanent hosting, check out Spaces (https://huggingface.co/spaces)




<iframe
    width="900"
    height="500"
    src="https://21615.gradio.app"
    frameborder="0"
    allowfullscreen
></iframe>






    (<Flask 'gradio.networking'>,
     'http://127.0.0.1:7860/',
     'https://21615.gradio.app')




```python

```
