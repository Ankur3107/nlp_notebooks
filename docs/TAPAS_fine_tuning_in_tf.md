<a href="https://colab.research.google.com/github/Ankur3107/nlp_notebooks/blob/master/table-qa/TAPAS_fine_tuning_in_tf.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
!pip install git+https://github.com/huggingface/transformers.git
```


```python
!pip install tensorflow_probability -qqq
```


```python
import requests, zipfile, io
import os

def download_files(dir_name):
  if not os.path.exists(dir_name): 
    # 28 training examples from the SQA training set + table csv data
    urls = ["https://www.dropbox.com/s/2p6ez9xro357i63/sqa_train_set_28_examples.zip?dl=1",
            "https://www.dropbox.com/s/abhum8ssuow87h6/table_csv.zip?dl=1"
    ]
    for url in urls:
      r = requests.get(url)
      z = zipfile.ZipFile(io.BytesIO(r.content))
      z.extractall()

dir_name = "sqa_data"
download_files(dir_name)
```


```python
import pandas as pd

data = pd.read_excel("sqa_train_set_28_examples.xlsx")
data.head()
```





  <div id="df-c6886234-94fb-4aba-aa0a-c3ce77f062d4">
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
      <th>id</th>
      <th>annotator</th>
      <th>position</th>
      <th>question</th>
      <th>table_file</th>
      <th>answer_coordinates</th>
      <th>answer_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nt-639</td>
      <td>0</td>
      <td>0</td>
      <td>where are the players from?</td>
      <td>table_csv/203_149.csv</td>
      <td>['(0, 4)', '(1, 4)', '(2, 4)', '(3, 4)', '(4, ...</td>
      <td>['Louisiana State University', 'Valley HS (Las...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nt-639</td>
      <td>0</td>
      <td>1</td>
      <td>which player went to louisiana state university?</td>
      <td>table_csv/203_149.csv</td>
      <td>['(0, 1)']</td>
      <td>['Ben McDonald']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nt-639</td>
      <td>1</td>
      <td>0</td>
      <td>who are the players?</td>
      <td>table_csv/203_149.csv</td>
      <td>['(0, 1)', '(1, 1)', '(2, 1)', '(3, 1)', '(4, ...</td>
      <td>['Ben McDonald', 'Tyler Houston', 'Roger Salke...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nt-639</td>
      <td>1</td>
      <td>1</td>
      <td>which ones are in the top 26 picks?</td>
      <td>table_csv/203_149.csv</td>
      <td>['(0, 1)', '(1, 1)', '(2, 1)', '(3, 1)', '(4, ...</td>
      <td>['Ben McDonald', 'Tyler Houston', 'Roger Salke...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nt-639</td>
      <td>1</td>
      <td>2</td>
      <td>and of those, who is from louisiana state univ...</td>
      <td>table_csv/203_149.csv</td>
      <td>['(0, 1)']</td>
      <td>['Ben McDonald']</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c6886234-94fb-4aba-aa0a-c3ce77f062d4')"
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
          document.querySelector('#df-c6886234-94fb-4aba-aa0a-c3ce77f062d4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c6886234-94fb-4aba-aa0a-c3ce77f062d4');
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
import ast

def _parse_answer_coordinates(answer_coordinate_str):
  """Parses the answer_coordinates of a question.
  Args:
    answer_coordinate_str: A string representation of a Python list of tuple
      strings.
      For example: "['(1, 4)','(1, 3)', ...]"
  """

  try:
    answer_coordinates = []
    # make a list of strings
    coords = ast.literal_eval(answer_coordinate_str)
    # parse each string as a tuple
    for row_index, column_index in sorted(
        ast.literal_eval(coord) for coord in coords):
      answer_coordinates.append((row_index, column_index))
  except SyntaxError:
    raise ValueError('Unable to evaluate %s' % answer_coordinate_str)
  
  return answer_coordinates


def _parse_answer_text(answer_text):
  """Populates the answer_texts field of `answer` by parsing `answer_text`.
  Args:
    answer_text: A string representation of a Python list of strings.
      For example: "[u'test', u'hello', ...]"
    answer: an Answer object.
  """
  try:
    answer = []
    for value in ast.literal_eval(answer_text):
      answer.append(value)
  except SyntaxError:
    raise ValueError('Unable to evaluate %s' % answer_text)

  return answer

data['answer_coordinates'] = data['answer_coordinates'].apply(lambda coords_str: _parse_answer_coordinates(coords_str))
data['answer_text'] = data['answer_text'].apply(lambda txt: _parse_answer_text(txt))

data.head(10)
```





  <div id="df-4f00d12e-e50f-4bb7-b3f6-d47c3db7f63d">
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
      <th>id</th>
      <th>annotator</th>
      <th>position</th>
      <th>question</th>
      <th>table_file</th>
      <th>answer_coordinates</th>
      <th>answer_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nt-639</td>
      <td>0</td>
      <td>0</td>
      <td>where are the players from?</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4...</td>
      <td>[Louisiana State University, Valley HS (Las Ve...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nt-639</td>
      <td>0</td>
      <td>1</td>
      <td>which player went to louisiana state university?</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 1)]</td>
      <td>[Ben McDonald]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nt-639</td>
      <td>1</td>
      <td>0</td>
      <td>who are the players?</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1...</td>
      <td>[Ben McDonald, Tyler Houston, Roger Salkeld, J...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nt-639</td>
      <td>1</td>
      <td>1</td>
      <td>which ones are in the top 26 picks?</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1...</td>
      <td>[Ben McDonald, Tyler Houston, Roger Salkeld, J...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nt-639</td>
      <td>1</td>
      <td>2</td>
      <td>and of those, who is from louisiana state univ...</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 1)]</td>
      <td>[Ben McDonald]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nt-639</td>
      <td>2</td>
      <td>0</td>
      <td>who are the players in the top 26?</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1...</td>
      <td>[Ben McDonald, Tyler Houston, Roger Salkeld, J...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>nt-639</td>
      <td>2</td>
      <td>1</td>
      <td>of those, which one was from louisiana state u...</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 1)]</td>
      <td>[Ben McDonald]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>nt-11649</td>
      <td>0</td>
      <td>0</td>
      <td>what are all the names of the teams?</td>
      <td>table_csv/204_135.csv</td>
      <td>[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1...</td>
      <td>[Cordoba CF, CD Malaga, Granada CF, UD Las Pal...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nt-11649</td>
      <td>0</td>
      <td>1</td>
      <td>of these, which teams had any losses?</td>
      <td>table_csv/204_135.csv</td>
      <td>[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1...</td>
      <td>[Cordoba CF, CD Malaga, Granada CF, UD Las Pal...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>nt-11649</td>
      <td>0</td>
      <td>2</td>
      <td>of these teams, which had more than 21 losses?</td>
      <td>table_csv/204_135.csv</td>
      <td>[(15, 1)]</td>
      <td>[CD Villarrobledo]</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4f00d12e-e50f-4bb7-b3f6-d47c3db7f63d')"
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
          document.querySelector('#df-4f00d12e-e50f-4bb7-b3f6-d47c3db7f63d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4f00d12e-e50f-4bb7-b3f6-d47c3db7f63d');
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
def get_sequence_id(example_id, annotator):
  if "-" in str(annotator):
    raise ValueError('"-" not allowed in annotator.')
  return f"{example_id}-{annotator}"

data['sequence_id'] = data.apply(lambda x: get_sequence_id(x.id, x.annotator), axis=1)
data.head()
```





  <div id="df-5a485d31-a71d-4e7f-986e-e000ed1f0c74">
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
      <th>id</th>
      <th>annotator</th>
      <th>position</th>
      <th>question</th>
      <th>table_file</th>
      <th>answer_coordinates</th>
      <th>answer_text</th>
      <th>sequence_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nt-639</td>
      <td>0</td>
      <td>0</td>
      <td>where are the players from?</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4...</td>
      <td>[Louisiana State University, Valley HS (Las Ve...</td>
      <td>nt-639-0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nt-639</td>
      <td>0</td>
      <td>1</td>
      <td>which player went to louisiana state university?</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 1)]</td>
      <td>[Ben McDonald]</td>
      <td>nt-639-0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nt-639</td>
      <td>1</td>
      <td>0</td>
      <td>who are the players?</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1...</td>
      <td>[Ben McDonald, Tyler Houston, Roger Salkeld, J...</td>
      <td>nt-639-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nt-639</td>
      <td>1</td>
      <td>1</td>
      <td>which ones are in the top 26 picks?</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1...</td>
      <td>[Ben McDonald, Tyler Houston, Roger Salkeld, J...</td>
      <td>nt-639-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nt-639</td>
      <td>1</td>
      <td>2</td>
      <td>and of those, who is from louisiana state univ...</td>
      <td>table_csv/203_149.csv</td>
      <td>[(0, 1)]</td>
      <td>[Ben McDonald]</td>
      <td>nt-639-1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5a485d31-a71d-4e7f-986e-e000ed1f0c74')"
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
          document.querySelector('#df-5a485d31-a71d-4e7f-986e-e000ed1f0c74 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5a485d31-a71d-4e7f-986e-e000ed1f0c74');
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
# let's group table-question pairs by sequence id, and remove some columns we don't need 
grouped = data.groupby(by='sequence_id').agg(lambda x: x.tolist())
grouped = grouped.drop(columns=['id', 'annotator', 'position'])
grouped['table_file'] = grouped['table_file'].apply(lambda x: x[0])
grouped.head(10)
```





  <div id="df-3ad59c92-9893-4667-b59a-2f3e3d48d94c">
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
      <th>question</th>
      <th>table_file</th>
      <th>answer_coordinates</th>
      <th>answer_text</th>
    </tr>
    <tr>
      <th>sequence_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ns-1292-0</th>
      <td>[who are all the athletes?, where are they fro...</td>
      <td>table_csv/204_521.csv</td>
      <td>[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, ...</td>
      <td>[[Tommy Green, Janis Dalins, Ugo Frigerio, Kar...</td>
    </tr>
    <tr>
      <th>nt-10730-0</th>
      <td>[what was the production numbers of each revol...</td>
      <td>table_csv/203_253.csv</td>
      <td>[[(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, ...</td>
      <td>[[1,900 (estimated), 14,500 (estimated), 6,000...</td>
    </tr>
    <tr>
      <th>nt-10730-1</th>
      <td>[what three revolver models had the least amou...</td>
      <td>table_csv/203_253.csv</td>
      <td>[[(0, 0), (6, 0), (7, 0)], [(0, 0)]]</td>
      <td>[[Remington-Beals Army Model Revolver, New Mod...</td>
    </tr>
    <tr>
      <th>nt-10730-2</th>
      <td>[what are all of the remington models?, how ma...</td>
      <td>table_csv/203_253.csv</td>
      <td>[[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, ...</td>
      <td>[[Remington-Beals Army Model Revolver, Remingt...</td>
    </tr>
    <tr>
      <th>nt-11649-0</th>
      <td>[what are all the names of the teams?, of thes...</td>
      <td>table_csv/204_135.csv</td>
      <td>[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, ...</td>
      <td>[[Cordoba CF, CD Malaga, Granada CF, UD Las Pa...</td>
    </tr>
    <tr>
      <th>nt-11649-1</th>
      <td>[what are the losses?, what team had more than...</td>
      <td>table_csv/204_135.csv</td>
      <td>[[(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, ...</td>
      <td>[[6, 6, 9, 10, 10, 12, 12, 11, 13, 14, 15, 14,...</td>
    </tr>
    <tr>
      <th>nt-11649-2</th>
      <td>[what were all the teams?, what were the loss ...</td>
      <td>table_csv/204_135.csv</td>
      <td>[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, ...</td>
      <td>[[Cordoba CF, CD Malaga, Granada CF, UD Las Pa...</td>
    </tr>
    <tr>
      <th>nt-639-0</th>
      <td>[where are the players from?, which player wen...</td>
      <td>table_csv/203_149.csv</td>
      <td>[[(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, ...</td>
      <td>[[Louisiana State University, Valley HS (Las V...</td>
    </tr>
    <tr>
      <th>nt-639-1</th>
      <td>[who are the players?, which ones are in the t...</td>
      <td>table_csv/203_149.csv</td>
      <td>[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, ...</td>
      <td>[[Ben McDonald, Tyler Houston, Roger Salkeld, ...</td>
    </tr>
    <tr>
      <th>nt-639-2</th>
      <td>[who are the players in the top 26?, of those,...</td>
      <td>table_csv/203_149.csv</td>
      <td>[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, ...</td>
      <td>[[Ben McDonald, Tyler Houston, Roger Salkeld, ...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3ad59c92-9893-4667-b59a-2f3e3d48d94c')"
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
          document.querySelector('#df-3ad59c92-9893-4667-b59a-2f3e3d48d94c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3ad59c92-9893-4667-b59a-2f3e3d48d94c');
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
# path to the directory containing all csv files
table_csv_path = "table_csv"

item = grouped.iloc[0]
table = pd.read_csv(table_csv_path + item.table_file[9:]).astype(str) 

display(table)
print("")
print(item.question)
```



  <div id="df-a87bdcd9-f979-453f-998d-d8130202968b">
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
      <th>Rank</th>
      <th>Name</th>
      <th>Nationality</th>
      <th>Time (hand)</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nan</td>
      <td>Tommy Green</td>
      <td>Great Britain</td>
      <td>4:50:10</td>
      <td>OR</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nan</td>
      <td>Janis Dalins</td>
      <td>Latvia</td>
      <td>4:57:20</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nan</td>
      <td>Ugo Frigerio</td>
      <td>Italy</td>
      <td>4:59:06</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>Karl Hahnel</td>
      <td>Germany</td>
      <td>5:06:06</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>Ettore Rivolta</td>
      <td>Italy</td>
      <td>5:07:39</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>Paul Sievert</td>
      <td>Germany</td>
      <td>5:16:41</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>Henri Quintric</td>
      <td>France</td>
      <td>5:27:25</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>Ernie Crosbie</td>
      <td>United States</td>
      <td>5:28:02</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>Bill Chisholm</td>
      <td>United States</td>
      <td>5:51:00</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>Alfred Maasik</td>
      <td>Estonia</td>
      <td>6:19:00</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>10</th>
      <td>nan</td>
      <td>Henry Cieman</td>
      <td>Canada</td>
      <td>nan</td>
      <td>DNF</td>
    </tr>
    <tr>
      <th>11</th>
      <td>nan</td>
      <td>John Moralis</td>
      <td>Greece</td>
      <td>nan</td>
      <td>DNF</td>
    </tr>
    <tr>
      <th>12</th>
      <td>nan</td>
      <td>Francesco Pretti</td>
      <td>Italy</td>
      <td>nan</td>
      <td>DNF</td>
    </tr>
    <tr>
      <th>13</th>
      <td>nan</td>
      <td>Arthur Tell Schwab</td>
      <td>Switzerland</td>
      <td>nan</td>
      <td>DNF</td>
    </tr>
    <tr>
      <th>14</th>
      <td>nan</td>
      <td>Harry Hinkel</td>
      <td>United States</td>
      <td>nan</td>
      <td>DNF</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a87bdcd9-f979-453f-998d-d8130202968b')"
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
          document.querySelector('#df-a87bdcd9-f979-453f-998d-d8130202968b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a87bdcd9-f979-453f-998d-d8130202968b');
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



    
    ['who are all the athletes?', 'where are they from?', 'along with paul sievert, which athlete is from germany?']



```python
import tensorflow as tf
from transformers import TapasTokenizer

# initialize the tokenizer
tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")
```


    Downloading:   0%|          | 0.00/256k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/154 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/490 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/1.49k [00:00<?, ?B/s]



```python
encoding = tokenizer(table=table, queries=item.question, answer_coordinates=item.answer_coordinates, answer_text=item.answer_text,
                     truncation=True, padding="max_length", return_tensors="tf")
encoding.keys()
```




    dict_keys(['input_ids', 'labels', 'numeric_values', 'numeric_values_scale', 'token_type_ids', 'attention_mask'])




```python
tokenizer.decode(encoding["input_ids"][0])
```




    '[CLS] who are all the athletes? [SEP] rank name nationality time ( hand ) notes [EMPTY] tommy green great britain 4 : 50 : 10 or [EMPTY] janis dalins latvia 4 : 57 : 20 [EMPTY] [EMPTY] ugo frigerio italy 4 : 59 : 06 [EMPTY] 4. 0 karl hahnel germany 5 : 06 : 06 [EMPTY] 5. 0 ettore rivolta italy 5 : 07 : 39 [EMPTY] 6. 0 paul sievert germany 5 : 16 : 41 [EMPTY] 7. 0 henri quintric france 5 : 27 : 25 [EMPTY] 8. 0 ernie crosbie united states 5 : 28 : 02 [EMPTY] 9. 0 bill chisholm united states 5 : 51 : 00 [EMPTY] 10. 0 alfred maasik estonia 6 : 19 : 00 [EMPTY] [EMPTY] henry cieman canada [EMPTY] dnf [EMPTY] john moralis greece [EMPTY] dnf [EMPTY] francesco pretti italy [EMPTY] dnf [EMPTY] arthur tell schwab switzerland [EMPTY] dnf [EMPTY] harry hinkel united states [EMPTY] dnf [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'




```python
encoding["labels"][0]
```




    <tf.Tensor: shape=(512,), dtype=int32, numpy=
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
           1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0], dtype=int32)>




```python
class TableDataset:

    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
    
    def __iter__(self):
        for idx in range(self.__len__()):
          item = self.df.iloc[idx]
          table = pd.read_csv(table_csv_path + item.table_file[9:]).astype(str) # TapasTokenizer expects the table data to be text only
          if item.position != 0:
            # use the previous table-question pair to correctly set the prev_labels token type ids
            previous_item = self.df.iloc[idx-1]
            encoding = self.tokenizer(table=table, 
                                      queries=[previous_item.question, item.question], 
                                      answer_coordinates=[previous_item.answer_coordinates, item.answer_coordinates], 
                                      answer_text=[previous_item.answer_text, item.answer_text],
                                      padding="max_length",
                                      truncation=True,
                                      return_tensors="tf"
            )
            # use encodings of second table-question pair in the batch
            encoding = {key: val[-1] for key, val in encoding.items()}
          else:
            # this means it's the first table-question pair in a sequence
            encoding = self.tokenizer(table=table, 
                                      queries=item.question, 
                                      answer_coordinates=item.answer_coordinates, 
                                      answer_text=item.answer_text,
                                      padding="max_length",
                                      truncation=True,
                                      return_tensors="tf"
            )
            # remove the batch dimension which the tokenizer adds 
            encoding = {key: tf.squeeze(val,0) for key, val in encoding.items()}
          yield encoding['input_ids'],encoding['attention_mask'],encoding['numeric_values'], \
                encoding['numeric_values_scale'], encoding['token_type_ids'], encoding['labels']

    def __len__(self):
        return len(self.df)
    
    __call__ = __iter__

train_dataset = TableDataset(df=data, tokenizer=tokenizer)
output_signature = (
         tf.TensorSpec(shape=(512,), dtype=tf.int32),
         tf.TensorSpec(shape=(512,), dtype=tf.int32),
         tf.TensorSpec(shape=(512,), dtype=tf.float32),
         tf.TensorSpec(shape=(512,), dtype=tf.float32),
         tf.TensorSpec(shape=(512,7), dtype=tf.int32),
         tf.TensorSpec(shape=(512,), dtype=tf.int32) )
train_dataloader = tf.data.Dataset.from_generator(train_dataset,output_signature=output_signature).batch(2)
```


```python
batch = next(iter(train_dataloader))
```


```python
batch[0].shape
```




    TensorShape([2, 512])




```python
batch[4].shape
```




    TensorShape([2, 512, 7])




```python
batch[2].shape
```




    TensorShape([2, 512])




```python
tokenizer.decode(batch[0][1])
```




    '[CLS] which player went to louisiana state university? [SEP] pick player team position school 1 ben mcdonald baltimore orioles rhp louisiana state university 2 tyler houston atlanta braves c valley hs ( las vegas, nv ) 3 roger salkeld seattle mariners rhp saugus ( ca ) hs 4 jeff jackson philadelphia phillies of simeon hs ( chicago, il ) 5 donald harris texas rangers of texas tech university 6 paul coleman saint louis cardinals of frankston ( tx ) hs 7 frank thomas chicago white sox 1b auburn university 8 earl cunningham chicago cubs of lancaster ( sc ) hs 9 kyle abbott california angels lhp long beach state university 10 charles johnson montreal expos c westwood hs ( fort pierce, fl ) 11 calvin murray cleveland indians 3b w. t. white high school ( dallas, tx ) 12 jeff juden houston astros rhp salem ( ma ) hs 13 brent mayne kansas city royals c cal state fullerton 14 steve hosey san francisco giants of fresno state university 15 kiki jones los angeles dodgers rhp hillsborough hs ( tampa, fl ) 16 greg blosser boston red sox of sarasota ( fl ) hs 17 cal eldred milwaukee brewers rhp university of iowa 18 willie greene pittsburgh pirates ss jones county hs ( gray, ga ) 19 eddie zosky toronto blue jays ss fresno state university 20 scott bryant cincinnati reds of university of texas 21 greg gohr detroit tigers rhp santa clara university 22 tom goodwin los angeles dodgers of fresno state university 23 mo vaughn boston red sox 1b seton hall university 24 alan zinter new york mets c university of arizona 25 chuck knoblauch minnesota twins 2b texas a & m university 26 scott burrell seattle mariners rhp hamden ( ct ) hs [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'




```python
assert sum(batch[4][0][:,3]) == 0
```


```python
print(sum(batch[4][1][:,3]))
```

    tf.Tensor(132, shape=(), dtype=int32)



```python
for id, prev_label in zip(batch[0][1], batch[4][1][:,3]):
  if id != 0:
    print(tokenizer.decode([id]), prev_label.numpy().item())
```

    [CLS] 0
    which 0
    player 0
    went 0
    to 0
    louisiana 0
    state 0
    university 0
    ? 0
    [SEP] 0
    pick 0
    player 0
    team 0
    position 0
    school 0
    1 0
    ben 0
    mcdonald 0
    baltimore 0
    orioles 0
    r 0
    ##hp 0
    louisiana 1
    state 1
    university 1
    2 0
    tyler 0
    houston 0
    atlanta 0
    braves 0
    c 0
    valley 1
    hs 1
    ( 1
    las 1
    vegas 1
    , 1
    n 1
    ##v 1
    ) 1
    3 0
    roger 0
    sal 0
    ##kel 0
    ##d 0
    seattle 0
    mariners 0
    r 0
    ##hp 0
    sa 1
    ##ug 1
    ##us 1
    ( 1
    ca 1
    ) 1
    hs 1
    4 0
    jeff 0
    jackson 0
    philadelphia 0
    phillies 0
    of 0
    simeon 1
    hs 1
    ( 1
    chicago 1
    , 1
    il 1
    ) 1
    5 0
    donald 0
    harris 0
    texas 0
    rangers 0
    of 0
    texas 1
    tech 1
    university 1
    6 0
    paul 0
    coleman 0
    saint 0
    louis 0
    cardinals 0
    of 0
    franks 1
    ##ton 1
    ( 1
    tx 1
    ) 1
    hs 1
    7 0
    frank 0
    thomas 0
    chicago 0
    white 0
    sox 0
    1b 0
    auburn 1
    university 1
    8 0
    earl 0
    cunningham 0
    chicago 0
    cubs 0
    of 0
    lancaster 1
    ( 1
    sc 1
    ) 1
    hs 1
    9 0
    kyle 0
    abbott 0
    california 0
    angels 0
    l 0
    ##hp 0
    long 1
    beach 1
    state 1
    university 1
    10 0
    charles 0
    johnson 0
    montreal 0
    expo 0
    ##s 0
    c 0
    westwood 1
    hs 1
    ( 1
    fort 1
    pierce 1
    , 1
    fl 1
    ) 1
    11 0
    calvin 0
    murray 0
    cleveland 0
    indians 0
    3 0
    ##b 0
    w 1
    . 1
    t 1
    . 1
    white 1
    high 1
    school 1
    ( 1
    dallas 1
    , 1
    tx 1
    ) 1
    12 0
    jeff 0
    jude 0
    ##n 0
    houston 0
    astros 0
    r 0
    ##hp 0
    salem 1
    ( 1
    ma 1
    ) 1
    hs 1
    13 0
    brent 0
    may 0
    ##ne 0
    kansas 0
    city 0
    royals 0
    c 0
    cal 1
    state 1
    fuller 1
    ##ton 1
    14 0
    steve 0
    hose 0
    ##y 0
    san 0
    francisco 0
    giants 0
    of 0
    fresno 1
    state 1
    university 1
    15 0
    ki 0
    ##ki 0
    jones 0
    los 0
    angeles 0
    dodgers 0
    r 0
    ##hp 0
    hillsborough 1
    hs 1
    ( 1
    tampa 1
    , 1
    fl 1
    ) 1
    16 0
    greg 0
    b 0
    ##los 0
    ##ser 0
    boston 0
    red 0
    sox 0
    of 0
    sara 1
    ##so 1
    ##ta 1
    ( 1
    fl 1
    ) 1
    hs 1
    17 0
    cal 0
    el 0
    ##dre 0
    ##d 0
    milwaukee 0
    brewers 0
    r 0
    ##hp 0
    university 1
    of 1
    iowa 1
    18 0
    willie 0
    greene 0
    pittsburgh 0
    pirates 0
    ss 0
    jones 1
    county 1
    hs 1
    ( 1
    gray 1
    , 1
    ga 1
    ) 1
    19 0
    eddie 0
    z 0
    ##os 0
    ##ky 0
    toronto 0
    blue 0
    jays 0
    ss 0
    fresno 1
    state 1
    university 1
    20 0
    scott 0
    bryant 0
    cincinnati 0
    reds 0
    of 0
    university 1
    of 1
    texas 1
    21 0
    greg 0
    go 0
    ##hr 0
    detroit 0
    tigers 0
    r 0
    ##hp 0
    santa 1
    clara 1
    university 1
    22 0
    tom 0
    goodwin 0
    los 0
    angeles 0
    dodgers 0
    of 0
    fresno 1
    state 1
    university 1
    23 0
    mo 0
    vaughn 0
    boston 0
    red 0
    sox 0
    1b 0
    seton 1
    hall 1
    university 1
    24 0
    alan 0
    z 0
    ##int 0
    ##er 0
    new 0
    york 0
    mets 0
    c 0
    university 1
    of 1
    arizona 1
    25 0
    chuck 0
    knob 0
    ##lau 0
    ##ch 0
    minnesota 0
    twins 0
    2 0
    ##b 0
    texas 1
    a 1
    & 1
    m 1
    university 1
    26 0
    scott 0
    burr 0
    ##ell 0
    seattle 0
    mariners 0
    r 0
    ##hp 0
    ham 1
    ##den 1
    ( 1
    ct 1
    ) 1
    hs 1



```python
from transformers import TFTapasForQuestionAnswering

model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base")
```


    Downloading:   0%|          | 0.00/422M [00:00<?, ?B/s]


    All model checkpoint layers were used when initializing TFTapasForQuestionAnswering.
    
    Some layers of TFTapasForQuestionAnswering were not initialized from the model checkpoint at google/tapas-base and are newly initialized: ['compute_column_logits', 'dropout_37', 'compute_token_logits']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
model.config.num_aggregation_labels
```




    0




```python
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)


for epoch in range(10):  # loop over the dataset multiple times
   print("Epoch:", epoch)
   for idx, batch in enumerate(train_dataloader):
        # get the inputs;
        input_ids = batch[0]
        attention_mask = batch[1]
        token_type_ids = batch[4]
        labels = batch[-1]
        
        with tf.GradientTape() as tape:
          outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        labels=labels, training=True)
        
        print("loss: ",outputs.loss.numpy().item())
        grads = tape.gradient(outputs.loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
```

    Epoch: 0
    loss:  2.2713990211486816
    WARNING:tensorflow:Gradients do not exist for variables ['tf_tapas_for_question_answering/tapas/pooler/dense/kernel:0', 'tf_tapas_for_question_answering/tapas/pooler/dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    loss:  2.0187385082244873
    WARNING:tensorflow:Gradients do not exist for variables ['tf_tapas_for_question_answering/tapas/pooler/dense/kernel:0', 'tf_tapas_for_question_answering/tapas/pooler/dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    loss:  1.3549939393997192



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-33-dbbf9094cd94> in <module>()
         16 
         17         print("loss: ",outputs.loss.numpy().item())
    ---> 18         grads = tape.gradient(outputs.loss, model.trainable_weights)
         19         optimizer.apply_gradients(zip(grads, model.trainable_weights))


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/backprop.py in gradient(self, target, sources, output_gradients, unconnected_gradients)
       1088         output_gradients=output_gradients,
       1089         sources_raw=flat_sources_raw,
    -> 1090         unconnected_gradients=unconnected_gradients)
       1091 
       1092     if not self._persistent:


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/imperative_grad.py in imperative_grad(tape, target, sources, output_gradients, sources_raw, unconnected_gradients)
         75       output_gradients,
         76       sources_raw,
    ---> 77       compat.as_str(unconnected_gradients.value))
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/backprop.py in _gradient_function(op_name, attr_tuple, num_inputs, inputs, outputs, out_grads, skip_input_indices, forward_pass_name_scope)
        157       gradient_name_scope += forward_pass_name_scope + "/"
        158     with ops.name_scope(gradient_name_scope):
    --> 159       return grad_fn(mock_op, *out_grads)
        160   else:
        161     return grad_fn(mock_op, *out_grads)


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/math_grad.py in _BatchMatMulV2(op, grad)
       1866   if not adj_x:
       1867     if not adj_y:
    -> 1868       grad_x = math_ops.matmul(grad, y, adjoint_a=False, adjoint_b=True)
       1869       grad_y = math_ops.matmul(x, grad, adjoint_a=True, adjoint_b=False)
       1870     else:


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
        148     filtered_tb = None
        149     try:
    --> 150       return fn(*args, **kwargs)
        151     except Exception as e:
        152       filtered_tb = _process_traceback_frames(e.__traceback__)


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/dispatch.py in op_dispatch_handler(*args, **kwargs)
       1094       # Fallback dispatch system (dispatch v1):
       1095       try:
    -> 1096         return dispatch_target(*args, **kwargs)
       1097       except (TypeError, ValueError):
       1098         # Note: convert_to_eager_tensor currently raises a ValueError, not a


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/math_ops.py in matmul(a, b, transpose_a, transpose_b, adjoint_a, adjoint_b, a_is_sparse, b_is_sparse, output_type, name)
       3652       else:
       3653         return gen_math_ops.batch_mat_mul_v2(
    -> 3654             a, b, adj_x=adjoint_a, adj_y=adjoint_b, name=name)
       3655 
       3656     # Neither matmul nor sparse_matmul support adjoint, so we conjugate


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/gen_math_ops.py in batch_mat_mul_v2(x, y, adj_x, adj_y, name)
       1570     try:
       1571       _result = pywrap_tfe.TFE_Py_FastPathExecute(
    -> 1572         _ctx, "BatchMatMulV2", name, x, y, "adj_x", adj_x, "adj_y", adj_y)
       1573       return _result
       1574     except _core._NotOkStatusException as e:


    KeyboardInterrupt: 



```python
import tensorflow_probability as tfp
```


```python
import collections
import numpy as np

def compute_prediction_sequence(model, data):
  """Computes predictions using model's answers to the previous questions."""
  
  # prepare data
  all_logits = []
  prev_answers = None
  batch_size = inputs["input_ids"].shape[0]

  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]
  token_type_ids = inputs["token_type_ids"]
  token_type_ids_example = None

  for index in range(batch_size):
      # If sequences have already been processed, the token type IDs will be created according to the previous
      # answer.
      if prev_answers is not None:
          prev_labels_example = token_type_ids_example[:, 3]  # shape (seq_len,)
          model_labels = np.zeros_like(prev_labels_example, dtype=np.int32)  # shape (seq_len,)

          token_type_ids_example = token_type_ids[index].numpy()  # shape (seq_len, 7)
          for i in range(model_labels.shape[0]):
              segment_id = token_type_ids_example[:, 0].tolist()[i]
              col_id = token_type_ids_example[:, 1].tolist()[i] - 1
              row_id = token_type_ids_example[:, 2].tolist()[i] - 1

              if row_id >= 0 and col_id >= 0 and segment_id == 1:
                  model_labels[i] = int(prev_answers[(col_id, row_id)])

          token_type_ids_example[:, 3] = model_labels

      input_ids_example = input_ids[index]
      attention_mask_example = attention_mask[index]  # shape (seq_len,)
      token_type_ids_example = token_type_ids[index]  # shape (seq_len, 7)
      outputs = model(
          input_ids=np.expand_dims(input_ids_example, axis=0),
          attention_mask=np.expand_dims(attention_mask_example, axis=0),
          token_type_ids=np.expand_dims(token_type_ids_example, axis=0),
      )
      logits = outputs.logits


      all_logits.append(logits)

      dist_per_token = tfp.distributions.Bernoulli(logits=logits)
      probabilities = dist_per_token.probs_parameter() * tf.cast(attention_mask_example, tf.float32)

      coords_to_probs = collections.defaultdict(list)
      token_type_ids_example = token_type_ids_example.numpy()
      for i, p in enumerate(tf.squeeze(probabilities).numpy().tolist()):
          segment_id = token_type_ids_example[:, 0].tolist()[i]
          col = token_type_ids_example[:, 1].tolist()[i] - 1
          row = token_type_ids_example[:, 2].tolist()[i] - 1
          if col >= 0 and row >= 0 and segment_id == 1:
              coords_to_probs[(col, row)].append(p)

      prev_answers = {key: np.array(coords_to_probs[key]).mean() > 0.5 for key in coords_to_probs}

  logits_batch = tf.concat(tuple(all_logits), 0)

  return logits_batch
```


```python
data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], 
        'Age': ["56", "45", "59"],
        'Number of movies': ["87", "53", "69"],
        'Date of birth': ["7 february 1967", "10 june 1996", "28 november 1967"]}
queries = ["How many movies has George Clooney played in?", "How old is he?", "What's his date of birth?"]

table = pd.DataFrame.from_dict(data)

inputs = tokenizer(table=table, queries=queries, padding='max_length', return_tensors="tf")
logits = compute_prediction_sequence(model, inputs)
```


```python
predicted_answer_coordinates, = tokenizer.convert_logits_to_predictions(inputs, logits)
```


```python
predicted_answer_coordinates
```




    [[(0, 3), (1, 3), (2, 3)], [(0, 3), (1, 3), (2, 3)], [(0, 3), (1, 3), (2, 3)]]




```python
# handy helper function in case inference on Pandas dataframe
answers = []
for coordinates in predicted_answer_coordinates:
  if len(coordinates) == 1:
    # only a single cell:
    answers.append(table.iat[coordinates[0]])
  else:
    # multiple cells
    cell_values = []
    for coordinate in coordinates:
      cell_values.append(table.iat[coordinate])
    answers.append(", ".join(cell_values))

display(table)
print("")
for query, answer in zip(queries, answers):
  print(query)
  print("Predicted answer: " + answer)
```



  <div id="df-eb723aa7-316e-4c7f-99bb-7e2b0be5ba2a">
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
      <th>Actors</th>
      <th>Age</th>
      <th>Number of movies</th>
      <th>Date of birth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brad Pitt</td>
      <td>56</td>
      <td>87</td>
      <td>7 february 1967</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Leonardo Di Caprio</td>
      <td>45</td>
      <td>53</td>
      <td>10 june 1996</td>
    </tr>
    <tr>
      <th>2</th>
      <td>George Clooney</td>
      <td>59</td>
      <td>69</td>
      <td>28 november 1967</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-eb723aa7-316e-4c7f-99bb-7e2b0be5ba2a')"
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
          document.querySelector('#df-eb723aa7-316e-4c7f-99bb-7e2b0be5ba2a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-eb723aa7-316e-4c7f-99bb-7e2b0be5ba2a');
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



    
    How many movies has George Clooney played in?
    Predicted answer: 7 february 1967, 10 june 1996, 28 november 1967
    How old is he?
    Predicted answer: 7 february 1967, 10 june 1996, 28 november 1967
    What's his date of birth?
    Predicted answer: 7 february 1967, 10 june 1996, 28 november 1967


# Reference 
https://github.com/kamalkraj/Tapas-Tutorial

