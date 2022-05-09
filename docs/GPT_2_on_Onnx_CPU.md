<a href="https://colab.research.google.com/github/Ankur3107/nlp_notebooks/blob/master/nlp-onnx/GPT_2_on_Onnx_CPU.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
!pip install onnxruntime==1.8.1 onnx==1.9.0 onnxconverter_common==1.8.1 transformers==4.8.2 psutil pytz pandas py-cpuinfo py3nvml
```

        Uninstalling transformers-4.14.1:
          Successfully uninstalled transformers-4.14.1
      Attempting uninstall: onnxruntime
        Found existing installation: onnxruntime 1.10.0
        Uninstalling onnxruntime-1.10.0:
          Successfully uninstalled onnxruntime-1.10.0
      Attempting uninstall: onnxconverter-common
        Found existing installation: onnxconverter-common 1.9.0
        Uninstalling onnxconverter-common-1.9.0:
          Successfully uninstalled onnxconverter-common-1.9.0
    Successfully installed huggingface-hub-0.0.12 onnx-1.9.0 onnxconverter-common-1.8.1 onnxruntime-1.8.1 transformers-4.8.2





```python
import os

# Create a cache directory to store pretrained model.
cache_dir = os.path.join(".", "cache_models")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
```


```python
!lscpu
```

    Architecture:        x86_64
    CPU op-mode(s):      32-bit, 64-bit
    Byte Order:          Little Endian
    CPU(s):              2
    On-line CPU(s) list: 0,1
    Thread(s) per core:  2
    Core(s) per socket:  1
    Socket(s):           1
    NUMA node(s):        1
    Vendor ID:           GenuineIntel
    CPU family:          6
    Model:               79
    Model name:          Intel(R) Xeon(R) CPU @ 2.20GHz
    Stepping:            0
    CPU MHz:             2199.998
    BogoMIPS:            4399.99
    Hypervisor vendor:   KVM
    Virtualization type: full
    L1d cache:           32K
    L1i cache:           32K
    L2 cache:            256K
    L3 cache:            56320K
    NUMA node0 CPU(s):   0,1
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm rdseed adx smap xsaveopt arat md_clear arch_capabilities



```python
!pip install coloredlogs
```

    Requirement already satisfied: coloredlogs in /usr/local/lib/python3.7/dist-packages (15.0.1)
    Requirement already satisfied: humanfriendly>=9.1 in /usr/local/lib/python3.7/dist-packages (from coloredlogs) (10.0)



```python
from onnxruntime.transformers.gpt2_beamsearch_helper import Gpt2BeamSearchHelper, GPT2LMHeadModel_BeamSearchStep
from transformers import AutoConfig
import torch
```


```python
model_name_or_path = "gpt2"
config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = GPT2LMHeadModel_BeamSearchStep.from_pretrained(model_name_or_path, config=config, batch_size=1, beam_size=4, cache_dir=cache_dir)
device = torch.device("cpu")
model.eval().to(device)

print(model.config)

num_attention_heads = model.config.n_head
hidden_size = model.config.n_embd
num_layer = model.config.n_layer
```

    GPT2Config {
      "_name_or_path": "gpt2",
      "activation_function": "gelu_new",
      "architectures": [
        "GPT2LMHeadModel"
      ],
      "attn_pdrop": 0.1,
      "batch_size": 1,
      "beam_size": 4,
      "bos_token_id": 50256,
      "embd_pdrop": 0.1,
      "eos_token_id": 50256,
      "gradient_checkpointing": false,
      "initializer_range": 0.02,
      "layer_norm_epsilon": 1e-05,
      "model_type": "gpt2",
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_inner": null,
      "n_layer": 12,
      "n_positions": 1024,
      "resid_pdrop": 0.1,
      "scale_attn_weights": true,
      "summary_activation": null,
      "summary_first_dropout": 0.1,
      "summary_proj_to_labels": true,
      "summary_type": "cls_index",
      "summary_use_proj": true,
      "task_specific_params": {
        "text-generation": {
          "do_sample": true,
          "max_length": 50
        }
      },
      "transformers_version": "4.8.2",
      "use_cache": true,
      "vocab_size": 50257
    }
    



```python
onnx_model_path = "gpt2_one_step_search.onnx"
Gpt2BeamSearchHelper.export_onnx(model, device, onnx_model_path) # add parameter use_external_data_format=True when model size > 2 GB
```

    /usr/local/lib/python3.7/dist-packages/onnxruntime/transformers/gpt2_beamsearch_helper.py:91: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
      selected_input_seq = selected_index_flat // self.config.beam_size
    /usr/local/lib/python3.7/dist-packages/torch/onnx/utils.py:100: UserWarning: `example_outputs' is deprecated and ignored. Will be removed in next PyTorch release.
      warnings.warn("`example_outputs' is deprecated and ignored. Will be removed in "
    /usr/local/lib/python3.7/dist-packages/torch/onnx/utils.py:103: UserWarning: `use_external_data_format' is deprecated and ignored. Will be removed in next PyTorch release. The code will work as it is False if models are not larger than 2GB, Otherwise set to False because of size limits imposed by Protocol Buffers.
      warnings.warn("`use_external_data_format' is deprecated and ignored. Will be removed in next "
    /usr/local/lib/python3.7/dist-packages/transformers/models/gpt2/modeling_gpt2.py:698: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert batch_size > 0, "batch_size has to be defined and > 0"
    /usr/local/lib/python3.7/dist-packages/transformers/models/gpt2/modeling_gpt2.py:249: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
      past_key, past_value = layer_past
    /usr/local/lib/python3.7/dist-packages/transformers/models/gpt2/modeling_gpt2.py:181: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)



```python
import onnxruntime
import numpy
from transformers import AutoTokenizer

EXAMPLE_Text = ['best hotel in bay area.']

def get_tokenizer(model_name_or_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    #okenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def get_example_inputs(prompt_text=EXAMPLE_Text):    
    tokenizer = get_tokenizer(model_name_or_path, cache_dir)
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)
    position_ids = (attention_mask.long().cumsum(-1) - 1)
    position_ids.masked_fill_(position_ids < 0, 0)

    #Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))
       
    return input_ids, attention_mask, position_ids, empty_past

input_ids, attention_mask, position_ids, empty_past = get_example_inputs()
beam_select_idx = torch.zeros([1, input_ids.shape[0]]).long()
input_log_probs = torch.zeros([input_ids.shape[0], 1])
input_unfinished_sents = torch.ones([input_ids.shape[0], 1], dtype=torch.bool)
prev_step_scores = torch.zeros([input_ids.shape[0], 1])

onnx_model_path = "gpt2_one_step_search.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)
ort_inputs = {
              'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy()),
              'attention_mask' : numpy.ascontiguousarray(attention_mask.cpu().numpy()),
              'position_ids': numpy.ascontiguousarray(position_ids.cpu().numpy()),
              'beam_select_idx': numpy.ascontiguousarray(beam_select_idx.cpu().numpy()),
              'input_log_probs': numpy.ascontiguousarray(input_log_probs.cpu().numpy()),
              'input_unfinished_sents': numpy.ascontiguousarray(input_unfinished_sents.cpu().numpy()),
              'prev_step_results': numpy.ascontiguousarray(input_ids.cpu().numpy()),
              'prev_step_scores': numpy.ascontiguousarray(prev_step_scores.cpu().numpy()),
             }
for i, past_i in enumerate(empty_past):
    ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past_i.cpu().numpy())
ort_outputs = session.run(None, ort_inputs)
```


```python
def inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, past, beam_select_idx, input_log_probs, input_unfinished_sents, prev_step_results, prev_step_scores, step, context_len):
    output_shapes = Gpt2BeamSearchHelper.get_output_shapes(batch_size=1,
                                                           context_len=context_len,
                                                           past_sequence_length=past[0].size(3),
                                                           sequence_length=input_ids.size(1),
                                                           beam_size=4,
                                                           step=step,
                                                           config=config,
                                                           model_class="GPT2LMHeadModel_BeamSearchStep")
    output_buffers = Gpt2BeamSearchHelper.get_output_buffers(output_shapes, device)

    io_binding = Gpt2BeamSearchHelper.prepare_io_binding(session, input_ids, position_ids, attention_mask, past, output_buffers, output_shapes, beam_select_idx, input_log_probs, input_unfinished_sents, prev_step_results, prev_step_scores)
    session.run_with_iobinding(io_binding)

    outputs = Gpt2BeamSearchHelper.get_outputs_from_io_binding_buffer(session, output_buffers, output_shapes, return_numpy=False)
    return outputs
```


```python
input_ids, attention_mask, position_ids, empty_past = get_example_inputs()
beam_select_idx = torch.zeros([1, input_ids.shape[0]]).long()
input_log_probs = torch.zeros([input_ids.shape[0], 1])
input_unfinished_sents = torch.ones([input_ids.shape[0], 1], dtype=torch.bool)
prev_step_scores = torch.zeros([input_ids.shape[0], 1])
outputs = inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, empty_past, beam_select_idx, input_log_probs, input_unfinished_sents, input_ids, prev_step_scores, 0, input_ids.shape[-1])
assert torch.eq(outputs[-2], torch.from_numpy(ort_outputs[-2])).all()
print("IO Binding result is good")
```

    IO Binding result is good



```python
def update(output, step, batch_size, beam_size, context_length, prev_attention_mask, device):
    """
    Update the inputs for next inference.
    """
    last_state = (torch.from_numpy(output[0]).to(device)
                        if isinstance(output[0], numpy.ndarray) else output[0].clone().detach().cpu())

    input_ids = last_state.view(batch_size * beam_size, -1).to(device)

    input_unfinished_sents_id = -3
    prev_step_results = (torch.from_numpy(output[-2]).to(device) if isinstance(output[-2], numpy.ndarray)
                                else output[-2].clone().detach().to(device))
    position_ids = (torch.tensor([context_length + step - 1
                                        ]).unsqueeze(0).repeat(batch_size * beam_size, 1).to(device))

    if prev_attention_mask.shape[0] != (batch_size * beam_size):
        prev_attention_mask = prev_attention_mask.repeat(batch_size * beam_size, 1)
    attention_mask = torch.cat(
        [
            prev_attention_mask,
            torch.ones([batch_size * beam_size, 1]).type_as(prev_attention_mask),
        ],
        1,
    ).to(device)

    beam_select_idx = (torch.from_numpy(output[input_unfinished_sents_id - 2]).to(device) if isinstance(
        output[input_unfinished_sents_id - 2], numpy.ndarray) else output[input_unfinished_sents_id - 2].clone().detach().to(device))
    input_log_probs = (torch.from_numpy(output[input_unfinished_sents_id - 1]).to(device) if isinstance(
        output[input_unfinished_sents_id - 1], numpy.ndarray) else output[input_unfinished_sents_id - 1].clone().detach().to(device))
    input_unfinished_sents = (torch.from_numpy(output[input_unfinished_sents_id]).to(device) if isinstance(
        output[input_unfinished_sents_id], numpy.ndarray) else
                                    output[input_unfinished_sents_id].clone().detach().to(device))
    prev_step_scores = (torch.from_numpy(output[-1]).to(device)
                                if isinstance(output[-1], numpy.ndarray) else output[-1].clone().detach().to(device))

    past = []
    if isinstance(output[1], tuple):  # past in torch output is tuple
        past = list(output[1])
    else:
        for i in range(model.config.n_layer):
            past_i = (torch.from_numpy(output[i + 1])
                        if isinstance(output[i + 1], numpy.ndarray) else output[i + 1].clone().detach())
            past.append(past_i.to(device)) 

    inputs = {
        'input_ids': input_ids,
        'attention_mask' : attention_mask,
        'position_ids': position_ids,
        'beam_select_idx': beam_select_idx,
        'input_log_probs': input_log_probs,
        'input_unfinished_sents': input_unfinished_sents,
        'prev_step_results': prev_step_results,
        'prev_step_scores': prev_step_scores,
    }
    ort_inputs = {
        'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy()),
        'attention_mask' : numpy.ascontiguousarray(attention_mask.cpu().numpy()),
        'position_ids': numpy.ascontiguousarray(position_ids.cpu().numpy()),
        'beam_select_idx': numpy.ascontiguousarray(beam_select_idx.cpu().numpy()),
        'input_log_probs': numpy.ascontiguousarray(input_log_probs.cpu().numpy()),
        'input_unfinished_sents': numpy.ascontiguousarray(input_unfinished_sents.cpu().numpy()),
        'prev_step_results': numpy.ascontiguousarray(prev_step_results.cpu().numpy()),
        'prev_step_scores': numpy.ascontiguousarray(prev_step_scores.cpu().numpy()),
    }
    for i, past_i in enumerate(past):
        ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past_i.cpu().numpy())
    
    return inputs, ort_inputs, past

def test_generation(tokenizer, input_text, use_onnxruntime_io, ort_session = None, num_tokens_to_produce = 30):
    print("Text generation using", "OnnxRuntime with IO binding" if use_onnxruntime_io else "OnnxRuntime", "...")    
    input_ids, attention_mask, position_ids, past = get_example_inputs(input_text)
    beam_select_idx = torch.zeros([1, input_ids.shape[0]]).long()
    input_log_probs = torch.zeros([input_ids.shape[0], 1])
    input_unfinished_sents = torch.ones([input_ids.shape[0], 1], dtype=torch.bool)
    prev_step_scores = torch.zeros([input_ids.shape[0], 1])
    inputs = {
        'input_ids': input_ids,
        'attention_mask' : attention_mask,
        'position_ids': position_ids,
        'beam_select_idx': beam_select_idx,
        'input_log_probs': input_log_probs,
        'input_unfinished_sents': input_unfinished_sents,
        'prev_step_results': input_ids,
        'prev_step_scores': prev_step_scores,
    }
    ort_inputs = {
        'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy()),
        'attention_mask' : numpy.ascontiguousarray(attention_mask.cpu().numpy()),
        'position_ids': numpy.ascontiguousarray(position_ids.cpu().numpy()),
        'beam_select_idx': numpy.ascontiguousarray(beam_select_idx.cpu().numpy()),
        'input_log_probs': numpy.ascontiguousarray(input_log_probs.cpu().numpy()),
        'input_unfinished_sents': numpy.ascontiguousarray(input_unfinished_sents.cpu().numpy()),
        'prev_step_results': numpy.ascontiguousarray(input_ids.cpu().numpy()),
        'prev_step_scores': numpy.ascontiguousarray(prev_step_scores.cpu().numpy()),
    }
    for i, past_i in enumerate(past):
        ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past_i.cpu().numpy())
    batch_size = input_ids.size(0)
    beam_size = 4
    context_length = input_ids.size(-1)

    for step in range(num_tokens_to_produce):
        if use_onnxruntime_io:
            outputs = inference_with_io_binding(ort_session, config, inputs['input_ids'], inputs['position_ids'], inputs['attention_mask'], past, inputs['beam_select_idx'], inputs['input_log_probs'], inputs['input_unfinished_sents'], inputs['prev_step_results'], inputs['prev_step_scores'], step, context_length)
        else:
            outputs = ort_session.run(None, ort_inputs) 
        inputs, ort_inputs, past = update(outputs, step, batch_size, beam_size, context_length, inputs['attention_mask'], device)

        if not inputs['input_unfinished_sents'].any():
            break

    print("------------")
    print(tokenizer.decode(inputs['prev_step_results'][0], skip_special_tokens=True))
```


```python
tokenizer = get_tokenizer(model_name_or_path, cache_dir)
input_text = EXAMPLE_Text
test_generation(tokenizer, input_text, use_onnxruntime_io=False, ort_session=session)
```

    Text generation using OnnxRuntime ...
    ------------
    best hotel in bay area.
    
    "It's a great place to stay," he said.



```python
test_generation(tokenizer, input_text, use_onnxruntime_io=True, ort_session=session)
```

    Text generation using OnnxRuntime with IO binding ...
    ------------
    best hotel in bay area.
    
    "It's a great place to stay," he said.



```python

```
