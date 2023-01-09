<!---
Copyright 2021 NVIDIA Corporation. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Huggingface QDQRoBERTa Quantization Example

The QDQRoBERTa model adds fake quantization (pair of QuantizeLinear/DequantizeLinear ops) to:
 * linear layer inputs and weights
 * matmul inputs
 * residual add inputs

In this example, we use QDQRoBERTa model to do quantization on SQuAD task, including Quantization Aware Training (QAT) and Post Training Quantization (PTQ).

## Quantization Aware Training (QAT)

Calibrate the pretrained model and finetune with quantization awared:

```
python3 run_quant_qa.py \
  --model_name_or_path deepset/tinyroberta-squad2 \
  --tokenizer_name roberta-base \
  --dataset_name squad_v2 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --output_dir calib \
  --do_calib \
  --calibrator percentile \
  --percentile 99.99
```

```
python3 run_quant_qa.py \
  --model_name_or_path calib \
  --dataset_name squad_v2 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 4e-5 \
  --num_train_epochs 2 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --output_dir finetuned_int8 \
  --tokenizer_name roberta-base \
  --save_steps 0
```

### Export QAT model to ONNX

To export the QAT model finetuned above:

```
python3 run_quant_qa.py \
  --model_name_or_path finetuned_int8 \
  --output_dir ./ \
  --save_onnx \
  --per_device_eval_batch_size 1 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --dataset_name squad_v2 \
  --tokenizer_name roberta-base
```

## Post Training Quantization (PTQ)

### PTQ by calibrating and evaluating the FP32 model:

```
python3 run_quant_qa.py \
  --model_name_or_path deepset/tinyroberta-squad2 \
  --tokenizer_name roberta-base \
  --dataset_name squad_v2 \
  --calibrator percentile \
  --percentile 99.99 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --output_dir ./calib \
  --save_steps 0 \
  --do_calib \
  --do_eval
```

### Export the INT8 PTQ model to ONNX

```
python3 run_quant_qa.py \
  --model_name_or_path ./calib \
  --output_dir ./ \
  --save_onnx \
  --per_device_eval_batch_size 1 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --dataset_name squad_v2 \
  --tokenizer_name roberta-base 
```
