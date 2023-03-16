# InstructLLaMa.cpp

Inference of [InstructLLaMA](https://arxiv.org/abs/2302.13971) model.

**Dev-notes:** We are switching away from [our C++ implementation of LLaMa](https://github.com/NolanoOrg/llama-int4-quant) to the more recent one [llama.cpp](https://github.com/ggerganov/llama.cpp) by [@ggerganov](https://github.com/ggerganov) that now offers nearly same performance (and output quality) on Macbook as well as support over Linux and Windows.

Supported platforms: Mac OS, Linux, Windows (via CMake)

License: MIT

If you use LLaMa weights, then it should only be used for non-commercial research purposes.

## Description

Here is a typical run using the adapter weights uploaded by `tloen/alpaca-lora-7b` under MIT license:

```java
make -j && ./main -m ./models/7B/ggml-model-q4_0.bin --instruction "Write an email to your friend about your plans for the weekend." -t 8 -n 128
```

```java
make -j && ./main -m ./models/7B/ggml-model-q4_0.bin --instruction "Calculate the area of the a circle given its radius." --input "radius = 3" -t 8 -n 128
```

These follow the Stanford's Alpaca format for instruction prompt (https://github.com/tatsu-lab/stanford_alpaca#data-release)

## Usage

Here are the step for the LLaMA-7B model (same as llama.cpp), defaults to the adapter weights uploaded by `tloen/alpaca-lora-7b` under MIT license:

```bash
# build this repo
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# obtain the original LLaMA model weights and place them in ./models
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model

# install Python dependencies
python3 -m pip install torch numpy sentencepiece transformers

# convert the 7B model to ggml FP16 format
python3 convert-pth-to-ggml.py models/7B/ 1

# quantize the model to 4-bits
./quantize.sh 7B

# run the inference
./main -m ./models/7B/ggml-model-q4_0.bin -t 8 -n 128 --instruction <instruction> --input <input_to_instruction>
```

