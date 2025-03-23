# VoxEval
[![Hugging Face%20-%20VoxEval](https://img.shields.io/badge/🤗%20Hugging%20Face%20-%20VoxEval-blue)](https://huggingface.co/datasets/qqjz/VoxEval)
[![arXiv](https://img.shields.io/badge/arXiv-2501.04962-red)](https://arxiv.org/abs/2501.04962)

Github repository for paper: [*VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models*](https://arxiv.org/abs/2501.04962)

Also check out our survey paper at [*Recent Advances in Speech Language Models: A Survey*](https://arxiv.org/abs/2410.03751)!

## News
- **2025/03/23:** We release the evaluation code of VoxEval benchmark.
- **2025/02/26:** We release the Chain-of-Thought few-shot prompts for three mathematical reasoning subjects (Elementary Math, High School Math, and College Math) in VoxEval.

## Info
**VoxEval** is a novel speech question-answering benchmark specifically designed to assess SLMs' knowledge understanding through purely speech-based interactions.

Below are the three highlights of our VoxEval benchmark:
- End-to-end speech-based evaluation: Both input and output are audio-based.
- Diverse audio conditions: VoxEval includes audio files featuring a variety of speakers, speaking styles, and audio qualities.
- Complex spoken evaluations: It supports advanced spoken assessments, including spoken math tasks.

![Intro Figure](./figures/intro_figure.png)
<!-- <img src="./figures/intro_figure.png" alt="Intro Figure" width="400"> -->

## Download Data
You can access our **[VoxEval Dataset Repository](https://huggingface.co/datasets/qqjz/VoxEval)** on 🤗 Hugging Face to directly download the dataset.

Below is the layout of the dataset folder. `all_fewshot_examples` folder contains the few shot examples for the evaluation. `math_CoT_fewshot` folder contains the few shot examples for evaluating the math subjects via Chain-of-Thought prompting.
`test` folder contains the actual test data in VoxEval.
```
  # ├── Root Folder
  # │   ├── all_fewshot_examples
  # │       ├── alloy (different speaker voice)
  # │           ├── abstract_algebra_4o (different subjects)
  # │               ├── XXX.mp3
  # │               ├── ...
  # │           ├── ...
  # │       ├── echo
  # │       ├── ...
  # │   ├── test
  # │       ├── alloy (different settings)
  # │           ├── abstract_algebra_4o (different subjects)
  # │               ├── XXX.mp3
  # │               ├── ...
  # │           ├── ...
  # │       ├── echo
  # │       ├── ...
```

## Evaluation Pipeline
The `VoxEval_evaluation.py` script contains the evaluation framework for VoxEval. It allows you to integrate and assess any SLMs within it. To evaluate a specific SLM, follow these steps:
1. Clone the repository of the SLM you wish to evaluate and download the model checkpoints provided by its authors.
2. Set up a conda environment as per the instructions given by the SLM's authors.
3. Create a file within the SLM's codebase that includes a function named `e2e_evaluation(input_audio, sample_rate)`. This function should take audio samples and their sampling rate (loaded using torchaudio) as input. The input audio acts as an audio prompt, and the function should return the SLM's generated response audio.
4. Begin the evaluation process using the command provided below.
```shell
python VoxEval_evaluation.py \
    --source_path /path/to/VoxEval data/test \
    --fewshot_path /path/to/VoxEval data/all_fewshot_examples \
    --whisper_model_path /path/to/whisper-large-v3 \
    --target_path /path/to/save/the/evaluation/results \
    --eval_slm_path /path/to/target SLM codebase \
    --e2e_eval_file file_name_containing_the_e2e_evaluation_function \
    --timbre timbre \
    --shots num_shots \
    --prompt_mode regular_or_CoT \
    --cut_audio False
```

### GLM-4-Voice Evaluation (Example)
We give an example to evaluate GLM-4-Voice using VoxEval
1. Copy `./examples/VoxEval_eval_glm.py` into the original repository of [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice).
2. Set up the GLM conda environment and activate it.
3. Start the evaluation using the command below (change the paths to your own):
```shell
python VoxEval_evaluation.py \
    --source_path /nfsdata/cuiwenqian/VoxEval/test \
    --fewshot_path /nfsdata/cuiwenqian/VoxEval/all_fewshot_examples \
    --whisper_model_path /nfsdata/cuiwenqian/hf_model_ckpt/whisper-large-v3 \
    --target_path /nfsdata/cuiwenqian/GLM-4-Voice/VoxEval_evaluation \
    --eval_slm_path /nfsdata/cuiwenqian/GLM-4-Voice \
    --e2e_eval_file VoxEval_eval_glm \
    --timbre alloy \
    --shots 5 \
    --prompt_mode regular \
    --cut_audio False
```

## Evaluation results of existing end-to-end Spoken Language Models
| **SLMs**            | **SpeechGPT** | **TWIST** | **SPIRIT-LM** | **Moshi** | **GLM-4-Voice** |
|----------------------|---------------|-----------|---------------|-----------|-----------------|
| **Speakers**         |               |           |               |           |                 |
| Alloy                | 0.0001        | 0.0480    | 0.2084        | 0.1216    | 0.3763          |
| Echo                 | 0.0001        | 0.0558    | 0.2096        | 0.1221    | 0.3764          |
| Fable                | 0.0000        | 0.0116    | 0.2084        | 0.1153    | 0.3642          |
| Nova                 | 0.0001        | 0.0332    | 0.2070        | 0.1298    | 0.3677          |
| Onyx                 | 0.0002        | 0.0275    | 0.1966        | 0.1192    | 0.3764          |
| Shimmer              | 0.0000        | 0.0516    | 0.2076        | 0.1205    | 0.3815          |
| **Speaking Styles**  |               |           |               |           |                 |
| Linguistic           | 0.0001        | 0.0488    | 0.2044        | 0.1187    | 0.3643          |
| Speed                | 0.0001        | 0.0503    | 0.1911        | 0.1013    | 0.3469          |
| Pitch                | 0.0000        | 0.0544    | 0.1788        | 0.0609    | 0.3345          |
| **Audio Qualities**  |               |           |               |           |                 |
| Noise                | 0.0000        | 0.0368    | 0.1950        | 0.1018    | 0.3695          |
| Other Env Acoustics  | 0.0001        | 0.0434    | 0.2019        | 0.1051    | 0.3728          |
| **Underlying Text LMs** | **Llama-7B** | **Llama-7B** | **Llama-2-7B** | **Helium-7B** | **GLM-4-9B** |
| Text MMLU            | 0.3510        | 0.3510    | 0.4530        | 0.5430    | 0.7470          |

## What's Next
- The VoxEval evaluation code will be released soon.

## License
The dataset is licensed under the Creative Commons Attribution 4.0.

## Citation
```
@article{cui2025voxeval,
  title={VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models},
  author={Cui, Wenqian and Jiao, Xiaoqi and Meng, Ziqiao and King, Irwin},
  journal={arXiv preprint arXiv:2501.04962},
  year={2025}
}
```
