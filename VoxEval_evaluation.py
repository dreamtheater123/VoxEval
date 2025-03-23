import os
cuda_num = 7
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)

import torchaudio
import torch
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import uuid
import argparse
import importlib

def parse_args():
    parser = argparse.ArgumentParser(description="VoxEval evaluation script")
    
    # Paths
    parser.add_argument("--source_path", type=str, default="/nfsdata/cuiwenqian/VoxEval/test",
                        help="Path to source test data")
    parser.add_argument("--fewshot_path", type=str, default="/nfsdata/cuiwenqian/VoxEval/all_fewshot_examples",
                        help="Path to few-shot examples")
    parser.add_argument("--whisper_model_path", type=str, default="/nfsdata/cuiwenqian/hf_model_ckpt/whisper-large-v3",
                        help="Path to whisper model checkpoint")
    parser.add_argument("--target_path", type=str, default="/nfsdata/cuiwenqian/GLM-4-Voice/VoxEval_evaluation",
                        help="Path to save evaluation results")
    parser.add_argument("--eval_slm_path", type=str, required=True,
                    help="Path to the code base of the evaluated model (the evaluated SLM's code base path)")
    parser.add_argument("--e2e_eval_file", type=str, required=True,
                    help="File name to import e2e_evaluation from")
    
    # Parameters
    parser.add_argument("--timbre", type=str, default="alloy", 
                        choices=["alloy", "echo", "fable", "nova", "onyx", "shimmer", "alloy_noise", "alloy_pitch", "alloy_tempo", "alloy_env_acoustics", "alloy_linguistic_variation"],
                        help="Timbre option to use")
    parser.add_argument("--shots", type=int, default=5,
                        help="Number of few-shot examples")
    parser.add_argument("--prompt_mode", type=str, default="regular", choices=["CoT", "regular"],
                        help="Prompt mode: Chain of Thought or regular")
    parser.add_argument("--cut_audio", type=bool, default=True,
                        help="Whether to cut audio")
    parser.add_argument("--save_folder", type=str, default=None,
                        help="Custom folder name for saving results (if not specified, will be generated automatically)")
    
    args = parser.parse_args()
    
    # Auto-configure based on prompt_mode if needed
    if args.prompt_mode == "CoT" and args.save_folder is None:
        args.shots = 3
        args.fewshot_path = args.fewshot_path.replace("all_fewshot_examples", "math_CoT_fewshot")
        args.save_folder = f"CoT_{args.timbre}_{args.shots}shot_cut_{args.cut_audio}"
    elif args.save_folder is None:
        args.save_folder = f"{args.timbre}_{args.shots}shot"
        
    return args


args = parse_args()
import sys
sys.path.append(args.eval_slm_path)
module = importlib.import_module(args.e2e_eval_file)
e2e_evaluation = module.e2e_evaluation

# load the whisper model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    args.whisper_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
whisper_model.to(device)
processor = AutoProcessor.from_pretrained(args.whisper_model_path)
pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)
print("Whisper model loaded")

def whisper_inference(pipe, waveform):
    waveform = waveform
    if waveform.ndim > 1:
        waveform = waveform[0, :]
    result = pipe(waveform)["text"]

    return result


def concat_question_answer(question_i, answer_i):
    prompt_q_audio, sample_rate = torchaudio.load(question_i)
    prompt_a_audio, sample_rate = torchaudio.load(answer_i)
    # concatenate the question and answer audios
    silence_duration = 1  # in seconds
    num_channels = prompt_q_audio.shape[0]
    silence = torch.zeros((num_channels, sample_rate * silence_duration))
    prompt_audio = torch.cat((prompt_q_audio, silence, prompt_a_audio), dim=1)

    return prompt_audio, sample_rate


def cli_main():
    # traverse all the categories of MMLU
    if args.prompt_mode == "CoT":
        subject_list = ["elementary_mathematics_4o.csv", "high_school_mathematics_4o.csv", "college_mathematics_4o.csv"]
    else:
        subject_list = []
        for item in os.listdir(args.source_path):
            if item.endswith(".csv"):
                subject_list.append(item)
    print(subject_list, len(subject_list))

    failed_dialogs = []
    for subject in subject_list:
        folder = subject.split(".csv")[0]
        print(f"Processing folder {folder}")
        if args.prompt_mode == "CoT":
            fewshot_folder = folder.replace("_4o", "_dev_4o")
        else:
            fewshot_folder = folder.replace("_test", "_val")
        save_path = os.path.join(args.target_path, args.save_folder, subject)
        # check if the file already exists
        if os.path.exists(save_path):
            print(f"File {save_path} already exists")
            continue

        prompt_audio = None
        for i in range(args.shots):  # concat question and answer audio for the few-shot prompt
            formatted_i = "%08d" % i
            question_i = f"{formatted_i}_question.mp3"
            if args.prompt_mode == "CoT":
                answer_i = f"{formatted_i}_CoT_answer.mp3"
            else:
                answer_i = f"{formatted_i}_answer.mp3"
            question_i = os.path.join(args.fewshot_path, args.timbre, fewshot_folder, question_i)
            answer_i = os.path.join(args.fewshot_path, args.timbre, fewshot_folder, answer_i)

            if i == 0:
                prompt_audio, sample_rate = concat_question_answer(question_i, answer_i)
            else:
                qa_audio, sample_rate = concat_question_answer(question_i, answer_i)
                # concatenate the question-answer audio with the previous prompt audio
                silence_duration = 1
                num_channels = prompt_audio.shape[0]
                silence = torch.zeros((num_channels, sample_rate * silence_duration))
                prompt_audio = torch.cat((prompt_audio, silence, qa_audio), dim=1)

        # open the subject csv file as dataframe
        df = pd.read_csv(os.path.join(args.source_path, subject), header=None)
        # create an empty column for df
        df[6] = None
        print(df)
        # traverse every row in the dataframe
        print(f"start processing questions")
        for i in tqdm(range(len(df))):
            print(f"Processing timbre {args.timbre} subject {subject}, question {i}, cuda_num: {cuda_num}")
            formatted_i = "%08d" % i
            question_i = f"{formatted_i}_question.mp3"
            answer_i = f"{formatted_i}_answer.mp3"
            question_i = os.path.join(args.source_path, args.timbre, folder, question_i)
            answer_i = os.path.join(args.source_path, args.timbre, folder, answer_i)
            # judge if the both question_i and answer_i exists
            if not os.path.exists(question_i) or not os.path.exists(answer_i):
                print(f"Question or answer does not exist for {i}")
                failed_dialogs.append([subject, i, "input audio not found"])
                continue
            
            # load the question audio
            try:
                question_audio, sample_rate = torchaudio.load(question_i)
            except Exception as e:
                print(f"Error: {e}")
                failed_dialogs.append([subject, i, "question audio load error"])
                continue
            # concat prompt audio and question audio
            silence_duration = 1
            num_channels = prompt_audio.shape[0]
            silence = torch.zeros((num_channels, sample_rate * silence_duration))
            input_audio = torch.cat((prompt_audio, silence, question_audio), dim=1)
            
            # cut the input_audio to its last 80 seconds
            input_audio = input_audio[:, -80 * sample_rate:]
            # generate the end-to-end audio
            tik = time.time()
            try:
                generated_audio = e2e_evaluation((input_audio, sample_rate), sample_rate)
            except Exception as e:
                print(f"Error: {e}")
                failed_dialogs.append([subject, i, "e2e evaluation error"])
                continue
            # # NOTE: if you'd like to save the generated_audio, uncomment the following two lines
            # torchaudio.save("generated_audio.wav", generated_audio.cpu(), 22050)
            # print("generated_audio shape: ", generated_audio.shape)
            generated_audio = generated_audio.squeeze().cpu().numpy()
            generated_transcription = whisper_inference(pipe, generated_audio)
            # save generated_transcription to last column of the df
            df.iloc[i, -1] = generated_transcription

        # save the df to a new csv file
        os.makedirs(os.path.join(args.target_path, args.save_folder), exist_ok=True)
        df.to_csv(save_path, header=None, index=None)
    
    print(f"Failed dialogs len: {len(failed_dialogs)}")
    # save the failed dialogs
    with open(os.path.join(args.target_path, args.save_folder, "failed_dialogs.txt"), "w") as f:
        for item in failed_dialogs:
            f.write(f"{item}\n")


if __name__ == "__main__":
    cli_main()
















