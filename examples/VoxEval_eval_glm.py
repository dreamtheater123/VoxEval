import sys
sys.path.append('/nfsdata/cuiwenqian/GLM-4-Voice/third_party/Matcha-TTS')
import os
# cuda_num = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)

import torch
import numpy as np
from transformers import AutoTokenizer, WhisperFeatureExtractor, AutoModel, BitsAndBytesConfig
import uuid
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder


class SpeechProcessor:
    def __init__(
        self,
        model_path="./glm-4-voice-9b",
        tokenizer_path="./glm-4-voice-tokenizer",
        flow_path="./glm-4-voice-decoder",
        device="cuda",
        dtype="bfloat16"
    ):
        self.device = device
        
        # Initialize GLM model
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if dtype == "int4" else None

        print("Loading GLM model...")
        self.glm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=self.bnb_config if self.bnb_config else None,
            device_map={"": 0}
        ).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Initialize speech tokenizer
        print("Loading Whisper model...")
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

        # Initialize audio decoder
        print("Loading audio decoder...")
        flow_config = os.path.join(flow_path, "config.yaml")
        flow_checkpoint = os.path.join(flow_path, 'flow.pt')
        hift_checkpoint = os.path.join(flow_path, 'hift.pt')
        self.audio_decoder = AudioDecoder(
            config_path=flow_config,
            flow_ckpt_path=flow_checkpoint,
            hift_ckpt_path=hift_checkpoint,
            device=device
        )

    def generate_speech(
        self,
        input_audio_path,
        temperature=0.2,
        top_p=0.8,
        max_new_tokens=2000,
        output_path=None
    ):
        """
        Process speech file through the entire pipeline:
        1. Speech to tokens
        2. Language model processing
        3. Token to speech synthesis
        """
        # print("Extracting speech tokens...")
        # Extract speech tokens
        audio_tokens = extract_speech_token(
            self.whisper_model,
            self.feature_extractor,
            [input_audio_path]
        )[0]
        
        if len(audio_tokens) == 0:
            raise ValueError("No audio tokens extracted")
            
        # Format tokens for GLM
        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
        
        # Prepare prompt
        system_prompt = "User will provide you with a question. Please choose your answer from options A, B, C, or D. Please only output the final answer and in English. It should be in this format: The correct answer is {your_answer}."
        prompt = f"<|system|>\n{system_prompt}<|user|>\n{audio_tokens}<|assistant|>streaming_transcription\n"

        # print("Generating response with GLM...")
        # Generate with GLM
        with torch.no_grad():
            inputs = self.glm_tokenizer([prompt], return_tensors="pt").to(self.device)
            input_length = inputs['input_ids'].shape[-1]
            outputs = self.glm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            outputs = outputs[:, input_length:]  # Remove input tokens
            
        # Separate text and audio tokens
        generated_tokens = outputs[0].tolist()
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        text_tokens = []
        audio_tokens = []
        
        for token in generated_tokens:
            if token >= audio_offset:
                audio_tokens.append(token - audio_offset)
            else:
                text_tokens.append(token)

        # print("Synthesizing audio...")
        # Convert audio tokens to waveform
        audio_tensor = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)
        this_uuid = str(uuid.uuid4())  # Generate a unique ID for the audio file
        tts_speech, _ = self.audio_decoder.token2wav(  # TODO: solve the bug: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
            audio_tensor,
            uuid=this_uuid,
            prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
            prompt_feat=torch.zeros(1, 0, 80).to(self.device)
        )
        
        return tts_speech

SLM_processor = SpeechProcessor(
    model_path="/nfsdata/cuiwenqian/hf_model_ckpt/GLM-4-Voice/glm-4-voice-9b",  # NOTE: change to your own GLM ckpt path
    tokenizer_path="/nfsdata/cuiwenqian/hf_model_ckpt/GLM-4-Voice/glm-4-voice-tokenizer",
    flow_path="/nfsdata/cuiwenqian/hf_model_ckpt/GLM-4-Voice/glm-4-voice-decoder",
    device="cuda",
    dtype="bfloat16"
)


def e2e_evaluation(input_audio, sample_rate):
    result = SLM_processor.generate_speech(
        input_audio,
        temperature=0.2,
        top_p=0.8,
        max_new_tokens=500,
    )

    return result
