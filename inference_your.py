import argparse

import torch
import torchaudio
import os

import random
import shutil
from TTS.config import load_config
from TTS.tts.models.vits import Vits
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.models import setup_model
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager


#model
CONFIG_PATH = "/workspace/data/YourTTS-CML-TTS-CertasPalavras-August-16-2024_12+12AM-0000000/config.json"
MODEL_PATH = "/workspace/data/YourTTS-CML-TTS-CertasPalavras-August-16-2024_12+12AM-0000000/checkpoint_190000.pth"
OUTPUT_DIRECTORY = "/workspace/data/teste_inf"
D_VECTOR_FILES = "/workspace/data/CP/wavs/speakers.json" 
LANGUAGE_EMBEDDING = "/workspace/data/YourTTS-CML-TTS-CertasPalavras-August-16-2024_12+12AM-0000000/language_ids.json" 
MODEL_INFO = "YourTTS"

# Load the configuration
config = load_config(CONFIG_PATH)
config.model_args['d_vector_file'] = D_VECTOR_FILES
config.model_args['language_ids_file'] = None
config.model_args['use_speaker_encoder_as_loss'] = False

# Set device to CPU
device = torch.device("cpu")
print("Initiating YourTTS model on CPU!")
YOURTTS_MODEL = setup_model(config)
print("Loading YourTTS model!")
YOURTTS_MODEL.load_checkpoint(config, checkpoint_path=MODEL_PATH)

YOURTTS_MODEL.language_manager = LanguageManager(LANGUAGE_EMBEDDING)
cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

model_weights = cp['model'].copy()
for key in list(model_weights.keys()):
    if "speaker_encoder" in key:
        del model_weights[key]
    if "audio_transform" in key:
        del model_weights[key]

YOURTTS_MODEL.load_state_dict(model_weights)
YOURTTS_MODEL.eval()

# if use_cuda:
#     YOURTTS_MODEL = YOURTTS_MODEL.cuda()

lang = "brsp"
tts_text = "Que Hollywood é essa que você vai nos apresentar agora?"
audio_path = "/workspace/data/CP/wavs/data_sr16k/sr16k_mono_0014-CP518_242.194-245.876.wav"

accented_chars = {
    'á': '\u00e1',  # a with acute
    'é': '\u00e9',  # e with acute
    'í': '\u00ed',  # i with acute
    'ó': '\u00f3',  # o with acute
    'ú': '\u00fa',  # u with acute
    'à': '\u00e0',  # a with grave
    'è': '\u00e8',  # e with grave
    'ì': '\u00ec',  # i with grave
    'ò': '\u00f2',  # o with grave
    'ù': '\u00f9',  # u with grave
    'ä': '\u00e4',  # a with diaeresis
    'ë': '\u00eb',  # e with diaeresis
    'ï': '\u00ef',  # i with diaeresis
    'ö': '\u00f6',  # o with diaeresis
    'ü': '\u00fc',  # u with diaeresis
    'â': '\u00e2',  # a with circumflex
    'ê': '\u00ea',  # e with circumflex
    'î': '\u00ee',  # i with circumflex
    'ô': '\u00f4',  # o with circumflex
    'û': '\u00fb',  # u with circumflex
    'ñ': '\u00f1',  # n with tilde
    'ç': '\u00e7',  # c with cedilla
}

def replace_chars_with_unicode(sentence):
    for char, unicode_seq in accented_chars.items():
        sentence = sentence.replace(char, unicode_seq)
    return sentence

replace_chars_with_unicode(tts_text)

# Compute speaker embedding on CPU
speaker_embedding = YOURTTS_MODEL.speaker_manager.compute_embedding_from_clip(audio_path)

# Iterate over each text in the list
lang_id = 0
use_language_embedding = YOURTTS_MODEL.config.use_language_embedding or YOURTTS_MODEL.args.use_language_embedding or YOURTTS_MODEL.args.use_adaptive_weight_text_encoder
# if use_language_embedding:
#     lang_id = YOURTTS_MODEL.language_manager.name_to_id[lang]

wav, _, _, _ = synthesis(
        YOURTTS_MODEL,
        tts_text,
        YOURTTS_MODEL.config,
        "cuda" in str(next(YOURTTS_MODEL.parameters()).device),
        speaker_id=None,
        d_vector=speaker_embedding,
        style_wav=None,
        language_id=lang_id,
        use_griffin_lim=True,
        do_trim_silence=False,
    ).values()

# Save audio with a unique filename for each language
out_path = f"{OUTPUT_DIRECTORY}/{lang}/{lang}to{lang}-{MODEL_INFO}.wav"
#sample_rate = YOURTTS_MODEL.config.audio.sample_rate
#print(f"Sample_rate = {sample_rate}")
sample_rate = 16000
torchaudio.save(out_path, torch.tensor(wav).unsqueeze(0), sample_rate)
print(f"Saved: {out_path}")

ground_truth_copy_path = f"{OUTPUT_DIRECTORY}/{lang}/ground_truth.wav"
shutil.copy(audio_path, ground_truth_copy_path)