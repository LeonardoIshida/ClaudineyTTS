# ClaudineyTTS

This project provides a TTS (Text-to-Speech) model trained to replicate the voice of Claudiney Ferreira, the renowned radio broadcaster known for the program "Certas Palavras," which aired from the 1980s to 1990s. Learn more about the program [here](https://memoriaglobo.globo.com/exclusivo-memoria-globo/projetos-especiais/cbn-30-anos/noticia/cbn-30-anos-decada-de-1990.ghtml).

This project was developed and supervised by [NILC](https://sites.google.com/view/nilc-usp/), the Natural Language Processing research group at the University of São Paulo - ICMC.

Leveraging the [coquiTTS](https://github.com/coqui-ai/TTS) framework, the project utilized and fine-tuned the [VITS](https://arxiv.org/pdf/2106.06103) and [YourTTS](https://arxiv.org/abs/2112.02418) models, respectively.  Modifications were necessary to adapt the framework for a Docker environment.

The models were trained on the [Arandu cluster](https://github.com/C4AI/arandu_user_guide) provided by C4AI São Carlos.

Further details about the research can be found in this [paper](https://sol.sbc.org.br/index.php/stil/article/view/31120).

## Installation

You'll need [Docker](https://www.docker.com/) installed. All other dependencies are handled within the container.

Due to its size, the model must be downloaded manually from [Google Drive](https://drive.google.com/file/d/1AwkaFO-22Xh5qM9nR9Fo67e4Wi08kRM1/view) (approximately 4.3GB). Please verify the file size after downloading.

Once downloaded, unzip the model and place the resulting folder in the root directory of this repository.

This project supports two primary use cases:

- Fine-tuning the existing model
- Synthesizing speech audio

Both scenarios require cloning the repository and building the Docker image:

```bash
git clone https://github.com/LeonardoIshida/ClaudineyTTS.git
cd ClaudineyTTS
docker build -t claudiney-tts:latest . 
```

## Fine-tuning

```bash
docker run -it -v <absolute-path-to-your-data-folder>:/workspace/data claudiney-tts:latest  
python3 train_your.py
```

### Important Considerations:

- **Dataset Structure:**  Your dataset must adhere to the following structure:

```
datasets/
└── <dataset-name>/
    ├── wavs/
    │   └── <folder-with-audios>/
    ├── train-file.csv (CSV file containing audio paths and corresponding transcriptions)
    └── speakers.json (Speaker embedding file)
```

- **speakers.json:** Creating this file may require manual intervention. You might need to extract data from a `.pkl` file and format it as a JSON file.

- **Speaker ID Mismatch:** Ensure consistent speaker IDs between the `train-file.csv` and the `train_your.py` script to avoid errors.

- **Audio Format:** Audio files must be mono (not stereo) and have a sample rate of 16kHz.


## Synthesizing Speech

```bash
docker run -it -v <absolute-path-to-your-data-folder>:/workspace/data claudiney-tts:latest  
python3 inference_your.py
```

Modify the `inference_your.py` file to synthesize desired phrases, ensuring accurate transcriptions.

## Built With

[![My Skills](https://skillicons.dev/icons?i=python)](https://skillicons.dev)