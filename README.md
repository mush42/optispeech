<div align="center">

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

</div>

<div align="center">

# OptiSpeech: Lightweight End-to-End text-to-speech model

**OptiSpeech** is ment to be an **efficient**, **lightweight** and **fast** text-to-speech model for **on-device** text-to-speech.

I would like to thank [Pneuma Solutions](https://pneumasolutions.com/) for providing GPU resources for training this model. Their support significantly accelerated my development process.
</div>

## Audio sample

https://github.com/user-attachments/assets/31350325-2062-47b3-bcb6-b803869553ce

Note that this is still WIP. Final model designed decisions are still being made.

## Installation

We use [Rye](https://rye.astral.sh/) to   manage the python runtime and dependencies.

[Install Rye first](https://rye.astral.sh/guide/installation/), then run the following:

```bash
$ git clone https://github.com/mush42/optispeech
$ cd optispeech
$ rye sync
```

## Inference

### Command line API

```bash
$ python3 -m optispeech.infer  --help
usage: infer.py [-h] [--d-factor D_FACTOR] [--p-factor P_FACTOR] [--e-factor E_FACTOR] [--cuda]
                checkpoint text output_dir

Speaking text using OptiSpeech

positional arguments:
  checkpoint           Path to OptiSpeech checkpoint
  text                 Text to synthesise
  output_dir           Directory to write generated audio to.

options:
  -h, --help           show this help message and exit
  --d-factor D_FACTOR  Scale to control speech rate
  --p-factor P_FACTOR  Scale to control pitch
  --e-factor E_FACTOR  Scale to control energy
  --cuda               Use GPU for inference
```

### Python API

```python
import soundfile as sf
from optispeech.model import OptiSpeech

# Load model
device = torch.device("cpu")
ckpt_path = "/path/to/checkpoint"
model = OptiSpeech.load_from_checkpoint(ckpt_path, map_location="cpu")
model = model.to(device)
model = model.eval()

# Text preprocessing and phonemization
sentence = "A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky."
inference_inputs = model.prepare_input(sentence)
inference_outputs = model.synthesize(inference_inputs)

inference_outputs = inference_outputs.as_numpy()
wav = inference_outputs.wav
sf.write("output.wav", wav.squeeze(), model.sample_rate)
```

## Training

Since this code uses [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template), you have all the powers that come with it.

Training is easy as 1, 2, 3:

### 1. Prepare Dataset

Given a dataset that is organized as follows:

```bash
├── train
│   ├── metadata.csv
│   └── wav
│       ├── aud-00001-0003.wav
│       └── ...
└── val
    ├── metadata.csv
    └── wav
        ├── aud-00764.wav
        └── ...
```

The `metadata.csv` file can contain 2, 3 or 4 columns delimited by **|** (bar character) in one of the following formats:

- 2 columns: file_id|text
- 3 columns: file_id|speaker_id|text
- 4 columns: file_id|speaker_id|language_id|text

Use the `preprocess_dataset` script to prepare the dataset for training:

```bash
$ python3 -m optispeech.tools.preprocess_dataset --help
usage: preprocess_dataset.py [-h] [--format {ljspeech}] dataset input_dir output_dir

positional arguments:
  dataset              dataset config relative to `configs/data/` (without the suffix)
  input_dir            original data directory
  output_dir           Output directory to write datafiles + train.txt and val.txt

options:
  -h, --help           show this help message and exit
  --format {ljspeech}  Dataset format.
```

If you are training on a new dataset, you must calculate and add **data_statistics ** using the following script:

```bash
$ python3 -m optispeech.tools.generate_data_statistics --help
usage: generate_data_statistics.py [-h] [-b BATCH_SIZE] [-f] [-o OUTPUT_DIR] input_config

positional arguments:
  input_config          The name of the yaml config file under configs/data

options:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Can have increased batch size for faster computation
  -f, --force           force overwrite the file
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory to save the data statistics
```

### 2. [Optional] Choose your backbone

**OptiSpeech** provides interchangeable types of backbones for the model's **encoder** and **decoder**, you choose the backbone based on your requirements.

To help you choose, here's a quick evaluation table of the available backbones:

| Backbone | Config File | FLOPs | MACs | #Params |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| ConvNeXt | `convnext_tts.yaml` | 13.78 GFLOPS | 6.88 GMACs | 17.43 M |
| Transformer | `optispeech.yaml` | 15.13 GFLOPS | 7.55 GMACs | 19.52 M |
| Conformer | `conformer_tts.yaml` | 19.96 GFLOPS | 9.95 GMACs | 25.89 M |
| LightSpeech | `lightspeech.yaml` | 9.6 GFLOPS | 4.78 GMACs | 13.29 M |

The default backbone is `Transformer`, but if you want to change it you can edit your experiment config.

### 3. Start training

To start training run the following command. Note that this training run uses **config** from [hfc_female-en_US](./configs/experiment/hfc_female-en_US.yaml). You can copy and update it with your own config values, and pass the name of the custom config file (without extension) instead.

```bash
$ python3 -m optispeech.train experiment=hfc_female-en_us
```

## ONNX support

### ONNX export

```bash
$ python3 -m optispeech.onnx.export --help
usage: export.py [-h] [--opset OPSET] [--seed SEED] checkpoint_path output

Export OptiSpeech checkpoints to ONNX

positional arguments:
  checkpoint_path  Path to the model checkpoint
  output           Path to output `.onnx` file

options:
  -h, --help       show this help message and exit
  --opset OPSET    ONNX opset version to use (default 15
  --seed SEED      Random seed
```

### ONNX inference

```bash
$ python3 -m optispeech.onnx.infer --help
usage: infer.py [-h] [--d-factor D_FACTOR] [--p-factor P_FACTOR] [--e-factor E_FACTOR] [--cuda]
                onnx_path text output_dir

ONNX inference of OptiSpeech

positional arguments:
  onnx_path            Path to the exported LeanSpeech ONNX model
  text                 Text to speak
  output_dir           Directory to write generated audio to.

options:
  -h, --help           show this help message and exit
  --d-factor D_FACTOR  Scale to control speech rate.
  --p-factor P_FACTOR  Scale to control pitch.
  --e-factor E_FACTOR  Scale to control energy.
  --cuda               Use GPU for inference
```

## Acknowledgements

Repositories I would like to acknowledge:

- [BetterFastspeech2](https://github.com/shivammehta25/betterfastspeech2): For repo backbone
- [LightSpeech](https://github.com/microsoft/NeuralSpeech/tree/master/LightSpeech): for the transformer backbone
- [JETS](https://github.com/espnet/espnet/tree/master/espnet2/gan_tts/jets): for the phoneme-mel alignment framework
- [Vocos](https://github.com/gemelo-ai/vocos/): For pioneering the use of ConvNext in TTS
- [Piper-TTS](https://github.com/rhasspy/piper): For leading the charge in on-device TTS. Also for the great phonemizer

## Reference

```
@inproceedings{luo2021lightspeech,
    title={Lightspeech: Lightweight and fast text to speech with neural architecture search},
    author={Luo, Renqian and Tan, Xu and Wang, Rui and Qin, Tao and Li, Jinzhu and Zhao, Sheng and Chen, Enhong and Liu, Tie-Yan},
    booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages={5699--5703},
    year={2021},
    organization={IEEE}
}

@article{siuzdak2023vocos,
  title={Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis},
  author={Siuzdak, Hubert},
  journal={arXiv preprint arXiv:2306.00814},
  year={2023}
}

@INPROCEEDINGS{10446890,
  author={Okamoto, Takuma and Ohtani, Yamato and Toda, Tomoki and Kawai, Hisashi},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Convnext-TTS And Convnext-VC: Convnext-Based Fast End-To-End Sequence-To-Sequence Text-To-Speech And Voice Conversion},
  year={2024},
  volume={},
  number={},
  pages={12456-12460},
  keywords={Vocoders;Neural networks;Signal processing;Transformers;Real-time systems;Acoustics;Decoding;ConvNeXt;JETS;text-to-speech;voice conversion;WaveNeXt},
  doi={10.1109/ICASSP48485.2024.10446890}
}
```

## Licence

Copyright (c) Musharraf Omer. MIT Licence. See [LICENSE](./LICENSE) for more details.
