# ospeech

Minimum dependency inference library for [OptiSpeech TTS model](https://github.com/mush42/optispeech).

## About **OptiSpeech**

**[OptiSpeech](https://github.com/mush42/optispeech)** is ment to be an **efficient**, **lightweight** and **fast** text-to-speech model for **on-device** text-to-speech.

## Install

This package is pip installable:

```
$ pip install ospeech
```

Most models are trained with IPA phonemized text. To use these models, install `ospeech` with the `espeak` feature, which pulls-in [piper-phonemize](https://github.com/rhasspy/piper-phonemize):

```
pip install ospeech[espeak]
```

If you want a gradio interface, install with the `gradio` feature:

```
pip install ospeech[gradio]
```

## Usage

### Command line usage

```
$ ospeech --help
usage: ospeech [-h] [--d-factor D_FACTOR] [--p-factor P_FACTOR] [--e-factor E_FACTOR] [--no-split] [--cuda]
               onnx_path text output_dir

ONNX inference of OptiSpeech

positional arguments:
  onnx_path            Path to the exported OptiSpeech ONNX model
  text                 Text to speak
  output_dir           Directory to write generated audio to.

options:
  -h, --help           show this help message and exit
  --d-factor D_FACTOR  Scale to control speech rate.
  --p-factor P_FACTOR  Scale to control pitch.
  --e-factor E_FACTOR  Scale to control energy.
  --no-split           Don't split input text into sentences.
  --cuda               Use GPU for inference
```

If you want to run with the gradio interface:

```
$ ospeech-gradio --help
usage: ospeech-gradio [-h] [-s] [--host HOST] [--port PORT] [--char-limit CHAR_LIMIT] onnx_file_path

positional arguments:
  onnx_file_path        Path to model ONNX file

options:
  -h, --help            show this help message and exit
  -s, --share           Generate gradio share link
  --host HOST           Host to serve the app on.
  --port PORT           Port to serve the app on.
  --char-limit CHAR_LIMIT
                        Input text character limit.
```

### Python API

```python
import soundfile as sf
from ospeech import OptiSpeechONNXModel


model_path = "./optispeech-en-us-lightspeech.onnx"
sentence = "OptiSpeech is awesome!"

model = OptiSpeechONNXModel.from_onnx_file_path(model_path)
model_inputs= onx.prepare_input(sentence)
outputs = onx.synthesise(model_inputs)

for (idx, wav) in enumerate(outputs):
    # Wav is a float array
    sf.write(f"output-{idx}.wav", wav, model.sample_rate)
```


## Licence

Copyright (c) Musharraf Omer. MIT Licence. See [LICENSE](./LICENSE) for more details.
