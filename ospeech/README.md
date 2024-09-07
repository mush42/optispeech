# ospeech

Minimum dependency inference library for [OptiSpeech TTS model](https://github.com/mush42/optispeech).

## About **OptiSpeech**

**[OptiSpeech](https://github.com/mush42/optispeech)** is ment to be an **efficient**, **lightweight** and **fast** text-to-speech model for **on-device** text-to-speech.

## Install

This package can be installed using `pip`:

```
$ pip install ospeech
```

If you want to run the `ospeech` command from anywhere, try:

```
$ pipx install ospeech
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

### Obtaining models

```
$ ospeech-models --help
usage: ospeech-models [-h] {ls,dl} ...

List and download ospeech models from HuggingFace.

positional arguments:
  {ls,dl}
    ls        List available models
    dl        Download ospeech models from HuggingFace

options:
  -h, --help  show this help message and exit
```
To list available models:

```
$ ospeech-models ls
Lang  | Speaker                 | ID
---------------------------------------------------------------------
en-us | lightspeech-hfc-female  | en-us-lightspeech-hfc-female
en-us | convnext-tts-hfc-female | en-us-convnext-tts-hfc-female
---------------------------------------------------------------------
```

Using the model ID, use the following command to download a model:

```
$ ospeech-models dl en-us-lightspeech-hfc-female .
Downloading `en-us-lightspeech-hfc-female.onnx`
Downloading:   100%|                                                                           | 38/38 [00:02<?, ?MB/s]
```


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
model_inputs= model.prepare_input(sentence)
outputs = model.synthesise(model_inputs)

for (idx, wav) in enumerate(outputs):
    # Wav is a float array
    sf.write(f"output-{idx}.wav", wav, model.sample_rate)
```


## Licence

Copyright (c) Musharraf Omer. MIT Licence. See [LICENSE](./LICENSE) for more details.
