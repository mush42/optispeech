import  argparse
import os
from collections import namedtuple
from tempfile import SpooledTemporaryFile

import mureq
from tqdm import tqdm


ModelInfo = namedtuple("ModelInfo", "id name lang url")
MODEL_BASE_URL = "https://huggingface.co/mush42/optispeech/resolve/main/"
MODEL_LIST_URL = MODEL_BASE_URL + "models.json"
CHUNK_SIZE = 1024 * 1024


def get_models():
    resp = mureq.get(MODEL_LIST_URL)
    resp.raise_for_status()
    onnx_models = resp.json()["onnx"]
    models = []
    for lang, variants in onnx_models.items():
        for vname, speakers in variants.items():
            for speaker, subpath in speakers.items():
                id = f"{lang}-{vname}-{speaker}"
                name = f"{vname}-{speaker}"
                url = MODEL_BASE_URL + subpath
                models.append(ModelInfo(id, name, lang, url))
    return models


def format_as_table(*cols: tuple[str, list[str]]) -> str:
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)

    # Get formatting width of each column
    col_widths = []
    for c in cols:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], w) for c, w in zip(cols, col_widths)]

    # Summary = header + divider + Rest of table
    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, w in zip(cols, col_widths):
            line.append(s.format(str(c[1][i]), w))
        summary += "\n" + " | ".join(line)
    summary += "\n" + "-" * total_width
    return summary



def do_download_file(url):
    with mureq.yield_response('GET', url) as resp:
        if resp.status == 302:
            url = resp.getheader("Location")
            yield from do_download_file(url)
            return
        yield int(resp.headers["content-length"])
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            yield chunk

def list_command():
    models = get_models()
    cols = (
        ("Lang", [m.lang for m in models]),
        ("Speaker", [m.name for m in models]),
        ("ID", [m.id for m in models]),
    )
    print(format_as_table(*cols))


def download_command(id, dir):
    if not os.path.isdir(dir):
        print(f"The output directory {dir} does not exist.")
        return
    models = get_models()
    try:
        model = next(filter(lambda m: m.id == id, models))
    except StopIteration:
        print(f"A model with the given ID not found: {id}`")
        return
    streamer = do_download_file(model.url)
    try:
        model_size = next(streamer)
    except Exception as e:
        print("Failed to download model file.")
        raise e
    num_chunks = model_size // CHUNK_SIZE
    filename = f"{model.id}.onnx"
    output_file = os.path.abspath(
        os.path.join(dir, filename)
    )
    print(f"Downloading `{filename}`")
    with SpooledTemporaryFile() as temp_file:
        for chunk in tqdm(streamer, total=num_chunks, desc="Downloading", unit="MB"):
            temp_file.write(chunk)
        temp_file.seek(0)
        with open(output_file, "wb") as outfile:
            outfile.write(temp_file.read())
    print(f"Downloaded model to: {output_file}")
    


def main():
    parser = argparse.ArgumentParser(description="List and download ospeech models from HuggingFace.")
    subparsers = parser.add_subparsers(dest='command')
    ls_parser = subparsers.add_parser('ls', help='List available models')
    ls_parser.set_defaults(func=list_command)
    dl_parser = subparsers.add_parser('dl', help='Download ospeech models from HuggingFace')
    dl_parser.add_argument('id', type=str, help='Model ID. Run ospeech ls to list available models.')
    dl_parser.add_argument('dir', type=str, help='Directory to download the model to')
    dl_parser.set_defaults(func=download_command)
    args = parser.parse_args()
    if 'func' in args:
        if args.command == 'dl':
            args.func(args.id, args.dir)
        else:
            args.func()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
