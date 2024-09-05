import shutil
import sys
from pathlib import Path


HERE = Path(__file__).parent
PKG_DIR = HERE / "ospeech"
OPTISPEECH_PKG_DIR = HERE.parent.joinpath("optispeech")
FILE_MAP = {
    OPTISPEECH_PKG_DIR / "onnx/infer.py": PKG_DIR / "inference/__init__.py",
    OPTISPEECH_PKG_DIR / "text": PKG_DIR / "text",
    OPTISPEECH_PKG_DIR / "values.py": PKG_DIR / "values.py",
}


def main():
    for src, dst in FILE_MAP.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dst)
        else:
            if not dst.is_dir():
                shutil.copytree(src, dst)


if __name__ == '__main__':
    main()