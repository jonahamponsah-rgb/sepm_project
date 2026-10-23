"""Copy the four existing input files into data/, retaining their exact bytes.

IDE users: set SOURCE_DIRECTORY below and run this file. No downloads occur.
"""
from pathlib import Path
import hashlib, json, shutil
SOURCE_DIRECTORY = Path.home() / "DataspellProjects" / "shga_ctgan" / "data"
def main():
    target = Path(__file__).resolve().parent / "data"
    expected = json.loads((target / "expected_sha256.json").read_text())
    for name, digest in expected.items():
        source = SOURCE_DIRECTORY / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing {source}; edit SOURCE_DIRECTORY or obtain the source dataset.")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"{name}: differs from the original input; check values, row order, headers and line endings.")
        destination = target / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        print(f"Verified {name}")
if __name__ == "__main__":
    main()
