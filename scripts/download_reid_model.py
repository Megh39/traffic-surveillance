from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/"
    "public/2022.1/vehicle-reid-0001/"
    "osnet_ain_x1_0_vehicle_reid.onnx"
)
EXPECTED_SHA384 = (
    "0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166"
    "343157273c02f3b0fede3050ef7abb7"
)
DESTINATION = (
    Path(__file__).resolve().parents[1]
    / "models" / "reid" / "osnet_ain_x1_0_vehicle_reid.onnx"
)


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(DESTINATION.suffix + ".download")

    print(f"Downloading official vehicle Re-ID weights to: {DESTINATION}")
    try:
        urllib.request.urlretrieve(URL, temporary)
        digest = hashlib.sha384(temporary.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA384:
            raise RuntimeError(
                "Downloaded model checksum does not match the official "
                "Open Model Zoo model metadata. The file will not be accepted."
            )
        temporary.replace(DESTINATION)
        print(f"Verified model: {DESTINATION}")
        print(f"SHA-384: {digest}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
