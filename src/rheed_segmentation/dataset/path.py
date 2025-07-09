from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelPairPath:
    """画像ファイルとJSONファイルのパスをペアで保持"""

    image_path: Path
    json_path: Path
    filename: str | None = None

    def __post_init__(self) -> None:
        if self.filename is None:
            object.__setattr__(self, "filename", self.image_path.name)
