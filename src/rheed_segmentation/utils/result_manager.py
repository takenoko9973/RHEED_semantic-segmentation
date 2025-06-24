import datetime
import json
import re as regex
from dataclasses import dataclass
from pathlib import Path

RESULTS_ROOT_DIR = "results"
TIMEZOME_JST = datetime.timezone(datetime.timedelta(hours=9))


@dataclass
class ResultDir:
    date_format = "%Y%m%d%H%M%S"

    def __init__(
        self,
        protocol: str,
        date: datetime.datetime,
        root_dir: str | Path = RESULTS_ROOT_DIR,
    ) -> None:
        self.protocol = protocol
        self.date = date
        self.root_dir = Path(root_dir)

        self.path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_file_name(cls, file_name: str, root_dir: str | Path = RESULTS_ROOT_DIR) -> "ResultDir":
        pattern = regex.compile(r"(\d{14})\-(.+)$")
        match = pattern.match(file_name)
        if match is None:
            msg = f"Invalid file name format: {file_name}"
            raise ValueError(msg)

        date = datetime.datetime.strptime(match.group(1), cls.date_format).astimezone(TIMEZOME_JST)
        return cls(
            protocol=match.group(2),
            date=date,
            root_dir=root_dir,
        )

    @classmethod
    def from_file_path(cls, file_path: str | Path) -> "ResultDir":
        file_name = Path(file_path).name
        log_dir = Path(file_path).parent

        return cls.from_file_name(file_name, log_dir)

    def write_history_file(self, history_data: dict) -> None:
        file_path = self.path / "history.jsonl"

        with file_path.open(mode="a", encoding="utf-8") as f:
            json.dump(history_data, f)
            f.write("\n")

    @property
    def name(self) -> str:
        date_str = self.date.strftime(self.date_format)
        return f"{date_str}-{self.protocol}"

    @property
    def path(self) -> Path:
        return self.root_dir / self.name


class ResultDirManager:
    def __init__(self, root_dir: str | Path = RESULTS_ROOT_DIR) -> None:
        self.root_dir = Path(root_dir)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[root_dir={self.root_dir}]"

    def get_result_dirs(self) -> list[ResultDir]:
        dirs = self.root_dir.iterdir()
        dirs = [ResultDir.from_file_path(result_dir) for result_dir in dirs]
        return sorted(dirs, key=lambda x: x.date)

    def get_latest_result_dir(self) -> ResultDir | None:
        log_files = self.get_result_dirs()

        if len(log_files) == 0:
            return None

        return log_files[-1]

    def create_result_dir(self, protocol: str, date: datetime.datetime | None = None) -> ResultDir:
        if date is None:
            date = datetime.datetime.now(TIMEZOME_JST)

        return ResultDir(
            protocol=protocol,
            date=date,
            root_dir=self.root_dir,
        )
