import datetime
import json
from dataclasses import dataclass
from pathlib import Path

RESULTS_ROOT_DIR = "results"
TIMEZOME_JST = datetime.timezone(datetime.timedelta(hours=9))


@dataclass
class ResultDir:
    def __init__(self, date_dir: Path, protocol: str) -> None:
        self.date_dir = date_dir
        self.protocol = protocol

    @classmethod
    def create(cls, date_dir: Path, protocol: str) -> "ResultDir":
        result_dir = cls(date_dir, protocol)

        result_dir.path.mkdir(parents=True, exist_ok=True)
        return result_dir

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[date_dir={self.date_dir}, protocol={self.protocol}]"

    def write_history_file(self, history_data: dict) -> None:
        with self.history_path.open(mode="a", encoding="utf-8") as f:
            json.dump(history_data, f)
            f.write("\n")

    @property
    def path(self) -> Path:
        return self.date_dir / self.protocol

    @property
    def history_path(self) -> Path:
        return self.path / "history.jsonl"


class ResultDateDir:
    date_format = "%Y%m%d%H%M%S"

    def __init__(self, root_dir: str | Path, date: datetime.datetime | str) -> None:
        self.root_dir = Path(root_dir)

        if isinstance(date, str):
            date = datetime.datetime.strptime(date, self.date_format)  # noqa: DTZ007
        self.date = date

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[root_dir={self.root_dir}, date={self.date_str}]"

    @classmethod
    def create(
        cls, root_dir: str | Path, date: datetime.datetime | str | None = None
    ) -> "ResultDateDir":
        if date is None:
            date = datetime.datetime.now(TIMEZOME_JST)

        result_date_dir = cls(root_dir, date)
        result_date_dir.path.mkdir(parents=True, exist_ok=True)
        return result_date_dir

    def create_protocol_dir(self, protocol: str) -> ResultDir:
        return ResultDir.create(self.path, protocol)

    def fetch_protocol_dirs(self) -> list[ResultDir]:
        dirs = self.path.iterdir()
        return [ResultDir(self.path, dir_path.name) for dir_path in dirs if dir_path.is_dir()]

    def fetch_protocol_dir_by_name(self, protocol: str) -> ResultDir | None:
        protocol_dirs = self.fetch_protocol_dirs()
        return next(
            filter(lambda protocol_dir: protocol_dir.protocol == protocol, protocol_dirs), None
        )

    @property
    def date_str(self) -> str:
        return self.date.strftime(self.date_format)

    @property
    def path(self) -> Path:
        return self.root_dir / self.date_str


class ResultDirManager:
    def __init__(self, root_dir: str | Path = RESULTS_ROOT_DIR) -> None:
        self.root_dir = Path(root_dir)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[root_dir={self.root_dir}]"

    def fetch_result_dirs(self) -> list[ResultDateDir]:
        dirs = self.root_dir.iterdir()
        return [
            ResultDateDir(self.root_dir, result_dir.name)
            for result_dir in dirs
            if result_dir.is_dir()
        ]

    def latest_result_dir(self) -> ResultDateDir | None:
        log_files = self.fetch_result_dirs()
        if len(log_files) == 0:
            return None

        return log_files[-1]

    def create_date_dir(self, date: datetime.datetime | None = None) -> ResultDateDir:
        if date is None:
            date = datetime.datetime.now(TIMEZOME_JST)

        return ResultDateDir.create(root_dir=self.root_dir, date=date)
