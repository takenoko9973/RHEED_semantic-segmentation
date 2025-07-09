import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path

RESULTS_ROOT_DIR = "results"
TIMEZOME_JST = datetime.timezone(datetime.timedelta(hours=9))


@dataclass
class ResultDir:
    def __init__(self, date_dir_path: Path, protocol: str) -> None:
        self.date_dir_path = date_dir_path
        self.protocol = protocol

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[date_dir={self.date_dir_path}, protocol={self.protocol}]"

    def create_dir(self) -> None:
        self.path.mkdir(exist_ok=True)

    def write_history_file(self, history_data: dict) -> None:
        with self.history_path.open(mode="a", encoding="utf-8") as f:
            json.dump(history_data, f)
            f.write("\n")

    @property
    def path(self) -> Path:
        return self.date_dir_path / self.protocol

    @property
    def history_path(self) -> Path:
        return self.path / "history.jsonl"


class ResultDateDir:
    DATE_FORMAT = "%Y%m%d%H%M%S"
    FOLDER_NAME_PATTERN = re.compile(r"^(\d{14})(?:-(.+))?$")  # YYYYMMDDhhmmss(-name) 形式を解析

    def __init__(
        self, root_dir: str | Path, date: datetime.datetime, name: str | None = None
    ) -> None:
        self.root_dir = Path(root_dir)
        self.date = date
        self.name = name

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"[root_dir={self.root_dir}, date={self.date}, name={self.name}]"
        )

    def __lt__(self, other: "ResultDateDir") -> bool:
        # 主に日付でソートし、日付が同じ場合は名前でソート
        if self.date != other.date:
            return self.date < other.date
        return (self.name or "") < (other.name or "")

    @classmethod
    def from_dir_name(cls, root_dir: str, dir_name: str) -> "ResultDateDir":
        match = cls.FOLDER_NAME_PATTERN.match(dir_name)
        if not match:
            msg = f"フォルダ名 '{dir_name}' は有効な日付フォルダのフォーマットではありません。"
            raise ValueError(msg)

        date_str = match.group(1)
        name = match.group(2)  # nameはNoneまたは文字列

        try:
            date_obj = datetime.datetime.strptime(date_str, cls.DATE_FORMAT)  # noqa: DTZ007
        except ValueError as e:
            msg = f"フォルダ名 '{dir_name}' から日付を解析できませんでした。"
            raise ValueError(msg) from e

        return cls(root_dir, date_obj, name)

    def create_dir(self) -> None:
        self.path.mkdir(exist_ok=True)

    def create_protocol_dir(self, protocol: str) -> ResultDir:
        result_dir = ResultDir(self.path, protocol)
        result_dir.create_dir()
        return result_dir

    def fetch_protocol_dirs(self) -> list[ResultDir]:
        """ディレクトリ内の各プロトコル結果ディレクトリを取得"""
        dirs = self.path.iterdir()
        return [ResultDir(self.path, dir_path.name) for dir_path in dirs if dir_path.is_dir()]

    def fetch_protocol_dir_by_name(self, protocol: str) -> ResultDir | None:
        protocol_dirs = self.fetch_protocol_dirs()
        return next(
            filter(lambda protocol_dir: protocol_dir.protocol == protocol, protocol_dirs), None
        )

    @property
    def date_str(self) -> str:
        return self.date.strftime(self.DATE_FORMAT)

    @property
    def dir_name(self) -> str:
        if self.name is None or self.name == "":
            return f"{self.date_str}"

        return f"{self.date_str}-{self.name}"

    @property
    def path(self) -> Path:
        return self.root_dir / self.dir_name


class ResultDirManager:
    def __init__(self, result_path: str | Path = RESULTS_ROOT_DIR) -> None:
        self.result_path = Path(result_path)
        self.result_path.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[root_dir={self.result_path}]"

    def get_result_dirs(self) -> list[ResultDateDir]:
        date_dirs = []

        for dir_path in self.result_path.iterdir():
            if not dir_path.exists():
                continue

            try:
                # フォルダ名からResultDateManagerを生成試行
                date_dir = ResultDateDir.from_dir_name(self.result_path, dir_path.name)
                date_dirs.append(date_dir)
            except ValueError:
                # 有効な日付フォルダ名でない場合はスキップ
                continue

        return sorted(date_dirs, reverse=True)

    def get_latest_result_dir(self) -> ResultDateDir | None:
        result_dirs = self.get_result_dirs()
        return next(iter(result_dirs), None)

    def get_result_dir_from_name(self, name: str) -> ResultDateDir:
        result_dirs = self.get_result_dirs()
        return next(filter(lambda result_dir: result_dir.dir_name == name, result_dirs), None)

    def create_date_dir(self, name: str | None = None) -> ResultDateDir:
        current_date = datetime.datetime.now(TIMEZOME_JST)

        new_date_dir = ResultDateDir(self.result_path, current_date, name)
        new_date_dir.create_dir()
        return new_date_dir
