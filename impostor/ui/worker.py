"""알고리즘 실행 스레드. 학습·추론이 UI를 멈추지 않도록 한다."""
import traceback

from PySide6.QtCore import QThread, Signal

from impostor.algorithms.base import Cancelled
from impostor.dataset import Dataset
from impostor.split import Split


class RunWorker(QThread):
    progress = Signal(str, float)
    algo_done = Signal(object)        # AlgoResult
    failed = Signal(str)

    def __init__(self, ds: Dataset, split: Split, algos: dict, parent=None):
        """algos: 이름 → 알고리즘 클래스"""
        super().__init__(parent)
        self.ds, self.split, self.algos = ds, split, algos
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        for name, cls in self.algos.items():
            if self._stop:
                break
            try:
                res = cls().run(self.ds, self.split, self.progress.emit, lambda: self._stop)
            except Cancelled:
                self.failed.emit("중단했습니다")
                return
            except Exception as e:   # 한 알고리즘 실패가 앱을 죽이면 안 된다
                traceback.print_exc()
                self.failed.emit(f"{name} 실행 실패: {e}")
                continue
            self.algo_done.emit(res)
