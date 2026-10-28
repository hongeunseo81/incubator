"""알고리즘 실행 스레드. 학습·추론이 UI를 멈추지 않도록 한다."""
import traceback

from PySide6.QtCore import QThread, Signal

from impostor.algorithms.base import AlgoResult, Cancelled
from impostor.readiness import Snapshot


class RunWorker(QThread):
    progress = Signal(str, float)
    done = Signal(object)        # AlgoResult — 실패·중단도 skipped 사유로 보낸다

    def __init__(self, snap: Snapshot, algo_name: str, algo_cls, params: dict, parent=None):
        super().__init__(parent)
        self.snap, self.algo_name, self.algo_cls, self.params = snap, algo_name, algo_cls, params
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            res = self.algo_cls(self.params).run(self.snap.ds, self.snap.split, self.progress.emit,
                                                 lambda: self._stop)
        except Cancelled:
            res = AlgoResult(self.algo_name, skipped="중단했습니다")
        except Exception as e:   # 실패해도 앱은 살아 있고, 사유는 결과 화면에 남긴다
            traceback.print_exc()
            res = AlgoResult(self.algo_name, skipped=f"실행 실패: {e}")
        self.done.emit(res)
