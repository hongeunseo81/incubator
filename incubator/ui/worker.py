"""알고리즘 실행 스레드. 학습·추론이 UI를 멈추지 않도록 한다."""
import shutil
import traceback

from PySide6.QtCore import QThread, Signal

from incubator.algorithms.base import AlgoResult, Cancelled
from incubator.export import export_bundle, verify
from incubator.readiness import Snapshot


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


class ExportWorker(QThread):
    """모델 묶음을 쓰고 ONNX 로 다시 채점한다. ONNX 변환·CPU 재채점이 수십 초 걸릴 수 있다."""
    progress = Signal(str, float)
    done = Signal(object)        # 성공 (묶음 경로, Verification), 실패 사유 str

    def __init__(self, res: AlgoResult, snap: Snapshot, threshold: float, metrics, out_root: str, params: dict,
                 parent=None):
        super().__init__(parent)
        self.res, self.snap, self.threshold, self.metrics = res, snap, threshold, metrics
        self.out_root, self.params = out_root, params

    def run(self) -> None:
        try:
            out = export_bundle(self.res, self.snap, self.threshold, self.metrics, self.out_root, self.progress.emit,
                                self.params)
        except Exception as e:
            traceback.print_exc()
            self.done.emit(f"내보내기 실패: {e}")
            return
        try:
            self.done.emit((out, verify(out, self.res, self.snap, self.progress.emit)))
        except Exception as e:            # 검증 안 된 묶음이 현장에 가지 않도록 지운다
            traceback.print_exc()
            shutil.rmtree(out, ignore_errors=True)
            self.done.emit(f"검증 실패로 묶음을 지웠습니다: {e}")
