import os

import pytest
from PySide6.QtCore import Qt

from incubator.dataset import Label

PATTERN = "*/cam_03_IMG_Texture_8Bit.png"




# ---- Task 7: 메인 창 ----
from PySide6.QtWidgets import QMessageBox

from incubator.dataset import save_dataset
from incubator.ui.app import MainWindow
from incubator.ui.dataset_dialogs import ImportDialog, default_label_for


@pytest.fixture
def main_window(qtbot, monkeypatch):
    # 정리 단계에서 창을 닫을 때 "저장할까요?" 모달이 테스트를 멈추지 않도록
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_main_window_opens(main_window):
    assert main_window.windowTitle().startswith("Incubator")


def test_open_dataset_warns_missing(main_window, small_ds, tmp_path, monkeypatch):
    p = tmp_path / "t.json"
    save_dataset(small_ds, p)
    os.remove(small_ds.items[0].path)
    seen = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: seen.append(a[2]))
    main_window.open_dataset(str(p))
    assert main_window.dataset.name == "t"
    assert seen and "찾을 수 없습니다" in seen[0]


def test_dirty_marker(qtbot, main_window, small_ds):
    main_window.set_dataset(small_ds)
    assert "*" not in main_window.windowTitle()
    main_window.wizard.data.grid.view.setCurrentRow(0)
    main_window.wizard.data.grid.view.item(0).setSelected(True)
    qtbot.keyClick(main_window.wizard.data.grid.view, Qt.Key_N)
    assert "*" in main_window.windowTitle()


def test_save_clears_dirty(qtbot, main_window, small_ds, tmp_path, monkeypatch):
    monkeypatch.setattr("incubator.ui.app.DATASET_DIR", tmp_path)
    main_window.set_dataset(small_ds)
    main_window.wizard.data.grid.view.setCurrentRow(0)
    main_window.wizard.data.grid.view.item(0).setSelected(True)
    qtbot.keyClick(main_window.wizard.data.grid.view, Qt.Key_N)
    main_window.save()
    assert (tmp_path / "t.json").exists() and "*" not in main_window.windowTitle()


def test_import_default_labels():
    assert default_label_for("normal") is Label.NORMAL
    assert default_label_for("inside") is Label.INTERFERENCE
    assert default_label_for("edge") is Label.INTERFERENCE
    assert default_label_for("minor") is Label.UNLABELED


def test_import_dialog_builds_dataset(qtbot, tmp_path):
    import numpy as np
    from tests.conftest import write_png
    img = np.zeros((48, 64, 3), np.uint8)
    write_png(tmp_path / "data" / "normal" / "a__LH.png", img)
    write_png(tmp_path / "data" / "inside" / "b__LH.png", img)
    d = ImportDialog()
    qtbot.addWidget(d)
    d.name.setText("fem")
    d.set_root(str(tmp_path / "data"))
    d.name_glob.setText("*__LH.png")
    d.mask.setText(str(tmp_path / "m.png"))
    ds = d.build()
    assert ds.name == "fem" and sorted(i.label for i in ds.items) == sorted([Label.NORMAL, Label.INTERFERENCE])


def test_scan_dialog_lists_kinds(qtbot, tmp_images):
    from incubator.ui.dataset_dialogs import ScanDialog
    d = ScanDialog()
    qtbot.addWidget(d)
    d.set_root(tmp_images.root)
    kinds = [d.kind.itemData(i) for i in range(d.kind.count())]
    assert kinds == ["cam_01_IMG_Texture_8Bit.png", "cam_03_IMG_Texture_8Bit.png"]
    d.kind.setCurrentIndex(1)
    d.name.setText("t")
    d.mask.setText(tmp_images.mask)
    assert len(d.build().items) == 2


def test_scan_dialog_suggests_master_mask(qtbot, tmp_path):
    import numpy as np
    from tests.conftest import write_png
    from incubator.ui.dataset_dialogs import ScanDialog
    img = np.zeros((48, 64, 3), np.uint8)
    write_png(tmp_path / "Source" / "car1" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    m = write_png(tmp_path / "Master" / "JG" / "fastenLHCam_02_IMG_Kungfu.png", img)
    write_png(tmp_path / "Master" / "JK" / "fastenLHCam_02_IMG_Kungfu.png", img)
    d = ScanDialog()
    qtbot.addWidget(d)
    d.set_root(str(tmp_path))
    assert d.mask.text() == m


def _fem_like_root(tmp_path):
    import numpy as np
    from tests.conftest import make_mask, write_png
    root = tmp_path / "clip"
    img = np.zeros((48, 64, 3), np.uint8)
    for k in range(3):
        write_png(root / "Source" / f"car{k}" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    write_png(root / "Master" / "JG" / "fastenLHCam_02_IMG_Kungfu.png", make_mask(64, 48))
    return root


def test_scan_dialog_only_root_needed(qtbot, tmp_path, monkeypatch):
    from incubator.ui.dataset_dialogs import ScanDialog
    d = ScanDialog()
    qtbot.addWidget(d)
    assert d.field_labels() == ["루트 폴더", "데이터셋 이름", "카메라 이름", "마스터 파일"]
    d.set_root(str(_fem_like_root(tmp_path)))
    assert d.kind.currentText() == "fastenLHCam_02  (3장)"
    assert d.name.text() == "clip_fastenLHCam_02"
    assert "3장" in d.result_info.text() and d.mask.text().endswith("fastenLHCam_02_IMG_Kungfu.png")
    d._accept()
    assert d.dataset is not None and len(d.dataset.items) == 3


def test_scan_dialog_opens_details_when_no_mask(qtbot, tmp_path):
    import numpy as np
    from tests.conftest import write_png
    from incubator.ui.dataset_dialogs import ScanDialog
    write_png(tmp_path / "r" / "car1" / "fastenLHCam_02_IMG_Texture_8Bit.png", np.zeros((4, 4, 3), np.uint8))
    d = ScanDialog()
    qtbot.addWidget(d)
    d.set_root(str(tmp_path / "r"))
    assert "마스터 파일을 찾지 못했습니다" in d.result_info.text()


def test_scan_name_avoids_overwrite(qtbot, tmp_path, monkeypatch):
    from incubator import dataset as dataset_mod
    from incubator.ui.dataset_dialogs import ScanDialog
    monkeypatch.setattr(dataset_mod, "DATASET_DIR", tmp_path / "saved")
    (tmp_path / "saved").mkdir()
    (tmp_path / "saved" / "clip_fastenLHCam_02.json").write_text("{}")
    d = ScanDialog()
    qtbot.addWidget(d)
    d.set_root(str(_fem_like_root(tmp_path)))
    assert d.name.text() == "clip_fastenLHCam_02_2"


def test_scan_dialog_has_no_exclude_field_but_skips_master(qtbot, tmp_path):
    import numpy as np
    from tests.conftest import write_png
    from incubator.ui.dataset_dialogs import ScanDialog
    root = _fem_like_root(tmp_path)
    write_png(root / "Master" / "JG" / "fastenLHCam_02_IMG_Texture_8Bit.png", np.zeros((48, 64, 3), np.uint8))
    d = ScanDialog()
    qtbot.addWidget(d)
    assert not hasattr(d, "exclude")
    d.set_root(str(root))
    assert len(d.build().items) == 3


def test_data_step_shows_and_changes_master_file(qtbot, tmp_path):
    from tests.conftest import make_shifted
    from incubator.ui.wizard import DataStep
    ds, _ = make_shifted(tmp_path)
    real_mask = ds.mask_path
    ds.mask_path = str(tmp_path / "Master" / "copy_of_mask.png")
    import shutil
    shutil.copy(real_mask, ds.mask_path)
    s = DataStep()
    qtbot.addWidget(s)
    s.set_dataset(ds)
    assert "copy_of_mask.png" in s.master_path.text() and "위치 보정 안 됨" in s.master_label.text()
    with qtbot.waitSignal(s.changed, timeout=1000):
        s.set_mask_path(real_mask)
    assert ds.mask_path == real_mask
    assert s.master_label.text() == ""        # 정상이면 문구 없음
    assert s.grid.aligner is not None


def test_master_row_layout_and_align_toggle(qtbot, tmp_path):
    from tests.conftest import make_shifted
    from incubator.ui.wizard import DataStep
    ds, _ = make_shifted(tmp_path)
    s = DataStep()
    qtbot.addWidget(s)
    s.set_dataset(ds)
    assert s.master_path.isReadOnly() and s.master_path.text() == ds.mask_path
    assert s.align_check.isEnabled() and s.align_check.isChecked()
    with qtbot.waitSignal(s.changed, timeout=1000):
        s.align_check.setChecked(False)
    assert ds.align is False and s.grid.aligner is None


def test_align_toggle_disabled_without_reference(qtbot, small_ds):
    from incubator.ui.wizard import DataStep
    s = DataStep()
    qtbot.addWidget(s)
    s.set_dataset(small_ds)
    assert not s.align_check.isEnabled() and not s.align_check.isChecked()


def test_data_step_has_single_list_view(qtbot, small_ds):
    from PySide6.QtWidgets import QTabWidget
    from incubator.ui.wizard import DataStep
    s = DataStep()
    qtbot.addWidget(s)
    s.set_dataset(small_ds)
    assert not s.findChildren(QTabWidget) and not hasattr(s, "label_tab")


def test_data_step_is_split_into_sections(qtbot, small_ds):
    from incubator.ui.wizard import DataStep, Wizard
    s = DataStep()
    qtbot.addWidget(s)
    s.set_dataset(small_ds)
    assert list(s.sections) == ["데이터셋", "라벨링"]
    assert s.grid.parent() is s.sections["라벨링"] or s.sections["라벨링"].isAncestorOf(s.grid)
    assert s.sections["데이터셋"].isAncestorOf(s.master_path)
    w = Wizard()
    qtbot.addWidget(w)
    assert w.step_bar.objectName() == "stepBar" and all(w.step_bar.isAncestorOf(t) for t in w.tabs)


def test_main_window_has_no_menu_bar_but_keeps_shortcuts(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from incubator.ui.app import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.menuBar().isHidden()
    assert any(a.shortcut().toString() == "Ctrl+S" for a in w.actions())


def test_data_step_hides_boxes_for_patchcore(qtbot, small_ds):
    from incubator.ui.wizard import DataStep
    s = DataStep()
    qtbot.addWidget(s)
    s.set_dataset(small_ds)
    s.set_algo("PatchCore")
    assert not s.grid.boxes_enabled
    s.set_algo("YOLO")
    assert s.grid.boxes_enabled
