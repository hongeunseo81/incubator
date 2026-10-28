import os
from collections import Counter
from pathlib import Path

import numpy as np

from impostor.dataset import (Item, Label, Pin, import_folders, load_dataset, save_dataset,
                              scan_dataset)
from impostor.imageio import imread, load_mask
from tests.conftest import write_png

PATTERN = "*/cam_03_IMG_Texture_8Bit.png"


def test_imread_korean_path(tmp_images):
    assert imread(tmp_images.any_image).shape == (48, 64, 3)


def test_load_mask_is_bool(tmp_images):
    m = load_mask(tmp_images.mask)
    assert m.dtype == bool and m.shape == (48, 64) and m.any()


def test_scan_groups_by_parent(tmp_images):
    ds = scan_dataset("t", tmp_images.root, PATTERN, tmp_images.mask)
    assert len(ds.items) == 2
    assert {i.group for i in ds.items} == {"1_7C  0001", "1_7C  0002"}
    assert all(i.label is Label.UNLABELED for i in ds.items)


def test_import_maps_subfolders_and_filters(tmp_path):
    root = tmp_path / "data"
    img = np.zeros((48, 64, 3), np.uint8)
    write_png(root / "normal" / "a__LH.png", img)
    write_png(root / "normal" / "b__LH.png", img)
    write_png(root / "normal" / "b__RH.png", img)          # 필터로 빠져야 함
    write_png(root / "inside" / "cabletest__retake_NG__unknown.png", img)
    write_png(root / "minor" / "c__LH.png", img)
    write_png(root / "masks" / "x__LH.png", img)           # 매핑에 없는 폴더
    mapping = {"normal": Label.NORMAL, "inside": Label.INTERFERENCE, "minor": Label.UNLABELED}
    ds = import_folders("t", str(root), mapping, "*__LH.png;*retake*", str(root / "m.png"))
    assert Counter(i.label for i in ds.items) == {Label.NORMAL: 2, Label.INTERFERENCE: 1, Label.UNLABELED: 1}
    assert {i.group for i in ds.items if i.label is Label.NORMAL} == {"a__LH", "b__LH"}


def test_add_box_sets_interference():
    it = Item("a.png", "g")
    it.add_box((1, 2, 10, 20))
    assert it.label is Label.INTERFERENCE and it.boxes == [(1, 2, 10, 20)]


def test_json_roundtrip(tmp_path, tmp_images):
    ds = scan_dataset("t", tmp_images.root, PATTERN, tmp_images.mask)
    ds.items[0].add_box((1, 2, 3, 4))
    ds.items[1].label = Label.NORMAL
    ds.items[1].pin = Pin.EVAL
    p = tmp_path / "저장" / "t.json"
    save_dataset(ds, p)
    back = load_dataset(p)
    assert back == ds


def test_missing_files_excluded(tmp_images):
    ds = scan_dataset("t", tmp_images.root, PATTERN, tmp_images.mask)
    for it in ds.items:
        it.label = Label.NORMAL
    os.remove(ds.items[0].path)
    assert ds.missing() == [ds.items[0]]
    assert ds.usable(Label.NORMAL) == [ds.items[1]]


def test_size_mismatch_reported(tmp_images):
    odd = write_png(Path(tmp_images.root) / "1_7C  0003" / "cam_03_IMG_Texture_8Bit.png",
                    np.zeros((100, 100, 3), np.uint8))
    ds = scan_dataset("t", tmp_images.root, PATTERN, tmp_images.mask)
    assert ds.check_sizes() == [odd]


# ---- 원본 폴더에서 바로 ----
from impostor.dataset import find_image_kinds


def test_find_image_kinds_counts_recursive(tmp_path):
    img = np.zeros((48, 64, 3), np.uint8)
    write_png(tmp_path / "Source" / "car1" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    write_png(tmp_path / "Source" / "car1" / "fastenRHCam_02_IMG_Texture_8Bit.png", img)
    write_png(tmp_path / "261020 FEM" / "NG" / "car2" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    write_png(tmp_path / "Source" / "car1" / "fastenLHCam_02_IMG_PointCloud_Z.png", img)   # Texture 아님
    assert find_image_kinds(str(tmp_path)) == [("fastenLHCam_02_IMG_Texture_8Bit.png", 2),
                                                ("fastenRHCam_02_IMG_Texture_8Bit.png", 1)]


def test_scan_recursive_any_depth(tmp_path):
    img = np.zeros((48, 64, 3), np.uint8)
    write_png(tmp_path / "Source" / "car1" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    write_png(tmp_path / "261020 FEM" / "NG" / "car2" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    ds = scan_dataset("t", str(tmp_path), "**/fastenLHCam_02_IMG_Texture_8Bit.png", "m.png")
    assert sorted(i.group for i in ds.items) == ["car1", "car2"]


def test_unlabeled_to_normal(tmp_images):
    ds = scan_dataset("t", tmp_images.root, PATTERN, tmp_images.mask)
    ds.items[0].label = Label.INTERFERENCE
    assert ds.unlabeled_to_normal() == 1
    assert [i.label for i in ds.items] == [Label.INTERFERENCE, Label.NORMAL]


def test_scan_excludes_master_folders(tmp_path):
    img = np.zeros((48, 64, 3), np.uint8)
    write_png(tmp_path / "Source" / "car1" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    write_png(tmp_path / "Master" / "JG" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    write_png(tmp_path / "3dvs_Master" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    ds = scan_dataset("t", str(tmp_path), "**/fastenLHCam_02_IMG_Texture_8Bit.png", "m.png", exclude=["master"])
    assert [i.group for i in ds.items] == ["car1"]
    assert find_image_kinds(str(tmp_path), exclude=["master"]) == [("fastenLHCam_02_IMG_Texture_8Bit.png", 1)]


def test_kinds_sorted_by_count(tmp_path):
    img = np.zeros((4, 4, 3), np.uint8)
    write_png(tmp_path / "calib" / "0000_IMG_Texture_8Bit.png", img)
    for k in range(3):
        write_png(tmp_path / f"car{k}" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    assert [n for n, _ in find_image_kinds(str(tmp_path))] == ["fastenLHCam_02_IMG_Texture_8Bit.png",
                                                               "0000_IMG_Texture_8Bit.png"]


def test_mask_prefers_shallow_master(tmp_path):
    from impostor.dataset import find_mask_for
    img = np.zeros((4, 4, 3), np.uint8)
    write_png(tmp_path / "3dvs_Master" / "a" / "b" / "fastenLHCam_02_IMG_Kungfu.png", img)
    m = write_png(tmp_path / "Master" / "JG" / "fastenLHCam_02_IMG_Kungfu.png", img)
    write_png(tmp_path / "car1" / "fastenLHCam_02_IMG_Kungfu.png", img)
    assert find_mask_for(str(tmp_path), "fastenLHCam_02_IMG_Texture_8Bit.png") == m


def test_imsize_reads_png_header_only(tmp_path, monkeypatch):
    import impostor.imageio as io
    p = write_png(tmp_path / "한글 폴더" / "a.png", np.zeros((48, 64, 3), np.uint8))
    monkeypatch.setattr(io, "imread", lambda path: (_ for _ in ()).throw(AssertionError("decoded")))
    assert io.imsize(p) == (48, 64)


def test_align_flag_roundtrip_and_default(tmp_path, tmp_images):
    import json
    ds = scan_dataset("t", tmp_images.root, PATTERN, tmp_images.mask)
    assert ds.align is True
    ds.align = False
    p = tmp_path / "t.json"
    save_dataset(ds, p)
    assert load_dataset(p).align is False
    data = json.loads(p.read_text(encoding="utf-8"))
    del data["align"]                                   # 예전 파일에는 키가 없다
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert load_dataset(p).align is True
