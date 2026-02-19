# -*- mode: python ; coding: utf-8 -*-
# このファイルはPyInstallerによって自動生成されたもので、それをカスタマイズして使用しています。
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2, copytree

from PyInstaller.utils.hooks import collect_data_files

parser = ArgumentParser()
parser.add_argument("--libcore_path", type=Path)
parser.add_argument("--libonnxruntime_path", type=Path)
parser.add_argument("--core_model_dir_path", type=Path)
options = parser.parse_args()

libcore_path: Path | None = options.libcore_path
if libcore_path is not None and not libcore_path.is_file():
    raise Exception(f"libcore_path: {libcore_path} is not file")

libonnxruntime_path: Path | None = options.libonnxruntime_path
if libonnxruntime_path is not None and not libonnxruntime_path.is_file():
    raise Exception(f"libonnxruntime_path: {libonnxruntime_path} is not file")

core_model_dir_path: Path | None = options.core_model_dir_path
if core_model_dir_path is not None and not core_model_dir_path.is_dir():
    raise Exception(f"core_model_dir_path: {core_model_dir_path} is not dir")


from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os

# 物理パスの取得
def get_path(pkg_name):
    try:
        return get_package_paths(pkg_name)[0]
    except:
        return None

import modelscope
import funasr
import librosa

import inflect
import typeguard
#import fastapi

import os
import site
site_packages_path = Path(site.getsitepackages()[0])
nvidia_path = site_packages_path / "nvidia"

nvidia_binaries = []

data = []

if nvidia_path.exists():
    # nvidia フォルダ以下のすべてのファイルを再帰的に取得
    for file_path in nvidia_path.rglob("*"):
        if file_path.is_file():
            # site-packages から見た相対パスを取得 (例: nvidia/cuda_runtime/lib/libcudart.so.12)
            relative_path = file_path.relative_to(site_packages_path)
            # (コピー元フルパス, コピー先ディレクトリ)
            # コピー先を relative_path.parent にすることで、構造が維持される
            nvidia_binaries.append((str(file_path), str(relative_path.parent)))
else:
    import torch
    torch_dir = os.path.dirname(torch.__file__)
    data.append((torch_dir, "torch"))

#fastapi_dir = os.path.dirname(fastapi.__file__)
modelscope_dir = os.path.dirname(modelscope.__file__)
funasr_dir = os.path.dirname(funasr.__file__)
librosa_dir = os.path.dirname(librosa.__file__)

inflect_dir = os.path.dirname(inflect.__file__)
typeguard_dir = os.path.dirname(typeguard.__file__)


data += collect_data_files("pyopenjtalk")
data += collect_data_files("contractions")
# wetext も怪しいなら一緒に入れておく
data += collect_data_files("wetext")

data += [('voxcpm', 'voxcpm')]

#data.append((fastapi_dir, "fastapi"))
data.append((modelscope_dir, "modelscope"))
data.append((funasr_dir, "funasr"))
data.append((typeguard_dir, "typeguard"))
data.append((inflect_dir, "inflect"))

# modelscope の全ソースとデータを同梱（これでASTスキャンを正常化させる）

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=nvidia_binaries,
    datas=data,
    hiddenimports=[
        'voxcpm',
        'modelscope',
        'librosa',
        'nvidia',
        'inflect',
        'typeguard',
        #'fastapi',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="run",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory="engine_internal",  # 実行時に"sys._MEIPASS"が参照するディレクトリ名
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="run",
)

# 実行ファイルのディレクトリに配置するファイルのコピー

# 実行ファイルと同じrootディレクトリ
target_dir = Path(DISTPATH) / "run"

# リソースをコピー
manifest_file_path = Path("engine_manifest.json")
copy2(manifest_file_path, target_dir)
copytree("resources", target_dir / "resources")

license_file_path = Path("licenses.json")
if license_file_path.is_file():
    copy2("licenses.json", target_dir)

# 動的ライブラリをコピー
if libonnxruntime_path is not None:
    copy2(libonnxruntime_path, target_dir)
if core_model_dir_path is not None:
    copytree(core_model_dir_path, target_dir / "model")
if libcore_path is not None:
    copy2(libcore_path, target_dir)
