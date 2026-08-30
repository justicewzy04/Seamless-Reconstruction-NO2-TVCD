"""File-discovery helpers shared by experiment scripts."""

from __future__ import annotations

from pathlib import Path


def _find_files(folder_path, suffix):
    return [str(path) for path in Path(folder_path).rglob(f"*{suffix}") if path.is_file()]


def find_nc_files(folder_path):
    return _find_files(folder_path, ".nc")


def find_npy_files(folder_path):
    return _find_files(folder_path, ".npy")


def find_files_with_keyword(file_list, keyword):
    return [file_name for file_name in file_list if keyword in file_name]
