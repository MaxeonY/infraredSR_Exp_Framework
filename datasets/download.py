import argparse
import shutil
import subprocess
import sys
import zipfile
import time
import gdown
import tarfile
from pathlib import Path


# -----------------------------
# KAIST Google Drive file IDs
# 来自你截图中的命令
# -----------------------------
KAIST_FILE_IDS = {
    "preview": "11nhHpmuh2FUjrLNfGs51R2Mqqy1GTjY8",
    "full": "1sBcAmFqNJmNMBZdMtKmO2X4BRjKPyKMc",
}


def run_command(command: list[str]) -> None:
    """Run shell command and raise error if failed."""
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def ensure_gdown_installed() -> None:
    """Check whether gdown is installed."""
    if shutil.which("gdown") is None:
        raise EnvironmentError(
            "gdown 未安装或不在当前环境 PATH 中。\n"
            "请先在当前 conda 环境中执行：\n"
            "pip install gdown"
        )


def download_with_gdown(file_id: str, output_path: Path, max_retries: int = 3) -> None:
    """
    Download file from Google Drive using gdown API with retries.
    """
    if output_path.exists():
        print(f"[INFO] 文件已存在，跳过下载：{output_path}")
        return

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[INFO] 开始下载到：{output_path}")
    
    for attempt in range(1, max_retries + 1):
        try:
            # gdown.download automatically handles progress bar and basic errors
            # We use the API directly for better integration
            gdown.download(url, str(output_path), quiet=False, fuzzy=True)
            print("[INFO] 下载完成。")
            return
        except Exception as e:
            print(f"[WARNING] 第 {attempt} 次下载尝试失败: {e}")
            if attempt < max_retries:
                wait_time = attempt * 5
                print(f"[INFO] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] 达到最大重试次数 ({max_retries})。")
                print("\n" + "="*50)
                print("如果由于网络原因持续失败，建议手动下载：")
                print(f"链接: {url}")
                print(f"下载后请将文件移动至: {output_path}")
                print("="*50 + "\n")
                raise RuntimeError(f"无法从 Google Drive 下载文件 ID: {file_id}")


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """
    Extract zip or tar archive.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"压缩文件不存在：{archive_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 开始解压：{archive_path}")
    
    # Try ZIP first
    if zipfile.is_zipfile(archive_path):
        print("[INFO] 检测到 ZIP 格式。")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    # Try TAR
    elif tarfile.is_tarfile(archive_path):
        print("[INFO] 检测到 TAR 格式。")
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
    else:
        # Fallback to shutil.unpack_archive which is more generic
        try:
            print("[INFO] 尝试使用通用解压方法...")
            shutil.unpack_archive(str(archive_path), str(extract_dir))
        except Exception as e:
            raise ValueError(
                f"无法识别或解压文件：{archive_path.name}\n"
                f"错误详情：{e}\n"
                "请手动确认文件格式是否完整。"
            )
    print(f"[INFO] 解压完成，目标目录：{extract_dir}")


def guess_archive_name(split: str) -> str:
    """
    Guess local archive filename.
    """
    return f"kaist_{split}.zip"


def validate_split(split: str) -> None:
    if split not in KAIST_FILE_IDS:
        raise ValueError(f"split 必须是 {list(KAIST_FILE_IDS.keys())} 之一，当前为：{split}")


def main():
    parser = argparse.ArgumentParser(description="Download KAIST dataset preview/full set.")
    parser.add_argument(
        "--split",
        type=str,
        default="preview",
        choices=["preview", "full"],
        help="下载 preview 或 full，默认 preview"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="KAIST 数据保存根目录。默认保存到项目根目录下的 data/raw/KAIST/"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="下载完成后自动解压"
    )
    parser.add_argument(
        "--remove-archive",
        action="store_true",
        help="解压成功后删除压缩包"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若压缩文件已存在，则强制重新下载"
    )

    args = parser.parse_args()
    validate_split(args.split)

    # 当前脚本路径：infraredSR/datasets/download.py
    # 项目根目录：infraredSR/
    project_root = Path(__file__).resolve().parent.parent

    if args.root is None:
        data_root = project_root / "data" / "raw" / "KAIST"
    else:
        data_root = Path(args.root).resolve()

    data_root.mkdir(parents=True, exist_ok=True)

    archive_name = guess_archive_name(args.split)
    archive_path = data_root / archive_name
    file_id = KAIST_FILE_IDS[args.split]

    # force 模式：删除旧压缩文件
    if args.force and archive_path.exists():
        print(f"[INFO] --force 已启用，删除已有压缩文件：{archive_path}")
        archive_path.unlink()

    # 下载
    download_with_gdown(file_id=file_id, output_path=archive_path)

    # 解压目录
    # 建议结构：
    # data/raw/KAIST/preview/
    # data/raw/KAIST/full/
    if args.extract:
        extract_dir = data_root / args.split
        extract_archive(archive_path=archive_path, extract_dir=extract_dir)

        if args.remove_archive:
            print(f"[INFO] 删除压缩包：{archive_path}")
            archive_path.unlink()

    print("[INFO] 所有操作完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)