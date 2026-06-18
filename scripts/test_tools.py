"""
工具服务测试脚本
================
测试抠图和检测裁切服务，并把结果保存为图片。

用法:
  python scripts/test_tools.py data/open_datasets/images/fashion_0000.jpg
  python scripts/test_tools.py https://example.com/image.jpg

输出目录:
  outputs/test_tools/<图片名>/
    - remove_bg.png       抠图结果（透明背景）
    - detect_crop_0.jpg   第 1 个检测框裁切
    - detect_crop_1.jpg   第 2 个检测框裁切
    ...
"""

import base64
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import REMOVE_BG_HOST, REMOVE_BG_PORT, DETECT_CROP_HOST, DETECT_CROP_PORT

REMOVE_BG_URL = f"http://{REMOVE_BG_HOST}:{REMOVE_BG_PORT}"
DETECT_CROP_URL = f"http://{DETECT_CROP_HOST}:{DETECT_CROP_PORT}"


def load_image(input_path_or_url: str) -> Image.Image:
    """加载本地图片或 URL 图片。"""
    if input_path_or_url.startswith(("http://", "https://")):
        resp = requests.get(input_path_or_url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))
    return Image.open(input_path_or_url)


def save_base64_image(b64_string: str, output_path: Path, fmt: str = "PNG"):
    """保存 base64 图片到文件。"""
    img = Image.open(io.BytesIO(base64.b64decode(b64_string)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format=fmt)
    print(f"  已保存: {output_path}")


def test_remove_bg(image_path: str, output_dir: Path) -> dict:
    """测试抠图服务。"""
    print("\n[1/2] 测试抠图服务...")
    url = f"{REMOVE_BG_URL}/remove_bg"

    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        start = time.time()
        resp = requests.post(url, files=files, timeout=120)
        resp.raise_for_status()
        elapsed = int((time.time() - start) * 1000)

    data = resp.json()
    save_base64_image(
        data["image"],
        output_dir / "remove_bg.png",
        fmt="PNG",
    )
    print(f"  原始尺寸: {data.get('original_size')}")
    print(f"  处理尺寸: {data.get('processed_size')}")
    print(f"  耗时: {data.get('process_time_ms', elapsed)} ms")
    return data


def test_detect_crop(image_path: str, output_dir: Path) -> dict:
    """测试检测裁切服务。"""
    print("\n[2/2] 测试检测裁切服务...")
    url = f"{DETECT_CROP_URL}/detect_crop"

    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        start = time.time()
        resp = requests.post(url, files=files, timeout=60)
        resp.raise_for_status()
        elapsed = int((time.time() - start) * 1000)

    data = resp.json()
    crops = data.get("crops_base64", [])
    detections = data.get("detections", [])

    print(f"  检测到 {len(detections)} 个目标，返回 {len(crops)} 个裁切")
    for i, crop_b64 in enumerate(crops):
        save_base64_image(crop_b64, output_dir / f"detect_crop_{i}.jpg", fmt="JPEG")

    print(f"  耗时: {elapsed} ms")
    return data


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/test_tools.py <图片路径或URL>")
        sys.exit(1)

    image_path = sys.argv[1]
    image_name = Path(image_path).stem
    output_dir = PROJECT_ROOT / "outputs" / "test_tools" / image_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输入图片: {image_path}")
    print(f"结果保存目录: {output_dir}")

    test_remove_bg(image_path, output_dir)
    test_detect_crop(image_path, output_dir)

    print(f"\n测试完成，所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
