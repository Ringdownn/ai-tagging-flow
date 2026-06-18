"""
商品打标编排 HTTP 服务
======================
对外暴露 POST /tag、POST /tag/upload 接口，并托管静态前端页面。

启动:
  uvicorn services.orchestration_service:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import ORCHESTRATION_PORT  # noqa: E402
from services.orchestration import tag_image, tag_image_bytes  # noqa: E402

app = FastAPI(title="商品打标编排服务", version="1.0.0")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TagRequest(BaseModel):
    image_url: str = Field(..., description="商品图片 URL")


@app.get("/")
async def index():
    """返回前端页面。"""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="前端页面未找到")
    return FileResponse(index_path)


@app.post("/tag")
async def tag(request: TagRequest):
    """为商品图片 URL 生成标准化标签。"""
    try:
        return await tag_image(request.image_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"打标失败: {e}") from e


@app.post("/tag/upload")
async def tag_upload(file: UploadFile = File(...)):
    """上传本地图片并生成标准化标签。"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    try:
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="图片文件为空")
        return await tag_image_bytes(image_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"打标失败: {e}") from e


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "orchestration",
        "port": ORCHESTRATION_PORT,
    }


if __name__ == "__main__":
    import uvicorn

    from config.settings import ORCHESTRATION_HOST

    uvicorn.run(
        "services.orchestration_service:app",
        host=ORCHESTRATION_HOST,
        port=ORCHESTRATION_PORT,
    )
