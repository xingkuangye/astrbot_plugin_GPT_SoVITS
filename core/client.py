import asyncio
import os
from dataclasses import dataclass

from aiohttp import ClientSession, ClientTimeout, FormData

from astrbot.api import logger

from .config import PluginConfig


@dataclass
class TTSRequestResult:
    ok: bool
    data: bytes | None = None
    error: str = ""
    text: str = ""
    file_path: str = ""

    @property
    def size(self) -> int:
        """音频数据大小（字节）"""
        return len(self.data) if self.data else 0

    @property
    def is_empty(self) -> bool:
        """是否无数据"""
        return self.size == 0

    def __bool__(self) -> bool:
        return self.ok and not self.is_empty


class IndexTTSApiClient:
    """
    Index-TTS 2.5 API 客户端。

    异步任务流程：
        1. POST /tts          提交合成任务（multipart 表单），返回 jobId
        2. GET  /progress/{jobId}   轮询真实进度
        3. GET  /result/{jobId}     完成后下载 wav 音频
    """

    def __init__(self, config: PluginConfig):
        self.cfg = config.client
        self.base_url = self.cfg.base_url.rstrip("/")
        self.session = ClientSession(timeout=ClientTimeout(total=self.cfg.timeout))
        self.poll_interval = 1.5  # 轮询间隔（秒）
        self.max_wait = 600  # 单次合成最长等待时间（秒）

    async def close(self):
        if self.session:
            await self.session.close()

    async def tts(self, params: dict) -> TTSRequestResult:
        """提交合成任务并等待完成，返回 wav 音频数据"""
        text = str(params.get("text", ""))
        ref_audio = str(params.get("ref_audio_path", ""))
        request_text = text

        if not ref_audio:
            return TTSRequestResult(False, error="缺少 ref_audio_path", text=request_text)
        if not os.path.exists(ref_audio):
            return TTSRequestResult(
                False, error=f"参考音频不存在: {ref_audio}", text=request_text
            )

        # 传给 Index-TTS 的表单字段（剔除插件内部字段）
        ignored = {"text", "ref_audio_path", "media_type"}
        fields = {
            k: v
            for k, v in params.items()
            if k not in ignored and v not in (None, "")
        }

        try:
            job_id = await self._submit(text, ref_audio, fields)
            logger.info(f"Index-TTS 任务已提交: {job_id}")
        except Exception as e:
            logger.exception("提交 Index-TTS 任务失败")
            return TTSRequestResult(False, error=f"提交失败: {e}", text=request_text)

        # 轮询进度
        start = asyncio.get_running_loop().time()
        while True:
            try:
                prog = await self._get_progress(job_id)
            except Exception as e:
                return TTSRequestResult(
                    False, error=f"查询进度失败: {e}", text=request_text
                )

            if prog.get("error"):
                return TTSRequestResult(
                    False, error=str(prog.get("error")), text=request_text
                )
            if prog.get("done"):
                break
            if asyncio.get_running_loop().time() - start > self.max_wait:
                return TTSRequestResult(False, error="合成超时", text=request_text)
            await asyncio.sleep(self.poll_interval)

        # 下载结果
        try:
            data = await self._download(job_id)
        except Exception as e:
            return TTSRequestResult(
                False, error=f"下载结果失败: {e}", text=request_text
            )

        if not data:
            return TTSRequestResult(False, error="下载结果为空", text=request_text)

        return TTSRequestResult(True, data=data, text=request_text)

    async def _submit(self, text: str, ref_audio_path: str, fields: dict) -> str:
        with open(ref_audio_path, "rb") as f:
            file_data = f.read()

        form = FormData()
        form.add_field(
            "file",
            file_data,
            filename=os.path.basename(ref_audio_path),
            content_type="audio/wav",
        )
        form.add_field("text", text)
        for k, v in fields.items():
            form.add_field(k, v if isinstance(v, str) else str(v))

        async with self.session.post(f"{self.base_url}/tts", data=form) as resp:
            if resp.status != 200:
                detail = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {detail[:300]}")
            js = await resp.json(content_type=None)
            job_id = js.get("jobId") if isinstance(js, dict) else None
            if not job_id:
                raise RuntimeError(f"响应缺少 jobId: {js}")
            return job_id

    async def _get_progress(self, job_id: str) -> dict:
        async with self.session.get(f"{self.base_url}/progress/{job_id}") as resp:
            if resp.status != 200:
                detail = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {detail[:300]}")
            js = await resp.json(content_type=None)
            return js if isinstance(js, dict) else {}

    async def _download(self, job_id: str) -> bytes:
        async with self.session.get(f"{self.base_url}/result/{job_id}") as resp:
            if resp.status == 200:
                return await resp.read()
            if resp.status == 202:
                raise RuntimeError("结果尚未就绪")
            detail = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {detail[:300]}")
