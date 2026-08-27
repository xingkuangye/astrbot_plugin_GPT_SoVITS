from typing import Any

from astrbot.api import logger

from .client import IndexTTSApiClient, TTSRequestResult
from .config import PluginConfig
from .local_data import LocalDataManager


class TTSService:
    def __init__(
        self,
        config: PluginConfig,
        client: IndexTTSApiClient,
        local_data: LocalDataManager,
    ):
        self.default_params = config.default_params
        self.client = client
        self.local_data = local_data

    async def inference(
        self,
        text: str,
        extra_params: dict[str, Any] | None = None,
    ) -> TTSRequestResult:
        """TTS 推理（Index-TTS）"""
        params = self.default_params.copy()
        if text:
            params["text"] = text

        if extra_params:
            filtered_params = {
                k: v for k, v in extra_params.items() if k in params
            }
            params.update(filtered_params)
            logger.debug(f"已更新已有参数: {filtered_params}")

        cached_audio = self.local_data.get_cached_audio(params)
        if cached_audio:
            cache_path, cached_data = cached_audio
            logger.debug("命中缓存，跳过 TTS 请求")
            return TTSRequestResult(
                ok=True,
                data=cached_data,
                text=str(params.get("text", "")),
                file_path=str(cache_path),
            )

        logger.debug(f"向 Index-TTS 发起请求，参数: {params}")
        result = await self.client.tts(params)

        if bool(result):
            cache_path = self.local_data.save_audio(result.data, params)
            if cache_path:
                result.file_path = str(cache_path)
        else:
            logger.error(f"TTS 推理失败: {result.error}")

        return result
