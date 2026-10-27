"""JMHL Image API —— ComfyUI 自定义节点。

OpenAI 兼容的图生图 / 文生图节点：

- 自定义 API 地址与密钥
- `宽度` / `高度` 指定输出像素尺寸，本地 Lanczos 缩放保证输出尺寸
- 宽度/高度为 0 时按 `宽高比` 选择目标尺寸并本地缩放
- 可自动在提示词末尾追加尺寸说明
"""

import base64
import io
import json
import os
import re

import numpy as np
import requests
import torch
from PIL import Image

TIMEOUT = 900

ASPECT_TO_REQUEST_SIZE = {
    "1:1": "1024x1024",
    "16:9": "1280x720",
    "9:16": "720x1280",
    "4:3": "1024x768",
    "3:4": "768x1024",
    "3:2": "1152x768",
    "2:3": "768x1152",
    "auto": "auto",
}


def _normalize_endpoint(base_url, operation):
    value = (base_url or "").strip().rstrip("/")
    if not value:
        raise ValueError("base_url 为空")
    if value.endswith("/images/edits") or value.endswith("/images/generations"):
        return value
    if value.endswith("/v1"):
        return f"{value}/images/{operation}"
    return f"{value}/v1/images/{operation}"


def _parse_json_object(value, label):
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"{label} 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} 必须是 JSON 对象")
    return data


def _parse_aspect_ratio(value):
    text = str(value or "").strip()
    if not text or text == "auto" or ":" not in text:
        return None
    try:
        left, right = text.split(":", 1)
        ratio = float(left) / float(right)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return ratio if ratio > 0 else None


def _tensor_to_png(image_tensor):
    arr = (image_tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _pil_to_tensor(image):
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(arr).unsqueeze(0)


def _decode_item(item):
    b64 = item.get("b64_json")
    if isinstance(b64, str):
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64)))

    url = item.get("url") or item.get("image_url")
    if isinstance(url, str) and url.startswith("http"):
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

    raise ValueError(f"响应条目里没有图像字段：{list(item)}")


class PixelAPIImage2Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模式": (["图生图", "文生图"], {"default": "图生图"}),
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "模型名称": ("STRING", {
                    "default": "gpt-image-2.5-Sunburst",
                    "multiline": False,
                }),
                "API地址": ("STRING", {
                    "default": "https://api.openai.com/v1",
                    "multiline": False,
                }),
                "API密钥": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "宽高比": ([
                    "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "auto",
                ], {"default": "1:1"}),
                "宽度": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "可选。0 表示不指定；和高度同时大于 0 时按该像素尺寸生成。",
                }),
                "高度": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "可选。0 表示不指定；和宽度同时大于 0 时按该像素尺寸生成。",
                }),
                "画质": (["auto", "low", "medium", "high", "xhigh"], {"default": "auto"}),
                "背景": (["opaque", "transparent", "auto"], {"default": "opaque"}),
                "输出格式": (["png", "jpeg", "webp"], {"default": "png"}),
                "随机种": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                }),
                "超时时间_秒": ("INT", {"default": 300, "min": 30, "max": 3600}),
                "额外参数JSON": ("STRING", {"multiline": True, "default": "{}"}),
                "额外请求头JSON": ("STRING", {"multiline": True, "default": "{}"}),
            },
            "optional": {
                "图片1": ("IMAGE",),
                "图片2": ("IMAGE",),
                "图片3": ("IMAGE",),
                "图片4": ("IMAGE",),
                "图片5": ("IMAGE",),
                "图片6": ("IMAGE",),
                "自动补充尺寸提示词": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "开启后，按宽度/高度或宽高比确定的尺寸，自动在提示词末尾追加尺寸说明。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("图像", "图片链接", "响应信息", "原始响应", "HTTP状态码")
    FUNCTION = "run"
    CATEGORY = "JMHL API"
    DESCRIPTION = "图生图/文生图；宽度和高度都为 0 时按宽高比选择目标尺寸并本地缩放，都大于 0 时按指定像素缩放。"

    def run(self, **kwargs):
        mode = str(kwargs.get("模式", "图生图") or "图生图")
        prompt = str(kwargs.get("提示词", "") or "").strip()
        model = str(kwargs.get("模型名称", "") or "").strip()
        base_url = str(kwargs.get("API地址", "") or "").strip()
        api_key = str(kwargs.get("API密钥", "") or "").strip()
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
        aspect_ratio = str(kwargs.get("宽高比", "1:1") or "1:1")
        quality = str(kwargs.get("画质", "auto") or "auto")
        background = str(kwargs.get("背景", "opaque") or "opaque")
        output_format = str(kwargs.get("输出格式", "png") or "png")
        seed = int(kwargs.get("随机种", 0) or 0)
        timeout = int(kwargs.get("超时时间_秒", 300) or 300)
        width = int(kwargs.get("宽度", 0) or 0)
        height = int(kwargs.get("高度", 0) or 0)
        extra_params = _parse_json_object(kwargs.get("额外参数JSON", "{}"), "额外参数JSON")
        extra_headers = _parse_json_object(kwargs.get("额外请求头JSON", "{}"), "额外请求头JSON")

        if (width > 0) != (height > 0):
            raise RuntimeError("宽度和高度需要同时填写，或都填 0 使用宽高比")
        custom_size = f"{width}x{height}" if width > 0 and height > 0 else ""
        request_size = custom_size or ASPECT_TO_REQUEST_SIZE.get(aspect_ratio, "auto")
        ratio_value = None if custom_size else _parse_aspect_ratio(aspect_ratio)

        if not prompt:
            raise RuntimeError("提示词为空")
        if not model:
            raise RuntimeError("模型名称为空")
        if not api_key:
            raise RuntimeError("API密钥为空，也没有设置 OPENAI_API_KEY 环境变量")

        raw_hint = kwargs.get("自动补充尺寸提示词", True)
        if isinstance(raw_hint, str):
            size_hint_enabled = raw_hint.strip().lower() not in ("false", "0", "", "none")
        else:
            size_hint_enabled = bool(raw_hint)
        size_hint_appended = False
        if size_hint_enabled and request_size and request_size != "auto":
            try:
                hint_w, hint_h = (int(part) for part in request_size.lower().split("x", 1))
                size_hint = f"生成的图片尺寸为{hint_w}px {hint_h}px"
            except (TypeError, ValueError):
                size_hint = ""
            if size_hint and size_hint not in re.sub(r"\s+", " ", prompt):
                prompt = f"{prompt}\n{size_hint}"
                size_hint_appended = True

        images = [kwargs.get(f"图片{i}") for i in range(1, 7)]
        images = [image for image in images if image is not None]

        headers = {"Authorization": f"Bearer {api_key}"}
        headers.update({str(key): str(value) for key, value in extra_headers.items()})

        common = {
            "model": model,
            "prompt": prompt,
            "quality": quality,
            "background": background,
            "output_format": output_format,
            "seed": seed,
            "size": request_size,
        }

        if mode == "图生图" and images:
            endpoint = _normalize_endpoint(base_url, "edits")
            files = [
                ("image[]", (f"image_{index}.png", _tensor_to_png(image), "image/png"))
                for index, image in enumerate(images, start=1)
            ]
            data = {
                key: (value if isinstance(value, str) else str(value))
                for key, value in common.items()
            }
            data.update({
                str(key): (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
                for key, value in extra_params.items()
            })
            response = requests.post(
                endpoint, headers=headers, data=data, files=files, timeout=timeout
            )
        else:
            endpoint = _normalize_endpoint(base_url, "generations")
            payload = dict(common)
            payload.update(extra_params)
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=timeout
            )

        status_code = response.status_code
        if status_code != 200:
            raise RuntimeError(f"HTTP {status_code}：{response.text[:800]}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"返回内容不是 JSON：{response.text[:500]}") from exc

        items = payload.get("data") or []
        if not items:
            raise RuntimeError(f"响应里没有 data 数组：{json.dumps(payload, ensure_ascii=False)[:500]}")

        pil_images = [_decode_item(item) for item in items]
        upstream_sizes = [f"{image.size[0]}x{image.size[1]}" for image in pil_images]

        resized_images = []
        resized_locally = False
        target_sizes = []
        for image in pil_images:
            target = None
            if custom_size:
                target = (width, height)
            elif ratio_value:
                source_w, source_h = image.size
                long_edge = max(source_w, source_h)
                if ratio_value >= 1.0:
                    target = (long_edge, max(1, int(round(long_edge / ratio_value))))
                else:
                    target = (max(1, int(round(long_edge * ratio_value))), long_edge)
            target_sizes.append(target)

            if target and image.size != target:
                resized_images.append(image.resize(target, Image.LANCZOS))
                resized_locally = True
            else:
                resized_images.append(image)
        pil_images = resized_images

        tensors = [_pil_to_tensor(image) for image in pil_images]
        first_shape = tensors[0].shape
        if all(tensor.shape == first_shape for tensor in tensors):
            batch = torch.cat(tensors, dim=0)
        else:
            batch = tensors[0]

        urls = []
        for item in items:
            if isinstance(item, dict):
                url = str(item.get("url") or item.get("image_url") or "")
                if url:
                    urls.append(url)

        response_info = {key: value for key, value in payload.items() if key != "data"}
        response_info["requested_size"] = request_size
        response_info["aspect_ratio"] = aspect_ratio
        response_info["output_size"] = f"{batch.shape[2]}x{batch.shape[1]}"
        response_info["upstream_size"] = upstream_sizes[0] if upstream_sizes else ""
        response_info["target_size"] = (
            f"{target_sizes[0][0]}x{target_sizes[0][1]}"
            if target_sizes and target_sizes[0] else ""
        )
        response_info["resized_locally"] = resized_locally
        response_info["size_hint_appended"] = size_hint_appended

        return (
            batch.cpu(),
            "\n".join(urls),
            json.dumps(response_info, ensure_ascii=False),
            json.dumps(payload, ensure_ascii=False),
            status_code,
        )


NODE_CLASS_MAPPINGS = {"PixelAPIImage2Node": PixelAPIImage2Node}
NODE_DISPLAY_NAME_MAPPINGS = {"PixelAPIImage2Node": "JMHL Image API"}
