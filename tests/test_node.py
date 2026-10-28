import base64
import importlib.util
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch
from PIL import Image

NODE_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("jmhl_image_api_node", NODE_DIR / "nodes.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

captured = []
responses = []


class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        captured.append({"path": self.path, "body": body})
        index = len(captured) - 1
        entry = responses[index] if index < len(responses) else ("image", (16, 16), (30, 120, 200))

        if entry[0] == "http":
            data = json.dumps({"error": {"message": "boom"}}).encode()
            self.send_response(entry[1])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        size, color = entry[1], entry[2]
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        payload = {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}], "created": 1}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def mock_api():
    captured.clear()
    responses.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def make_node():
    return mod.PixelAPIImage2Node()


def base_inputs(api_base):
    return {
        "模式": "图生图",
        "提示词": "test prompt",
        "模型名称": "gpt-image-2.5-Sunburst",
        "API地址": api_base,
        "API密钥": "test-key-12345",
        "宽高比": "1:1",
        "宽度": 0,
        "高度": 0,
        "画质": "auto",
        "背景": "opaque",
        "输出格式": "png",
        "随机种": 1,
        "超时时间_秒": 60,
        "额外参数JSON": "{}",
        "额外请求头JSON": "{}",
        "图片1": torch.zeros((1, 8, 8, 3), dtype=torch.float32),
    }


def test_node_contract():
    assert "PixelAPIImage2Node" in mod.NODE_CLASS_MAPPINGS
    cls = mod.NODE_CLASS_MAPPINGS["PixelAPIImage2Node"]
    assert cls.FUNCTION == "run"
    assert cls.RETURN_TYPES == ("IMAGE", "STRING", "STRING", "STRING", "INT")
    assert cls.RETURN_NAMES == ("图像", "图片链接", "响应信息", "原始响应", "HTTP状态码")
    assert cls.CATEGORY == "JMHL API"

    inputs = cls.INPUT_TYPES()
    required = set(inputs["required"])
    assert {
        "模式", "提示词", "模型名称", "API地址", "API密钥",
        "宽高比", "宽度", "高度", "画质", "背景", "输出格式",
        "随机种", "超时时间_秒", "额外参数JSON", "额外请求头JSON",
    } <= required
    optional = set(inputs["optional"])
    assert {"图片1", "图片6", "自动补充尺寸提示词"} <= optional


def test_normalize_endpoint():
    assert mod._normalize_endpoint("https://api.example.com", "edits") == "https://api.example.com/v1/images/edits"
    assert mod._normalize_endpoint("https://api.example.com/v1", "generations") == "https://api.example.com/v1/images/generations"
    assert (
        mod._normalize_endpoint("https://api.example.com/v1/images/edits", "edits")
        == "https://api.example.com/v1/images/edits"
    )
    with pytest.raises(ValueError):
        mod._normalize_endpoint("   ", "edits")


def test_custom_size_requests_and_resizes(mock_api):
    node = make_node()
    inputs = base_inputs(mock_api)
    inputs["宽度"] = 64
    inputs["高度"] = 32

    images, urls, info_text, raw, status = node.run(**inputs)

    assert status == 200
    assert captured[0]["path"] == "/v1/images/edits"
    assert b"64x32" in captured[0]["body"]
    assert images.shape == (1, 32, 64, 3)
    info = json.loads(info_text)
    assert info["requested_size"] == "64x32"
    assert info["output_size"] == "64x32"
    assert info["resized_locally"] is True


def test_ratio_mode_scales_to_ratio(mock_api):
    node = make_node()
    inputs = base_inputs(mock_api)
    inputs["宽高比"] = "16:9"

    images, urls, info_text, raw, status = node.run(**inputs)

    assert b"1280x720" in captured[0]["body"]
    assert images.shape == (1, 9, 16, 3)
    info = json.loads(info_text)
    assert info["requested_size"] == "1280x720"
    assert info["target_size"] == "16x9"
    assert info["resized_locally"] is True


def test_auto_keeps_upstream_size(mock_api):
    node = make_node()
    inputs = base_inputs(mock_api)
    inputs["宽高比"] = "auto"

    images, urls, info_text, raw, status = node.run(**inputs)

    assert images.shape == (1, 16, 16, 3)
    info = json.loads(info_text)
    assert info["requested_size"] == "auto"
    assert info["resized_locally"] is False


def test_size_hint_appended(mock_api):
    node = make_node()
    inputs = base_inputs(mock_api)
    inputs["宽度"] = 970
    inputs["高度"] = 600

    node.run(**inputs)

    hint = "生成的图片尺寸为970px 600px".encode("utf-8")
    assert hint in captured[0]["body"]


def test_size_hint_not_duplicated(mock_api):
    node = make_node()
    inputs = base_inputs(mock_api)
    inputs["宽度"] = 970
    inputs["高度"] = 600
    inputs["提示词"] = "test prompt\n生成的图片尺寸为970px 600px"

    node.run(**inputs)

    hint = "生成的图片尺寸为970px 600px".encode("utf-8")
    assert captured[0]["body"].count(hint) == 1


def test_size_hint_disabled(mock_api):
    node = make_node()
    inputs = base_inputs(mock_api)
    inputs["宽度"] = 970
    inputs["高度"] = 600
    inputs["自动补充尺寸提示词"] = False

    node.run(**inputs)

    hint = "生成的图片尺寸为970px 600px".encode("utf-8")
    assert hint not in captured[0]["body"]


def test_missing_api_key(mock_api, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    node = make_node()
    inputs = base_inputs(mock_api)
    inputs["API密钥"] = ""

    with pytest.raises(RuntimeError, match="API密钥为空"):
        node.run(**inputs)


def test_missing_prompt():
    node = make_node()
    inputs = base_inputs("https://api.example.com/v1")
    inputs["提示词"] = "  "

    with pytest.raises(RuntimeError, match="提示词为空"):
        node.run(**inputs)


def test_one_sided_size_rejected():
    node = make_node()
    inputs = base_inputs("https://api.example.com/v1")
    inputs["宽度"] = 970
    inputs["高度"] = 0

    with pytest.raises(RuntimeError, match="同时填写"):
        node.run(**inputs)


def test_retry_after_blank_then_success(mock_api):
    responses.append(("image", (16, 16), (0, 0, 0)))
    responses.append(("image", (16, 16), (30, 120, 200)))
    node = make_node()
    inputs = base_inputs(mock_api)
    inputs["最大重试次数"] = 2

    images, urls, info_text, raw, status = node.run(**inputs)

    assert len(captured) == 2
    assert images.shape == (1, 16, 16, 3)
    info = json.loads(info_text)
    assert info["attempts"] == 2
    assert info["max_attempts"] == 3
    assert info["blank_retries"] == 1


def test_retry_exhausted_blank_raises(mock_api):
    for _ in range(3):
        responses.append(("image", (16, 16), (0, 0, 0)))
    node = make_node()
    inputs = base_inputs(mock_api)
    inputs["最大重试次数"] = 2

    with pytest.raises(RuntimeError, match="共尝试 3 次"):
        node.run(**inputs)
    assert len(captured) == 3


def test_retry_on_http_error(mock_api):
    responses.append(("http", 500))
    responses.append(("image", (16, 16), (30, 120, 200)))
    node = make_node()
    inputs = base_inputs(mock_api)
    inputs["最大重试次数"] = 2

    images, urls, info_text, raw, status = node.run(**inputs)

    assert len(captured) == 2
    assert images.shape == (1, 16, 16, 3)
    info = json.loads(info_text)
    assert info["attempts"] == 2


def test_no_retry_when_disabled(mock_api):
    responses.append(("image", (16, 16), (0, 0, 0)))
    node = make_node()
    inputs = base_inputs(mock_api)
    inputs["最大重试次数"] = 0

    with pytest.raises(RuntimeError, match="共尝试 1 次"):
        node.run(**inputs)
    assert len(captured) == 1
