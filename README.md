# JMHL Image API

ComfyUI 自定义节点：通过任意 **OpenAI 兼容接口** 调用图生图 / 文生图模型，支持自定义 `API地址`，并可指定输出像素尺寸（本地缩放保证最终输出大小）。

- 节点显示名：**JMHL Image API**
- 节点类名：`PixelAPIImage2Node`
- 分类：`JMHL API`

## 功能

- 图生图（有图片输入时走 `/v1/images/edits`，multipart 直接上传 PNG）
- 文生图（无图片输入时走 `/v1/images/generations`，JSON）
- 最多 6 张参考图
- 自定义 `API地址`（官方 OpenAI、中转站、自建网关均可）
- `宽度` / `高度` 指定输出像素尺寸；接口返回尺寸不一致时，本地用 Lanczos 缩放到目标尺寸
- 宽度/高度为 0 时，按 `宽高比` 选择目标尺寸并本地缩放
- `自动补充尺寸提示词`：自动在提示词末尾追加 `生成的图片尺寸为970px 600px` 这样的说明
- 请求超时、失败重试提示、`额外参数JSON` / `额外请求头JSON` 灵活扩展

## 安装

### ComfyUI Manager

选择 **Install via Git URL**，填入本仓库地址：

```text
https://github.com/<your-account>/ComfyUI-JMHL-Image-API
```

### 手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-account>/ComfyUI-JMHL-Image-API.git JMHL_Image_API
```

重启 ComfyUI，节点位于 `JMHL API` 分类下。

依赖只有 `requests`、`Pillow`、`numpy`，ComfyUI 已内置；`torch` 由 ComfyUI 提供，不重复安装。

## 输入

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `模式` | 下拉 | `图生图` / `文生图` |
| `提示词` | STRING | 提示词，必填 |
| `模型名称` | STRING | 模型名，例如 `gpt-image-2.5-Sunburst` |
| `API地址` | STRING | OpenAI 兼容地址，如 `https://api.openai.com/v1` 或中转站 |
| `API密钥` | STRING | 留空时读取环境变量 `OPENAI_API_KEY` |
| `宽高比` | 下拉 | `1:1`、`16:9`、`9:16`、`4:3`、`3:4`、`3:2`、`2:3`、`auto` |
| `宽度` | INT | 可选。0 表示不指定；和高度同时大于 0 时按该像素尺寸生成 |
| `高度` | INT | 可选。同上 |
| `画质` | 下拉 | `auto` / `low` / `medium` / `high` / `xhigh` |
| `背景` | 下拉 | `opaque` / `transparent` / `auto` |
| `输出格式` | 下拉 | `png` / `jpeg` / `webp` |
| `随机种` | INT | 随机种子 |
| `超时时间_秒` | INT | 单次请求超时，默认 300 |
| `额外参数JSON` | STRING | 合并进请求参数，同名会覆盖节点字段 |
| `额外请求头JSON` | STRING | 合并进请求头，可覆盖 `Authorization` |
| `图片1` ~ `图片6` | IMAGE（可选） | 参考图，接任意一张即走图生图 |
| `自动补充尺寸提示词` | BOOLEAN（可选） | 默认开启，见下文 |

## 输出

| 输出 | 类型 | 说明 |
| --- | --- | --- |
| `图像` | IMAGE | 生成结果，多张时为一个 batch |
| `图片链接` | STRING | 接口返回的图片 URL（换行分隔），b64 响应时为空 |
| `响应信息` | STRING | JSON：请求尺寸、目标尺寸、上游实际尺寸、是否本地缩放、是否追加提示词等 |
| `原始响应` | STRING | 原始 JSON 响应。b64 响应会包含完整 base64，不要直接写日志 |
| `HTTP状态码` | INT | 成功时为 200 |

## 尺寸规则

| 宽度 / 高度 | 行为 |
| --- | --- |
| 都 > 0 | 请求发送 `size=宽x高`；接口返回尺寸不一致时，本地 Lanczos 缩放到指定像素 |
| 都为 0，比例不是 `auto` | 请求发送该比例对应的标准尺寸，并把接口返回图缩放到所选比例（以返回图长边为基准） |
| 都为 0，比例 `auto` | 发送 `size=auto`，不做本地缩放，输出尺寸由接口决定 |
| 只填一个 | 报错 |

注意：

- 本地缩放是**直接拉伸**。上游固定返回 1:1 时，目标填 970×600 会产生拉伸变形；比例一致时为等比缩放，无损观感。
- 图生图模式下，很多模型会**跟随输入图的尺寸**返回；需要固定尺寸时必须填写 `宽度/高度`。

## 自动补充尺寸提示词

开启（默认）时，只要确定了目标尺寸（自定义像素或比例映射），节点会在提示词末尾自动追加：

```text
生成的图片尺寸为970px 600px
```

- 提示词里已有相同内容（忽略空格差异）时不会重复追加；
- `宽高比=auto` 且宽高为 0 时不追加；
- 关闭开关后提示词原样发送。

## 接口地址与请求

`API地址` 填 OpenAI 兼容地址，节点会自动补全：

```text
https://your-host            -> https://your-host/v1/images/edits
https://your-host/v1         -> https://your-host/v1/images/edits
https://your-host/v1/images/edits -> 原样使用
```

- 图生图：POST `/v1/images/edits`，multipart 上传 `image[]`（PNG），字段含 `model`、`prompt`、`n`、`quality`、`background`、`output_format`、`size`、`seed`；
- 文生图：POST `/v1/images/generations`，JSON body 同上；
- 响应支持 `b64_json` 和 `url` 两种格式。

## 本地测试

```bash
pip install pytest
pytest -q
```

`examples/api_example.json` 是一个可直接 POST 到 ComfyUI `/prompt` 接口的示例（API 格式），也可以在 ComfyUI 里通过"加载"导入。

## 安全提醒

`API密钥` 是普通 STRING 控件，ComfyUI 会把它写进工作流 JSON。分享工作流前请清空该字段，或使用环境变量 `OPENAI_API_KEY`。

## License

MIT
