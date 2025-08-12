# fusion-smp-rk3588

轻量化双光（IR+VIS）图像融合模板：
- **SMP + MobileNetV3 UNet**（`in_channels=2, classes=1`）
- 无监督融合损失：`L1 + SSIM + Sobel 边缘`
- 固定输入尺寸（默认 256×256），便于 **ONNX → RKNN（INT8）**
- 提供 **RK3588 rknn-lite 推理**脚本

## 环境
- Python 3.8+
- 训练：PyTorch + segmentation-models-pytorch
- 转换：ONNX 与 rknn-toolkit2（Rockchip 提供的轮子；若 `pip install rknn-toolkit2` 不可用，请使用官方对应平台的 whl）

```bash
pip install -r requirements.txt
# 若需要：pip install rknn-toolkit2  # 或安装官方 whl
