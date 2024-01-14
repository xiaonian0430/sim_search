# 推理加速

## 1、torch 加速优化
```
model_name = 'D:\\chinese_roberta_wwm_large_ext'
model = BertModel.from_pretrained(model_name)
model_quantized = torch.quantization.quantize_dynamic(
    model.to("cpu"), {torch.nn.Linear}, dtype=torch.qint8
)
```

## ONNX加速BERT特征抽取

### 将 transformers 模型导出为 ONNX

pip install onnx==1.15.0

pip install onnxruntime==1.16.3



