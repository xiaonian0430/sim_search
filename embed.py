from transformers import BertTokenizer, BertModel
import torch

# 模型下载的地址
model_name = 'D:\\chinese_roberta_wwm_large_ext'


def embeddings(docs, max_length=300):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    # 对文本进行分词、编码和填充
    input_ids = []
    attention_masks = []
    for doc in docs:
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    # 提取最后一层的CLS向量作为文本表示
    last_hidden_state = outputs.last_hidden_state
    cls_embeddings = last_hidden_state[:, 0, :]
    return cls_embeddings


if __name__ == '__main__':
    res = embeddings(["你好，你叫什么名字"])
    print(res)
    print(len(res))
    print(len(res[0]))
