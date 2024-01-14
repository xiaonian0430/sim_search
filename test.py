from elasticsearch import Elasticsearch
from transformers import BertTokenizer, BertModel
import torch
import time


def embeddings_doc(doc, tokenizer, model, max_length=300):
    timestamp = int(time.time() * 1000)
    encoded_dict = tokenizer.encode_plus(
        doc,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    timestamp2 = int(time.time() * 1000)
    print(timestamp2-timestamp)
    # 前向传播
    with torch.no_grad():
        outputs = model(input_id, attention_mask=attention_mask)
    timestamp3 = int(time.time() * 1000)
    print(timestamp3-timestamp2)

    # 提取最后一层的CLS向量作为文本表示
    last_hidden_state = outputs.last_hidden_state
    cls_embeddings = last_hidden_state[:, 0, :]
    return cls_embeddings[0]


def search_similar(index_name, query_text, tokenizer, model, es, top_k=3):
    query_embedding = embeddings_doc(query_text, tokenizer, model)
    query = {
        "_source": [
            "ask",
            "answer"
        ],
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'ask_vector')",
                    "lang": "painless",
                    "params": {
                        "queryVector": query_embedding.tolist()
                    }
                }
            }
        },
        "size": top_k
    }
    res = es.search(index=index_name, body=query)
    return res['hits']['hits']


def main():
    # 模型下载的地址
    model_name = 'D:\\chinese_roberta_wwm_large_ext'

    # ES 信息
    es_host = "http://192.168.243.128"
    es_port = 9200
    es_user = "elastic"
    es_password = "aa5626188"
    index_name = "medical_index"

    # 分词器和模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # ES 连接
    es = Elasticsearch(
        [es_host],
        port=es_port,
        http_auth=(es_user, es_password)
    )

    query_text = "我有高血压可以拿党参泡水喝吗"
    timestamp = int(time.time() * 1000)
    similar_documents = search_similar(index_name, query_text, tokenizer, model, es)
    timestamp2 = int(time.time() * 1000)
    print((timestamp2-timestamp)/1000)
    for item in similar_documents:
        print("================================")
        print(item)


if __name__ == '__main__':
    main()
