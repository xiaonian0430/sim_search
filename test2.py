from elasticsearch import Elasticsearch
import time

def search_similar(index_name, query_text, es, top_k=3):
    query = {
        "_source": [
            "ask",
            "answer"
        ],
        "query": {
            "more_like_this": {
                "fields": ["ask", "answer"],
                "like": query_text,
                "min_term_freq": 1,
                "max_query_terms": 12
            }
        },
        "size": top_k
    }
    res = es.search(index=index_name, body=query)
    print(res)
    return res['hits']['hits']


def main():
    # ES 信息
    es_host = "http://192.168.243.128"
    es_port = 9200
    es_user = "elastic"
    es_password = "aa5626188"
    index_name = "medical_index"

    # ES 连接
    es = Elasticsearch(
        [es_host],
        port=es_port,
        http_auth=(es_user, es_password)
    )

    query_text = "我有高血压可以拿党参泡水喝吗"
    timestamp = int(time.time() * 1000)
    similar_documents = search_similar(index_name, query_text, es)
    timestamp2 = int(time.time() * 1000)
    print((timestamp2 - timestamp) / 1000)
    for item in similar_documents:
        print("================================")
        print(item)


if __name__ == '__main__':
    main()
