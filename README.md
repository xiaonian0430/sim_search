# 相似性搜索

当前时代，是一个信息大爆炸的时代，快速有效的搜索和分析海量数据成为了许多企业和组织的重要需求。

Elasticsearch是一个分布式，RESTful风格的搜索和数据分析引擎，能够为我们提供优秀的解决方案。除了传统的短文本模糊匹配搜索外，Elasticsearch 7.x版本中还引入了向量检索的概念，能够进行向量计算，最大程度的提高了向量相似度的查询性能。

## 使用场景

- 相似文档搜索。将文档转换成向量，使用向量相似性函数，例如：do product，cosine similarity，可以快速检索到最相似的文档，从而实现精确且高效的相似文档检索。
- 推荐系统。将用户画像和商品等表示为向量，根据用户喜好，行为，推荐与其兴趣相似的商品。
- 图像搜索。将图像转换为向量，使用向量相似度计算函数，从图像库中快速检索到最相似的图像。

## 软件要求

python 3.8.18

anaconda 最新版本，修改默认环境目录

conda create -n test python=3.8

conda env remove --name test -y

activate test

pip install transformers==4.35.2
pip install torch==2.1.2


解决：将chinese_L-12_H-768_A-12解压后的目录中Bert_config.json 改名为 config.json

## 测试数据和模型

https://blog.csdn.net/qq_43692950/article/details/132645864

**数据集**

https://github.com/xiaonian0430/Chinese-medical-dialogue-data

**测试模型**

https://huggingface.co/hfl/chinese-roberta-wwm-ext-large

https://github.com/ymcui/Chinese-BERT-wwm

https://github.com/ymcui/Chinese-LLaMA-Alpaca-2





