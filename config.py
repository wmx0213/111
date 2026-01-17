"""
系统配置文件
"""
import os

# 豆包API配置
DOUBAO_TOKEN = os.getenv('DOUBAO_TOKEN', '67eec362-e72e-45cf-bb03-03aaf16874af')
DOUBAO_HOST = "ark.cn-beijing.volces.com"
DOUBAO_MODEL = "doubao-seed-1-6-lite-251015"

# NLP模型路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLP_BASE_PATH = os.path.join(BASE_DIR, '..', 'NLP', 'nlp_deeplearn')

# 文本分类模型路径
TEXT_CLASSIFIER_MODEL_PATH = os.path.join(NLP_BASE_PATH, 'tmp', 'my_model.h5')
TEXT_CLASSIFIER_VOCAB_PATH = os.path.join(NLP_BASE_PATH, 'data', 'cnews.vocab.txt')

# 情感分析模型路径
SENTIMENT_MODEL_PATH = os.path.join(NLP_BASE_PATH, 'model', 'sentiment_analysis_model.h5')
SENTIMENT_NEG_DATA_PATH = os.path.join(NLP_BASE_PATH, 'data', 'neg.xls')
SENTIMENT_POS_DATA_PATH = os.path.join(NLP_BASE_PATH, 'data', 'pos.xls')

# 机器翻译模型路径
TRANSLATOR_CHECKPOINT_DIR = os.path.join(NLP_BASE_PATH, 'tmp', 'training_checkpoints')
TRANSLATOR_DATA_PATH = os.path.join(NLP_BASE_PATH, 'data', 'en-ch.txt')

# Flask配置
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True
