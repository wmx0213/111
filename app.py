"""
智能问答系统主应用
整合豆包API、文本分类、情感分析、机器翻译功能
"""
from flask import Flask, render_template, request, jsonify
import os
import sys

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.doubao_api import DoubaoAPI
from services.text_classifier import TextClassifier
from services.sentiment_analyzer import SentimentAnalyzer
from services.translator import Translator

app = Flask(__name__)

# 初始化各个服务
doubao_api = DoubaoAPI()
text_classifier = TextClassifier()
sentiment_analyzer = SentimentAnalyzer()
translator = Translator()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """豆包API智能问答接口"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': '消息不能为空'}), 400
        
        # 调用豆包API
        response = doubao_api.chat(user_message)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify():
    """文本分类接口"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '文本不能为空'}), 400
        
        # 调用文本分类
        category, confidence = text_classifier.predict(text)
        
        return jsonify({
            'success': True,
            'category': category,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment', methods=['POST'])
def sentiment():
    """情感分析接口"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '文本不能为空'}), 400
        
        # 调用情感分析
        sentiment_label, confidence = sentiment_analyzer.predict(text)
        
        return jsonify({
            'success': True,
            'sentiment': sentiment_label,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    """机器翻译接口"""
    try:
        data = request.json
        text = data.get('text', '')
        target_lang = data.get('target_lang', 'en')  # 默认翻译为英文
        
        if not text:
            return jsonify({'error': '文本不能为空'}), 400
        
        # 调用机器翻译
        translated_text = translator.translate(text, target_lang)
        
        return jsonify({
            'success': True,
            'original': text,
            'translated': translated_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/comprehensive', methods=['POST'])
def comprehensive():
    """综合分析接口 - 对输入文本进行多维度分析"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '文本不能为空'}), 400
        
        results = {}
        
        # 1. 文本分类
        try:
            category, cat_confidence = text_classifier.predict(text)
            results['classification'] = {
                'category': category,
                'confidence': float(cat_confidence)
            }
        except Exception as e:
            results['classification'] = {'error': str(e)}
        
        # 2. 情感分析
        try:
            sentiment_label, sent_confidence = sentiment_analyzer.predict(text)
            results['sentiment'] = {
                'label': sentiment_label,
                'confidence': float(sent_confidence)
            }
        except Exception as e:
            results['sentiment'] = {'error': str(e)}
        
        # 3. 机器翻译（翻译为英文）
        try:
            translated_text = translator.translate(text, 'en')
            results['translation'] = {
                'original': text,
                'translated': translated_text
            }
        except Exception as e:
            results['translation'] = {'error': str(e)}
        
        # 4. 智能问答（使用豆包API）
        try:
            qa_response = doubao_api.chat(f"请简要分析以下文本：{text}")
            results['qa_analysis'] = {
                'response': qa_response
            }
        except Exception as e:
            results['qa_analysis'] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
    except ImportError:
        FLASK_HOST = '0.0.0.0'
        FLASK_PORT = 5000
        FLASK_DEBUG = True
    
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
