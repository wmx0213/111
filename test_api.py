"""
API测试脚本
用于测试各个API接口是否正常工作
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_chat():
    """测试智能问答接口"""
    print("\n=== 测试智能问答接口 ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": "什么是人工智能？"},
            timeout=30
        )
        data = response.json()
        if data.get('success'):
            print("✓ 智能问答接口正常")
            print(f"  回答: {data.get('response', '')[:100]}...")
        else:
            print(f"✗ 智能问答接口错误: {data.get('error')}")
    except Exception as e:
        print(f"✗ 智能问答接口异常: {str(e)}")

def test_classify():
    """测试文本分类接口"""
    print("\n=== 测试文本分类接口 ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/classify",
            json={"text": "今天股市大涨，投资者信心增强"},
            timeout=30
        )
        data = response.json()
        if data.get('success'):
            print("✓ 文本分类接口正常")
            print(f"  类别: {data.get('category')}")
            print(f"  置信度: {data.get('confidence', 0):.2%}")
        else:
            print(f"✗ 文本分类接口错误: {data.get('error')}")
    except Exception as e:
        print(f"✗ 文本分类接口异常: {str(e)}")

def test_sentiment():
    """测试情感分析接口"""
    print("\n=== 测试情感分析接口 ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/sentiment",
            json={"text": "这个产品真的很棒，我非常满意！"},
            timeout=30
        )
        data = response.json()
        if data.get('success'):
            print("✓ 情感分析接口正常")
            print(f"  情感: {data.get('sentiment')}")
            print(f"  置信度: {data.get('confidence', 0):.2%}")
        else:
            print(f"✗ 情感分析接口错误: {data.get('error')}")
    except Exception as e:
        print(f"✗ 情感分析接口异常: {str(e)}")

def test_translate():
    """测试机器翻译接口"""
    print("\n=== 测试机器翻译接口 ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/translate",
            json={"text": "你好，世界！", "target_lang": "en"},
            timeout=30
        )
        data = response.json()
        if data.get('success'):
            print("✓ 机器翻译接口正常")
            print(f"  原文: {data.get('original')}")
            print(f"  译文: {data.get('translated')}")
        else:
            print(f"✗ 机器翻译接口错误: {data.get('error')}")
    except Exception as e:
        print(f"✗ 机器翻译接口异常: {str(e)}")

def test_comprehensive():
    """测试综合分析接口"""
    print("\n=== 测试综合分析接口 ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/comprehensive",
            json={"text": "今天科技股表现强劲，投资者情绪高涨"},
            timeout=60
        )
        data = response.json()
        if data.get('success'):
            print("✓ 综合分析接口正常")
            results = data.get('results', {})
            if 'classification' in results:
                print(f"  分类: {results['classification'].get('category', 'N/A')}")
            if 'sentiment' in results:
                print(f"  情感: {results['sentiment'].get('label', 'N/A')}")
            if 'translation' in results:
                print(f"  翻译: {results['translation'].get('translated', 'N/A')[:50]}...")
        else:
            print(f"✗ 综合分析接口错误: {data.get('error')}")
    except Exception as e:
        print(f"✗ 综合分析接口异常: {str(e)}")

def main():
    """主测试函数"""
    print("=" * 50)
    print("智能问答系统 API 测试")
    print("=" * 50)
    print(f"\n测试服务器: {BASE_URL}")
    print("请确保Flask应用已启动！\n")
    
    # 检查服务器是否运行
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print("✓ 服务器连接正常\n")
    except Exception as e:
        print(f"✗ 无法连接到服务器: {str(e)}")
        print("请先启动Flask应用: python app.py")
        return
    
    # 运行各项测试
    test_chat()
    test_classify()
    test_sentiment()
    test_translate()
    test_comprehensive()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()
