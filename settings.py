# settings.py
"""
共享设置模块，使用内存变量存储 URL、API 和 QA Bot
"""

# 内存中的缓存变量
_url = ""
_api = ""
_qa_bot = None

def has_existing_settings():
    """检查是否已有现有配置"""
    global _url, _api
    return bool(_url and _api)

def clear_settings():
    """清除现有配置"""
    global _url, _api
    _url = ""
    _api = ""

def get_url():
    """获取 URL"""
    global _url
    return _url

def set_url(url):
    """设置 URL"""
    global _url
    _url = url

def get_api():
    """获取 API"""
    global _api
    return _api

def set_api(api):
    """设置 API"""
    global _api
    _api = api

def get_settings():
    """获取所有设置"""
    return {
        'url': _url,
        'api': _api
    }

def set_settings(url, api):
    """一次性设置所有值"""
    global _url, _api
    _url = url
    _api = api

def get_qa_bot():
    """获取 QA Bot 实例"""
    global _qa_bot
    return _qa_bot

def set_qa_bot(bot):
    """设置 QA Bot 实例"""
    global _qa_bot
    _qa_bot = bot