import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

# 数据库文件路径
DB_PATH = 'users.db'

def init_db():
    """初始化数据库，创建用户表"""
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        ''')
        conn.commit()
        conn.close()
        print(f"数据库初始化完成: {DB_PATH}")
    else:
        print(f"数据库已存在: {DB_PATH}")

def register_user(username, password):
    """
    注册新用户
    
    参数:
        username (str): 用户名
        password (str): 密码
    
    返回:
        dict: 包含注册结果的字典 {'success': bool, 'message': str}
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 检查用户名是否已存在
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            return {'success': False, 'message': '用户名已存在'}
        
        # 密码哈希处理
        password_hash = generate_password_hash(password)
        
        # 插入新用户
        cursor.execute(
            'INSERT INTO users (username, password_hash) VALUES (?, ?)',
            (username, password_hash)
        )
        conn.commit()
        conn.close()
        
        return {'success': True, 'message': '注册成功'}
    
    except sqlite3.Error as e:
        return {'success': False, 'message': f'数据库错误: {str(e)}'}

def verify_user(username, password):
    """
    验证用户登录
    
    参数:
        username (str): 用户名
        password (str): 密码
    
    返回:
        dict: 包含验证结果的字典 {'success': bool, 'message': str}
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 查询用户
        cursor.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return {'success': False, 'message': '用户不存在'}
        
        # 验证密码
        if check_password_hash(user[1], password):
            return {'success': True, 'message': '登录成功'}
        else:
            return {'success': False, 'message': '密码错误'}
    
    except sqlite3.Error as e:
        return {'success': False, 'message': f'数据库错误: {str(e)}'}