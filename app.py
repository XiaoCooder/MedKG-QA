# app.py
import os
from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
import settings  # 导入共享设置模块
import user_db   # 导入用户数据库模块

# 创建 Flask 应用
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)  # 用于session加密

# 检查用户是否已认证的函数
def is_authenticated():
    return session.get('authenticated', False)

# 登录路由
@app.route('/', methods=['GET'])
def index():
    """渲染登录页面或重定向到首页"""
    if is_authenticated():
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """处理用户登录"""
    if is_authenticated():
        return redirect(url_for('home'))
    
    message = None
    message_type = None
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # 验证用户输入
        if not username or not password:
            message = '请输入用户名和密码'
            message_type = 'warning'
        else:
            # 验证用户凭据
            result = user_db.verify_user(username, password)
            
            if result['success']:
                # 登录成功，设置session
                session['authenticated'] = True
                session['username'] = username
                # 重定向到首页
                return redirect(url_for('home'))
            else:
                message = result['message']
                message_type = 'danger'
    
    return render_template('login.html', message=message, message_type=message_type)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """处理用户注册"""
    if is_authenticated():
        return redirect(url_for('home'))
    
    message = None
    message_type = None
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # 验证用户输入
        if not username or not password:
            message = '请输入用户名和密码'
            message_type = 'warning'
        elif password != confirm_password:
            message = '两次输入的密码不一致'
            message_type = 'warning'
        else:
            # 注册新用户
            result = user_db.register_user(username, password)
            
            if result['success']:
                message = '注册成功，请登录'
                message_type = 'success'
            else:
                message = result['message']
                message_type = 'danger'
    
    return render_template('register.html', message=message, message_type=message_type)

@app.route('/home')
def home():
    """渲染主页"""
    if not is_authenticated():
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/loading')
def loading_page():
    """渲染默认加载页面 - 加载模型"""
    if not is_authenticated():
        return redirect(url_for('login'))
    return render_template('loading.html',
                          title="模型加载中",
                          heading="模型加载中",
                          initial_text="正在准备模型，请稍候...",
                          status_message="正在初始化编码器...",
                          check_status_url="/check_loading_status",
                          redirect_url="/data_choice",
                          check_interval=1500)

@app.route('/loading/model')
def model_loading_page():
    """渲染模型加载页面"""
    if not is_authenticated():
        return redirect(url_for('login'))
    return render_template('loading.html',
                          title="模型加载中",
                          heading="模型加载中",
                          initial_text="正在准备机器学习模型，请稍候...",
                          status_message="正在初始化编码器...",
                          check_status_url="/check_loading_status",
                          redirect_url="/data_choice",
                          check_interval=1500)

@app.route('/loading/process_data')
def process_data_loading_page():
    """渲染数据处理加载页面"""
    if not is_authenticated():
        return redirect(url_for('login'))
    return render_template('loading.html',
                          title="数据处理中",
                          heading="数据处理中",
                          initial_text="正在处理您的数据，请稍候...",
                          status_message="正在提取知识图谱...",
                          check_status_url="/check_process_data_status",
                          redirect_url="/qa",
                          check_interval=1000)

@app.route('/loading/load_data')
def load_data_loading_page():
    """渲染数据加载页面"""
    if not is_authenticated():
        return redirect(url_for('login'))
    return render_template('loading.html',
                          title="数据加载中",
                          heading="数据加载中",
                          initial_text="正在加载已有数据，请稍候...",
                          status_message="正在读取数据库...",
                          check_status_url="/check_load_data_status",
                          redirect_url="/qa",
                          check_interval=1000)

@app.route('/check_loading_status')
def check_loading_status():
    """检查模型加载状态"""
    if not is_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401
    
    encoder_ready = settings.get_encoder_ready()
    
    if encoder_ready:
        return jsonify({
            'completed': True,
            'message': '模型已加载完成，即将跳转...'
        })
    else:
        return jsonify({
            'completed': False,
            'message': '正在加载模型，请稍候...'
        })

@app.route('/check_process_data_status')
def check_process_data_status():
    """检查数据处理状态"""
    if not is_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401
    
    process_data_ready = settings.get_process_data_ready()
    
    if process_data_ready:
        return jsonify({
            'completed': True,
            'message': '数据处理完成，即将跳转...'
        })
    else:
        return jsonify({
            'completed': False,
            'message': '正在处理数据，提取知识图谱中...'
        })

@app.route('/check_load_data_status')
def check_load_data_status():
    """检查数据加载状态"""
    if not is_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401
    
    load_data_ready = settings.get_load_data_ready()
    
    if load_data_ready:
        return jsonify({
            'completed': True,
            'message': '数据加载完成，即将跳转到问答界面...'
        })
    else:
        return jsonify({
            'completed': False,
            'message': '正在加载已有数据，请稍候...'
        })

@app.route('/data_choice')
def data_choice_page():
    """渲染数据选择页面"""
    if not is_authenticated():
        return redirect(url_for('login'))
    
    # 确保编码器已经准备好
    if not settings.get_encoder_ready():
        return redirect(url_for('loading_page'))
    return render_template('data_choice.html')

@app.route('/select_data_option', methods=['POST'])
def select_data_option():
    """处理用户对数据加载的选择"""
    if not is_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # 从 JSON 中获取选择
        data = request.get_json()
        choice = data.get('choice', '')
        
        # 验证输入
        if choice not in ['yes', 'no']:
            return jsonify({'success': False, 'message': '无效的选择'}), 400
        
        # 保存用户选择
        settings.set_data_choice(choice)
        
        # 根据选择设置重定向页面
        redirect_url = ""
        if choice == 'yes':
            # 设置加载数据状态为未完成
            settings.set_load_data_ready(False)
            redirect_url = '/loading/load_data'
        else:  # choice == 'no'
            # 设置处理数据状态为未完成
            settings.set_process_data_ready(False)
            redirect_url = '/loading/process_data'
        
        # 返回成功，包含重定向URL
        return jsonify({
            'success': True, 
            'message': f'选择已保存: {choice}',
            'redirect': redirect_url
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'发生错误: {str(e)}'}), 500

@app.route('/qa')
def qa_page():
    """渲染问答页面"""
    if not is_authenticated():
        return redirect(url_for('login'))
    return render_template('qa.html')

@app.route('/save_settings', methods=['POST'])
def save_settings():
    """保存用户设置"""
    if not is_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # 从表单中获取 URL 和 API
        url = request.form.get('url', '')
        api = request.form.get('api', '')
        
        # 验证输入不为空
        if not url or not api:
            return jsonify({'success': False, 'message': '请填写 URL 和 API'}), 400
        
        # 保存设置
        settings.set_url(url)
        settings.set_api(api)
        
        # 打印保存的值（调试用）
        print(f"已保存 URL: {url}")
        print(f"已保存 API: {api}")
        
        # 重置编码器状态
        settings.set_encoder_ready(False)
        
        # 返回成功，并指示前端重定向到加载页面
        return jsonify({
            'success': True, 
            'message': '设置已保存',
            'url': url,
            'api': api,
            'redirect': '/loading/model'  # 使用模型加载页面
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'发生错误: {str(e)}'}), 500

@app.route('/get_settings', methods=['GET'])
def get_settings():
    """获取当前设置"""
    if not is_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(settings.get_settings())

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """处理用户问题并返回答案"""
    if not is_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # 获取 qa_bot 实例
        qa_bot = settings.get_qa_bot()
        
        # 检查 qa_bot 是否已初始化
        if qa_bot is None:
            return jsonify({
                'success': False,
                'message': '问答系统尚未初始化完成，请稍后再试'
            }), 503
        
        # 获取问题
        question = request.form.get('question', '')
        
        if not question:
            return jsonify({'success': False, 'message': '请输入问题'}), 400
        
        # 调用 qa_bot 处理问题
        answer = qa_bot.process_web_question(question)
        
        return jsonify({
            'success': True,
            'answer': answer
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'处理问题时出错: {str(e)}'
        }), 500

def start_flask(port=5000):
    """启动 Flask 服务"""
    # 确保用户数据库已初始化
    user_db.init_db()
    
    print(f"启动 Flask 服务，请访问 http://localhost:{port} 登录系统")
    # 禁用调试模式以避免在线程中运行时出现问题
    app.run(debug=False, port=port, threaded=True, use_reloader=False)