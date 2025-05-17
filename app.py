# app.py
import os
from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
import settings  # 导入共享设置模块
import user_db   # 导入用户数据库模块

import os
import time
import uuid
from werkzeug.utils import secure_filename


# 创建 Flask 应用
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)  # 用于session加密



# 配置文件上传
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploadFile')
ALLOWED_EXTENSIONS = {
    'text': {'txt', 'pdf', 'doc', 'docx', 'csv', 'xls', 'xlsx', 'json', 'md'},
    'image': {'jpg', 'jpeg', 'png', 'gif', 'svg', 'webp', 'bmp', 'tif', 'tiff'},
    'video': {'mp4', 'avi', 'mov', 'wmv', 'mkv', 'webm', 'flv'},
    'audio': {'mp3', 'wav', 'ogg', 'aac', 'flac'}
}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

# 添加到 Flask 应用配置中
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH



@app.route('/upload_files', methods=['POST'])
def upload_files():
    """
    处理多种格式文件上传的函数
    支持文本、图片、视频等多种格式
    文件存储在根目录的 uploadFile 文件夹中
    """
    # 检查用户是否已认证
    if not is_authenticated():
        return jsonify({'success': False, 'message': '请先登录'}), 401
    
    # 检查请求中是否有文件
    if 'files[]' not in request.files:
        return jsonify({'success': False, 'message': '没有文件被上传'}), 400
    
    # 获取所有上传的文件
    files = request.files.getlist('files[]')
    
    # 如果没有选择文件，浏览器也会提交空的文件，所以需要检查
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'message': '没有选择文件'}), 400
    
    # 创建上传目录结构
    if not os.path.exists(UPLOAD_FOLDER):
        try:
            os.makedirs(UPLOAD_FOLDER)
        except Exception as e:
            return jsonify({'success': False, 'message': f'创建上传目录失败: {str(e)}'}), 500
    
    # 为每种文件类型创建子目录
    for file_type in ALLOWED_EXTENSIONS.keys():
        type_dir = os.path.join(UPLOAD_FOLDER, file_type)
        if not os.path.exists(type_dir):
            try:
                os.makedirs(type_dir)
            except Exception as e:
                print(f"创建目录失败: {type_dir}, 错误: {str(e)}")
    
    # 处理结果存储
    results = []
    uploaded_count = 0
    
    # 处理每个文件
    for file in files:
        result = {
            'filename': file.filename,
            'success': False,
            'message': '',
            'path': ''
        }
        
        try:
            # 安全地获取文件名
            filename = secure_filename(file.filename)
            if not filename:
                result['message'] = '文件名不合法'
                results.append(result)
                continue
            
            # 获取文件扩展名
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            
            # 验证文件类型
            file_type = None
            for type_name, extensions in ALLOWED_EXTENSIONS.items():
                if file_ext in extensions:
                    file_type = type_name
                    break
            
            if not file_type:
                result['message'] = f'不支持的文件类型: .{file_ext}'
                results.append(result)
                continue
            
            # 创建唯一的文件名 (时间戳 + UUID + 原始文件名)
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4().hex[:8])
            new_filename = f"{timestamp}_{unique_id}_{filename}"
            
            # 确定存储路径
            save_path = os.path.join(UPLOAD_FOLDER, file_type, new_filename)
            relative_path = os.path.join('uploadFile', file_type, new_filename)
            
            # 保存文件
            file.save(save_path)
            
            # 更新结果
            result['success'] = True
            result['message'] = '上传成功'
            result['path'] = relative_path
            uploaded_count += 1
            
        except Exception as e:
            result['message'] = f'上传出错: {str(e)}'
        
        results.append(result)
    
    # 返回处理结果
    return jsonify({
        'success': uploaded_count > 0,
        'message': f'成功上传 {uploaded_count}/{len(files)} 个文件',
        'files': results
    })

@app.route('/get_uploaded_files', methods=['GET'])
def get_uploaded_files():
    """获取已上传文件列表"""
    if not is_authenticated():
        return jsonify({'success': False, 'message': '请先登录'}), 401
    
    file_type = request.args.get('type', None)  # 可选参数，按类型筛选
    
    if not os.path.exists(UPLOAD_FOLDER):
        return jsonify({'success': True, 'files': []})
    
    files_list = []
    
    # 如果指定了类型，只查找该类型目录
    if file_type and file_type in ALLOWED_EXTENSIONS:
        type_dir = os.path.join(UPLOAD_FOLDER, file_type)
        if os.path.exists(type_dir):
            for filename in os.listdir(type_dir):
                file_path = os.path.join('uploadFile', file_type, filename)
                files_list.append({
                    'name': filename,
                    'type': file_type,
                    'path': file_path
                })
    else:
        # 否则查找所有类型目录
        for type_name in ALLOWED_EXTENSIONS.keys():
            type_dir = os.path.join(UPLOAD_FOLDER, type_name)
            if os.path.exists(type_dir):
                for filename in os.listdir(type_dir):
                    file_path = os.path.join('uploadFile', type_name, filename)
                    files_list.append({
                        'name': filename,
                        'type': type_name,
                        'path': file_path
                    })
    
    return jsonify({'success': True, 'files': files_list})


    
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