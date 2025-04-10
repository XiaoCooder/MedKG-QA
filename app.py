# app.py
import os
from flask import Flask, request, render_template, jsonify, redirect, url_for
import settings  # 导入共享设置模块

# 创建 Flask 应用
app = Flask(__name__)

@app.route('/')
def home():
    """渲染首页"""
    return render_template('index.html')

@app.route('/loading')
def loading_page():
    """渲染加载页面"""
    return render_template('loading.html')

@app.route('/check_loading_status')
def check_loading_status():
    """检查模型加载状态"""
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

@app.route('/qa')
def qa_page():
    """渲染问答页面"""
    return render_template('qa.html')

@app.route('/data_choice')
def data_choice_page():
    """渲染数据选择页面"""
    # 确保编码器已经准备好
    if not settings.get_encoder_ready():
        return redirect(url_for('loading_page'))
    return render_template('data_choice.html')

@app.route('/select_data_option', methods=['POST'])
def select_data_option():
    """处理用户对数据加载的选择"""
    try:
        # 从 JSON 中获取选择
        data = request.get_json()
        choice = data.get('choice', '')
        
        # 验证输入
        if choice not in ['yes', 'no']:
            return jsonify({'success': False, 'message': '无效的选择'}), 400
        
        # 保存用户选择
        settings.set_data_choice(choice)
        
        # 返回成功
        return jsonify({
            'success': True, 
            'message': f'选择已保存: {choice}',
            'choice': choice
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'发生错误: {str(e)}'}), 500

@app.route('/save_settings', methods=['POST'])
def save_settings():
    """保存用户设置"""
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
        
        # 返回成功，并指示前端重定向到加载页面
        return jsonify({
            'success': True, 
            'message': '设置已保存',
            'url': url,
            'api': api,
            'redirect': '/loading'  # 修改重定向到加载页面
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'发生错误: {str(e)}'}), 500

@app.route('/get_settings', methods=['GET'])
def get_settings():
    """获取当前设置"""
    return jsonify(settings.get_settings())

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """处理用户问题并返回答案"""
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
    print(f"启动 Flask 服务，请访问 http://localhost:{port} 设置 URL 和 API")
    # 禁用调试模式以避免在线程中运行时出现问题
    app.run(debug=False, port=port, threaded=True, use_reloader=False)