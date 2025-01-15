import re

      
if __name__=="__main__":

    print('正在读取访谈内容...')
    with open("/home/wcy/code/InterviewSystem-v0.1/input/test.txt", "r", encoding="utf8") as fin:
        lines = fin.readlines()
    data = []  # 用来存储说话人的内容
    character = []  # 用来存储说话人的序号
    speaker = None  # 当前说话人
    content = ""  # 当前发言内容

    # 正则表达式：提取说话人编号和时间戳
    speaker_pattern = re.compile(r'(\d+)说话人\d+:\d{2}')

    # 逐行分析文件内容
    for line in lines:
        line = line.strip()  # 去掉行首尾的空格和换行符
        # 如果是一个新的说话人，记录当前发言并准备处理新的发言
        match = speaker_pattern.match(line)
        if match:
            if speaker is not None:  # 如果之前有说话人，保存之前的内容
                data.append(content.strip())
                character.append(speaker)
            speaker = int(match.group(1))  # 更新当前说话人
            content = ""  # 重置内容
            print(f"Matched Speaker: {speaker}")

        else:
            # 不是新的说话人，累积发言内容
            content += line + " "

    # 最后一条内容处理（文件结束时）
    if speaker is not None:
        data.append(content.strip())
        character.append(speaker)
    

    names = []
    descriptions = []

    # 正则表达式：提取名字和描述（假设格式为 '名字: 描述'）
    pattern = re.compile(r'([^:]+):\s*(.+)')

    # 逐行读取文件
    with open("/home/wcy/code/InterviewSystem-v0.1/input/character.txt", "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()  # 去掉首尾空格和换行符

            # 使用正则表达式匹配
            match = pattern.match(line)
            if match:
                name = match.group(1)  # 提取名字
                description = match.group(2)  # 提取描述
                names.append(name)
                descriptions.append(description)
    
    Prompt = "The content is below:\n"
    for i in range(len(data)):
       Prompt = Prompt + f"{names[character[i]-1]}:{data[i]}\n"
    print(Prompt)
