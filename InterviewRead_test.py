import re

      
if __name__=="__main__":

    print('正在读取访谈内容...')
    with open("/home/wcy/code/InterviewSystem-v0.1/input/Musk.txt", "r", encoding="utf8") as fin:
        lines = fin.readlines()
    data = []  # 用来存储说话人的内容
    character = []  # 用来存储说话人的序号
    speaker = None  # 当前说话人
    content = ""  # 当前发言内容

    # 正则表达式：提取说话人编号和时间戳
    speaker_pattern = re.compile(r"说话人(\d)")

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
    
    print(len(data))
    print(len(character))
    print(data)
    print(character)