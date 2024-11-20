def split_text_file(file_path, chunk_size, overlap):
    """
    将非结构化文本文件分块，并支持块间重叠。

    :param file_path: 文本文件路径
    :param chunk_size: 每块的字符数
    :param overlap: 块之间的重叠字符数
    :return: 分块后的文本列表
    """
    if overlap >= chunk_size:
        raise ValueError("重叠大小不能大于或等于块大小。")

    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()  # 读取整个文本
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(text[start:end])
                start += chunk_size - overlap  # 移动起始点，考虑重叠
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"发生错误: {e}")
    return chunks