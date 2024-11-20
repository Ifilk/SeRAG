import requests
from bs4 import BeautifulSoup
import os
from main_logger import logger


def fetch_wikipedia_content(keyword, save_directory):
    # 创建保存目录（如果不存在的话）
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    url = f"https://en.wikipedia.org/wiki/{keyword}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # title = soup.find('h1').text
        content = ''
        for paragraph in soup.find_all('p'):
            content += paragraph.text
        file_path = os.path.join(save_directory, f"{keyword}.txt")
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

        logger.info(f"内容已保存到 {file_path}")
        return file_path
    else:
        logger.error(f"无法访问{keyword}页面，状态码: {response.status_code}")
        return None
