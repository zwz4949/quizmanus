# from utils import process_pdf, process_pdfs_in_directory
# from utils import remove_jpg_lines
# pdf_root = "/hpc2hdd/home/fye374/ZWZ_Other/langmanus/src/RAG/knowledge/pdf/物理学"  # 替换为你的实际目录路径
# process_pdfs_in_directory(pdf_root)
# pdf_path = "/hpc2hdd/home/fye374/ZWZ_Other/langmanus/src/RAG_src/课本pdf/生物学 7年级上册.pdf"
# process_pdf(pdf_path)

import re
from typing import List
def get_absolute_file_paths(absolute_dir,file_type)->List[str]:
    '''
    absolute_dir: 文件夹
    file_type: "md","json"...
    '''
    json_files = [os.path.join(absolute_dir,f) for f in os.listdir(absolute_dir) if f.endswith(f".{file_type}")]
    return json_files
def remove_jpg_lines(md_file_path):
    """
    读取Markdown文件并删除包含.jpg图片的行
    
    参数:
        md_file_path (str): Markdown文件的路径
        
    返回:
        str: 处理后的Markdown内容
    """
    with open(md_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 使用正则表达式匹配包含.jpg的行
    pattern = re.compile(r'.*\.jpg.*')
    new_lines = [line for line in lines if not pattern.search(line)]
    
    output_content = ''.join(new_lines)

    with open(md_file_path, 'w', encoding='utf-8') as file:
        file.write(output_content)



import json
import re
import sys
sys.path.append("/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src")
from utils import saveData, get_absolute_file_paths

import glob

def get_title_content(dir_path):
    def parse_md(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
        
        result = []
        current_title = None
        current_content = []
        
        for line in lines:
            if line.startswith('#'):
                if current_title is not None:
                    content = '\n'.join(current_content)
                    if content.strip() != "" and  "第" not in current_title and "章" not in current_title and "单元" not in current_title and "节" not in current_title and "上册" not in current_title and "下册" not in current_title and "目录" not in current_title.replace(" ",""):
                        result.append({"title":current_title.strip(), "content":content})
                current_title = line
                current_content = []
            else:
                if current_title is not None:
                    current_content.append(line)
        
        if current_title is not None:
            content = '\n'.join(current_content)
            if content.strip() != "" and  "第" not in current_title and "章" not in current_title and "单元" not in current_title and "节" not in current_title and "上册" not in current_title and "下册" not in current_title and "目录" not in current_title.replace(" ",""):
                result.append({"title":current_title.strip(), "content":content})
        
        return result

    # 使用示例：解析当前目录下所有.md文件
    absolute_paths = get_absolute_file_path_from_absolute_path(absolute_path = dir_path, file_type = "md")
    print(absolute_paths)
    # md_file = "/hpc2hdd/home/fye374/ZWZ_Other/langmanus/src/RAG_src/课本pdf/地理学/初中地理 八年级上册/auto/初中地理 八年级上册.md"
    for md_file in absolute_paths:
        remove_jpg_lines(md_file)
        save_path = md_file.split(".md")[0] + ".json"
        print(save_path)
        saveData(parse_md(md_file),save_path)
# 打印结果
# get_title_content("/hpc2hdd/home/fye374/ZWZ_Other/langmanus/src/RAG_src/课本md/初中/生物学")

# # 读取 md 文件内容
# def parse_text_to_json_(md_file,parse_text_to_json):
    
#     json_file = md_file.split(".md")[0] + ".json"
#     with open(md_file, 'r', encoding='utf-8') as file:
#         text = file.read()

#     result = parse_text_to_json(text)

#     # 将结果存到 json 文件
#     with open(json_file, 'w', encoding='utf-8') as json_file:
#         json.dump(result, json_file, ensure_ascii=False, indent=2)

# md_file = '/hpc2hdd/home/fye374/ZWZ_Other/langmanus/src/RAG_src/课本pdf/生物学/生物学 8年级下册/auto/生物学 8年级下册.md'
# # remove_jpg_lines(md_file)
# parse_text_to_json_(md_file,parse_text_to_json_1)


def parse_md(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
        
        result = []
        current_title = None
        current_content = []
        
        for line in lines:
            if line.startswith('#'):
                if current_title is not None:
                    content = '\n'.join(current_content)
                    if content.strip() != "" and  "第" not in current_title and "章" not in current_title and "单元" not in current_title and "节" not in current_title and "上册" not in current_title and "下册" not in current_title and "目录" not in current_title.replace(" ",""):
                        result.append({"title":current_title.strip(), "content":content})
                current_title = line
                current_content = []
            else:
                if current_title is not None:
                    current_content.append(line)
        
        if current_title is not None:
            content = '\n'.join(current_content)
            if content.strip() != "" and  "第" not in current_title and "章" not in current_title and "单元" not in current_title and "节" not in current_title and "上册" not in current_title and "下册" not in current_title and "目录" not in current_title.replace(" ",""):
                result.append({"title":current_title.strip(), "content":content})
        
        return result

paths = get_absolute_file_paths('/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md',"md")

# for path in paths:
#     remove_jpg_lines(path)
for md_file in paths:
        remove_jpg_lines(md_file)
        save_path = md_file.split(".md")[0] + ".json"
        print(save_path)
        saveData(parse_md(md_file),save_path)

'''

magic-pdf -p "/hpc2hdd/home/fye374/ZWZ_Other/langmanus/src/RAG_src/课本pdf/生物学 8年级下册.pdf" -o "/hpc2hdd/home/fye374/ZWZ_Other/langmanus/src/RAG_src/课本pdf/生物学"

'''


'''
magic-pdf --help
Usage: magic-pdf [OPTIONS]

Options:
  -v, --version                display the version and exit
  -p, --path PATH              local filepath or directory. support PDF, PPT,
                               PPTX, DOC, DOCX, PNG, JPG files  [required]
  -o, --output-dir PATH        output local directory  [required]
  -m, --method [ocr|txt|auto]  the method for parsing pdf. ocr: using ocr
                               technique to extract information from pdf. txt:
                               suitable for the text-based pdf only and
                               outperform ocr. auto: automatically choose the
                               best method for parsing pdf from ocr and txt.
                               without method specified, auto will be used by
                               default.
  -l, --lang TEXT              Input the languages in the pdf (if known) to
                               improve OCR accuracy.  Optional. You should
                               input "Abbreviation" with language form url: ht
                               tps://paddlepaddle.github.io/PaddleOCR/en/ppocr
                               /blog/multi_languages.html#5-support-languages-
                               and-abbreviations
  -d, --debug BOOLEAN          Enables detailed debugging information during
                               the execution of the CLI commands.
  -s, --start INTEGER          The starting page for PDF parsing, beginning
                               from 0.
  -e, --end INTEGER            The ending page for PDF parsing, beginning from
                               0.
  --help                       Show this message and exit.


## show version
magic-pdf -v

## command line example
magic-pdf -p {some_pdf} -o {some_output_dir} -m auto
'''