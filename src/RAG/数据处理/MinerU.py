import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from typing import List
def get_absolute_file_paths(absolute_dir,file_type)->List[str]:
    '''
    absolute_dir: 文件夹
    file_type: "md","json"...
    '''
    json_files = [os.path.join(absolute_dir,f) for f in os.listdir(absolute_dir) if f.endswith(f".{file_type}")]
    return json_files

import os

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

def MinerU(pdf_file_name,remain_image = False):
    print(f"开始{pdf_file_name}")
    if remain_image:
        # args
        # pdf_file_name = "small_ocr.pdf"  # replace with the real pdf path
        name_without_suff = pdf_file_name.split(".")[0]

        # prepare env
        local_image_dir, local_md_dir = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md/images", "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/课本md/高中/md"
        image_dir = str(os.path.basename(local_image_dir))

        os.makedirs(local_image_dir, exist_ok=True)

        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
            local_md_dir
        )

        # read bytes
        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

        # proc
        ## Create Dataset Instance
        ds = PymuDocDataset(pdf_bytes)

        ## inference
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)

            ## pipeline
            pipe_result = infer_result.pipe_ocr_mode(image_writer)

        else:
            infer_result = ds.apply(doc_analyze, ocr=False)

            ## pipeline
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        ### dump markdown
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

        
    else:
        # args
        # pdf_file_name = "small_ocr.pdf"  # replace with the real pdf path
        name_without_suff = pdf_file_name.split(".")[0].split('/')[-1]

        # prepare env
        local_image_dir, local_md_dir = "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/md/images", "/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/md"
        image_dir = str(os.path.basename(local_image_dir))

        os.makedirs(local_image_dir, exist_ok=True)

        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
            local_md_dir
        )

        # read bytes
        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

        # proc
        ## Create Dataset Instance
        ds = PymuDocDataset(pdf_bytes)

        ## inference
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)

            ## pipeline
            pipe_result = infer_result.pipe_ocr_mode(image_writer)

        else:
            infer_result = ds.apply(doc_analyze, ocr=False)

            ## pipeline
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        ### dump markdown
        pipe_result.dump_md(md_writer, f"{local_md_dir}/{name_without_suff}.md", image_dir)
    print(f"完成{pdf_file_name}")



subjects = ['/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/高中政治','/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/高中生物','/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/高中历史',"/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/2000题练习册/高中地理"]
for subj in subjects:
    paths = get_absolute_file_paths(subj,"pdf")
    for path in paths:  
        MinerU(path)