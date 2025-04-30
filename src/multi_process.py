import os
import json
import multiprocessing
import threading
import traceback


def multiprocess_thread_process(
    input_path: str,
    output_path: str,
    process_item,
    num_processes: int = None,
    threads_per_process: int = 4
):
    """
    并行处理 JSON 列表文件，支持续跑，仅输出为 JSONL 格式（每行一个 JSON 对象）。
    并且在运行时打印实际启动的进程数与每个进程使用的线程并发数。

    参数:
        input_path: 输入 JSON 文件路径，文件为列表，每元素为 dict，需包含 'id' 字段。
        output_path: 输出 JSONL 文件路径，每个记录独占一行。
        process_item: 处理函数，接收 dict item，返回可 JSON 序列化的记录 dict，必须包含相同 'id'.
        num_processes: 使用的进程数，默认为 CPU 核心数。
        threads_per_process: 每个进程的线程数。
    """
    # 1. 读取输入数据
    try:
        with open(input_path, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
    except Exception as e:
        print(f"Error reading input file '{input_path}': {e}")
        return

    # 2. 检查已完成 ID
    processed_ids = set()
    resume = False
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as fout:
                for line in fout:
                    try:
                        rec = json.loads(line)
                        pid = rec.get('id')
                        if pid is not None:
                            processed_ids.add(pid)
                    except json.JSONDecodeError:
                        continue
            resume = bool(processed_ids)
        except Exception:
            pass

    # 3. 过滤待处理数据
    todo = [item for item in data if item.get('id') not in processed_ids]
    if not todo:
        print("No new items to process. Exiting.")
        return

    # 4. 初始化输出文件（首次或续跑）
    if not resume:
        try:
            open(output_path, 'w', encoding='utf-8').close()
        except Exception as e:
            print(f"Error initializing output file '{output_path}': {e}")
            return

    # 5. 确定进程数
    total = len(todo)
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    chunk_size = (total + num_processes - 1) // num_processes
    chunks = [todo[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes) if todo[i*chunk_size:(i+1)*chunk_size]]
    print(f"[Main] Detected CPU cores: {multiprocessing.cpu_count()}, launching {len(chunks)} processes.")

    # 6. 进程间同步原语
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    # 7. 线程工作函数
    def worker_thread(item):
        try:
            record = process_item(item)
            if record is None:
                raise ValueError("process_item must be not None.")
            pid = record.get('id')
            if pid is None:
                raise ValueError("process_item must return dict with 'id'.")
            text = json.dumps(record, ensure_ascii=False)
            with lock:
                with open(output_path, 'a', encoding='utf-8') as fout:
                    fout.write(text + '\n')
        except Exception:
            err = traceback.format_exc()
            print(f"Error processing item {item.get('id')}: {err}")

    # 8. 进程工作函数
    def process_worker(chunk):
        proc_name = multiprocessing.current_process().name
        print(f"[{proc_name}] Starting. Will process {len(chunk)} items with up to {threads_per_process} threads concurrently.")
        threads = []
        for item in chunk:
            t = threading.Thread(target=worker_thread, args=(item,))
            t.start()
            threads.append(t)
            if len(threads) >= threads_per_process:
                for th in threads:
                    th.join()
                threads = []
        # 等待剩余线程
        for th in threads:
            th.join()
        print(f"[{proc_name}] Completed. Spawned total threads: {len(chunk)}.")

    # 9. 启动多进程
    processes = []
    try:
        for idx, chunk in enumerate(chunks):
            p = multiprocessing.Process(target=process_worker, args=(chunk,), name=f"Proc-{idx}")
            p.start()
            processes.append(p)
        print(f"[Main] Started {len(processes)} processes.")
        for p in processes:
            p.join()
        print(f"[Main] All processes have finished.")
    except Exception as e:
        print(f"Error in multiprocessing execution: {e}")
        traceback.print_exc()



from utils import *

# 示例用法
if __name__ == '__main__':
    def example_process_item(item):
        try:
            prompt = '''为给定的文本生成一个qa对，返回json格式如下：
            {
                "qa_pair":""...
            }
            
            文本：{content}
            
            输出：'''
            # 处理单条记录
            content = item.get('context', '')
            qa = call_Hkust_api(prompt.replace("{content}",content))
            # qa = ""
            if qa.strip() == "":
                return None
            return {'id': item.get('id'), 'qa': qa}
        except Exception as e:
            return None

    multiprocess_thread_process(
        input_path='/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/dataset/sft/gaokao/data/规范data/final_data/test.json',
        output_path='/hpc2hdd/home/fye374/ZWZ_Other/quizmanus/src/output.jsonl',
        process_item=example_process_item,
        threads_per_process=4
    )
