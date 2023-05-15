import openai
import argparse
import yaml
import time
import json
import requests
import copy
import threading

from pprint import pprint
from get_token_ids import get_token_ids_for_task_parsing, get_token_ids_for_choose_model, count_tokens, get_max_context_length

# 参数
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/config.tool.yaml")
parser.add_argument("--mode", type=str, default="test")
args = parser.parse_args()
print(args)

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

LLM         = config["openai"]["model"]
API_KEY     = config["openai"]["api_key"]
API_ENPOINT = config["openai"]["api_endpoint"]

print(config)


# 预备材料 

# few-shot
parse_task_few_shot = open(config["few_shot"]["parse_task"], "r").read()
choose_model_few_shot = open(config["few_shot"]["choose_model"], "r").read()
response_results_few_shot = open(config["few_shot"]["response_results"], "r").read()

# user prompt
parse_task_use_prompt = config["user_prompt"]["parse_task"]
choose_tool_use_prompt = config["user_prompt"]["choose_tool"]
response_results_use_prompt = config["user_prompt"]["response_results"]

# system prompt
parse_task_sys_prompt = config["sys_prompt"]["parse_task"]
choose_tool_sys_prompt = config["sys_prompt"]["choose_tool"]
response_results_sys_prompt = config["sys_prompt"]["response_results"]

def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key +"}}", value.replace('"', "'").replace('\n', ""))
    return text

def replace_content(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key +"}}", value)
    return text


def send_request(data):
    # if use_completion:
    #     data = convert_chat_to_completion(data)
    HEADER = {
        "Authorization": f"Bearer {API_KEY}"
    }
    print()
    response = requests.post(API_ENPOINT, json=data, headers=HEADER, proxies=None)
    if "error" in response.json():
        return response.json()

    return response.json()["choices"][0]["message"]["content"].strip()

def parse_task(chat_context,new_input):

    # 构建 发送 message
    
    ## 1、准备 prompt TODO
    task_sys_prompt = parse_task_sys_prompt
    ## 2、准备 few shot TODO
    demo = parse_task_few_shot

    messages = json.loads(parse_task_few_shot)
    # 插入 消息最前面
    messages.insert(0, {"role": "system", "content": task_sys_prompt})

    start = 0
    # 当开始位置小于等于上下文长度时
    while start <= len(chat_context):
        # 把上下文的开始位置到结束位置的内容赋值给 history
        history = chat_context[start:]
        # 把 input 和 history 替换到 parse_task_prompt 中
        prompt = replace_slot(parse_task_use_prompt, {
            "input": new_input,
            "context": history 
        })
        # 把 prompt 追加到 messages 里
        messages.append({"role": "user", "content": prompt})
        print("messages:",messages)
        # q: 以下代码是什么意思？
        # a: 把 messages 里的内容拼接起来，中间用 <im_start> 分隔
        history_text = "<im_end>\nuser<im_start>".join([m["content"] for m in messages])
        # print("history_text:",history_text)
        # 计算 history_text 里的 token 数量
        num = count_tokens("gpt-3.5-turbo", history_text)
        print()
        print("num:",num)
        print()
        # 如果 token 数量大于 800，就跳出循环
        # 留出 800 个 token 给模型生成
        if get_max_context_length(LLM) - num > 800:
            break
        # 如果 token 数量小于等于 800，就把 messages 里的最后一个元素弹出
        messages.pop()
        start += 2

    # data = {
    #     "model": LLM,
    #     "messages": messages,
    #     "temperature": 0,
    #     "logit_bias": {item: config["logit_bias"]["parse_task"] for item in task_parsing_highlight_ids},
    # }
    data = {
        "model": LLM,
        "messages": messages,
        "temperature": 0,
    }

    return send_request(data)


def fix_dep(tasks):
    # 修复依赖关系
    for task in tasks:
        # 取出 args
        args = task["args"]
        # 依赖 置空
        task["dep"] = []
        # 遍历 args，如果有 <GENERATED>，则将其加入依赖
        # for o in args:
        for k, v in args.items():
            if type(v) == str and "<GENERATED>" in v:
                dep_task_id = int(v.split("-")[1])
                if dep_task_id not in task["dep"]:
                    task["dep"].append(dep_task_id)
        if len(task["dep"]) == 0:
            task["dep"] = [-1]
    return tasks

# 把依赖项 拉 平
def unfold(tasks):
    flag_unfold_task = False
    try:
        # 遍历 task，如果有 args 中有 <GENERATED>，则将其展开
        for task in tasks:
            for key, value in task["args"].items():
                if type(value) == str and "<GENERATED>" in value:
                    generated_items = value.split(",")
                    # 如果生成的项只有一个，直接替换
                    if len(generated_items) > 1:
                    # 如果生成的项有多个，需要将 task 展开
                        flag_unfold_task = True
                        # 遍历items，为每个 item 生成一个新的 task
                        for item in generated_items:
                            # 深拷贝 task
                            new_task = copy.deepcopy(task)
                            # 生成的项的 id
                            dep_task_id = int(item.split("-")[1])
                            # 将生成的项替换为 id
                            new_task["dep"] = [dep_task_id]
                            # 将生成的项替换为具体的值
                            new_task["args"][key] = item
                            tasks.append(new_task)
                        tasks.remove(task)
    except Exception as e:
        print(e)
        print("unfold task failed.")

    if flag_unfold_task:
        print(f"unfold tasks: {tasks}")
        
    return tasks

def onlychat(messages):
    data = {
        "model": LLM,
        "messages": messages
    }
    return send_request(data)
     

def web_search(query):

    # Add your Bing Search V7 subscription key and endpoint to your environment variables.
    subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']
    endpoint = os.environ['BING_SEARCH_V7_ENDPOINT'] + "/bing/v7.0/search"

    # Query term(s) to search for. 
    query = "Microsoft Cognitive Services"

    # Construct a request
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

    # Call the API
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()

        print("\nHeaders:\n")
        print(response.headers)

        print("\nJSON Response:\n")
        pprint(response.json())
    except Exception as ex:
        raise ex


def choose_caculate_api(new_input, command,result):
    import numpy as np
    print(new_input)
    print(command)
    id = command["id"]
    op = command["op"]
    args = command["args"]
    l = []
    out = 0
    for k,v in args.items():
        if isinstance(v, (int, float, complex)):
            l.append(int(v))
        else:
            l = v
    match op:
        case "sum":
            content = np.sum(l)
        case "add":
            content = f'''
keystroke "{l[0]}"
delay 0.5
keystroke "+"
delay 0.5
keystroke "{l[1]}"
delay 0.5
keystroke "="
delay 0.5
                '''
        case "exp":
            content = np.exp(l)
        case "sub":
            content = f'''
keystroke "{l[0]}"
delay 0.5
keystroke "-"
delay 0.5
keystroke "{l[1]}"
delay 0.5
keystroke "="
delay 0.5
                '''
        case "mul":
            content = f'''
keystroke "{l[0]}"
delay 0.5
keystroke "*"
delay 0.5
keystroke "{l[1]}"
delay 0.5
keystroke "="
delay 0.5
                '''
        case "div":
            content = f'''
keystroke "{l[0]}"
delay 0.5
keystroke "/"
delay 0.5
keystroke "{l[1]}"
delay 0.5
keystroke "="
delay 0.5
                '''
        case "power":
            content = f'''
keystroke "{l[0]}"
delay 0.5
keystroke "^"
delay 0.5
keystroke "{l[1]}"
delay 0.5
keystroke "="
delay 0.5
                '''
        case "mod":
            content = f'''
keystroke "{l[0]}"
delay 0.5
keystroke "/"
delay 0.5
keystroke "{l[1]}"
delay 0.5
keystroke "="
delay 0.5
                '''
        case _:
            out = None


    import subprocess

    applescript = '''
    try
        tell application "Calculator"
            activate
            tell application "System Events"
                keystroke "AC"
                {{content}}
            end tell
            tell application "System Events"
                set result to (get value of static text 1 of group 1 of window 1 of process "Calculator")
                return result
            end tell
        end tell
    on error errorMessage
        return errorMessage
    end try
    '''

    applescript = replace_content(applescript, {
            "content": content
    })
    out = subprocess.check_output(['osascript', '-e', applescript])
    out = out.strip().decode('utf-8')
    out = out.replace(',', '')
    if is_numeric(out) and "mod" not in op:
        out = float(out)
        print(type(out))
    else:
        out = int(out)
        print(type(out))

    subprocess.check_output(['osascript', '-e', "quit app \"Calculator\""])


    
    result[id] = {"inference result": out}
    return out

    # 调用 mac 计算器 备注 要界面 "完成"

def is_numeric(string):
    dot_count = string.count('.')
    return dot_count <= 1 and string.replace('.', '').isnumeric()

def collect_result(command, choose, inference_result):
    result = {"task": command}
    result["inference result"] = inference_result
    result["choose api result"] = choose
    print(f"inference result: {inference_result}")
    return result

# 运行任务 --- 选择api，推理，收集结果
def run_task(new_input, command, results):
    print("run task")
    id = command["id"]
    task = command["task"]
    deps = command["dep"]
    args = command["args"]

    # 有依赖的任务，查看 result 获取依赖的结果
    if deps[0] != -1:
        for k,v in args.items():
            if type(v) == str and "<GENERATED>" in v:
                resource_id = int(v.split("-")[1])
                if results[resource_id]["inference result"] is not None:
                    args[k] = results[resource_id]["inference result"]



    if deps[0] != -1:
        dep_tasks = [results[dep] for dep in deps]
    else:
        dep_tasks = []
    
    print(f"Run task: {id} - {task}")
    print("Deps: " + json.dumps(dep_tasks))

    choose_str = ""
    match task:
        case "calculator":
            choose_str = choose_caculate_api(new_input, command,results)
        case "web-serach":
            choose_str = choose_web_search_api(new_input, command,results)
        case _:
            choose_str = None

    # 选择api
    # choose_str = choose_api(new_input, command)
    print(f"result: {choose_str}")
    # try:
    #     choose = json.loads(choose_str)
    #     reason = choose["reason"]
    #     best_model_id = choose["id"]
    #     hosted_on = "local" if best_model_id in all_avaliable_models["local"] else "huggingface"
    # except Exception as e:
    #     logger.warning(f"the response [ {choose_str} ] is not a valid JSON, try to find the model id and reason in the response.")
    #     choose_str = find_json(choose_str)
    #     best_model_id, reason, choose  = get_id_reason(choose_str)
    #     hosted_on = "local" if best_model_id in all_avaliable_models["local"] else "huggingface"
    # # 模型推理
    # inference_result = model_inference(best_model_id, args, hosted_on, command['task'])

    
    # 收集结果
    # results[id] = collect_result(command, choose, inference_result)
    return True

def chat_use_tool(message):
    # 开始时间
    start   = time.time()
    print()

    # 获取上下文
    chat_context = message[:-1]
    # 获取新的输入cd 
    new_input   = message[-1]["content"]

    # Step1: 解析任务
    task_json_str = parse_task(chat_context,new_input)

    task_json_str = task_json_str.strip()
    print( "reply json : ", task_json_str)

    ## 处理 json
    try:
        tasks = json.loads(task_json_str)
    except Exception as e:
        print(e)
        print("task_str is not json , justchat")
        response = onlychat(messages)
        return {"message": response}

    if task_json_str == "[]":  # using LLM response for empty task
        print("task == [] , justchat")
        response = onlychat(messages)
        return {"message": response}

    tasks = unfold(tasks)
    tasks = fix_dep(tasks)
    print()
    print("return_planning : ",tasks)


    results = {}
    threads = []
    tasks = tasks[:]
    d = dict()
    retry = 0
    while True:
        num_thread = len(threads)
        print("num_thread : ",num_thread)
        for task in tasks:
            for dep_id in task["dep"]:
                if dep_id >= task["id"]:
                    task["dep"] = [-1]
                    break
            dep = task["dep"]
            # 如果依赖的第一项是-1 或者 依赖的所有项都在d中，那么就可以运行这个task
            if dep[0] == -1 or len(list(set(dep).intersection(d.keys()))) == len(dep):
                # 删除已经运行的task
                tasks.remove(task)
                # 运行task
                thread = threading.Thread(target=run_task, args=(new_input, task, d))
                thread.start()
                threads.append(thread)
        # 如果没有新的线程启动，那么就等待0.5秒
        if num_thread == len(threads):
            time.sleep(0.5)
            retry += 1
        if retry > 160:
            print("retry > 160")
            break
        if len(tasks) == 0:
            break
    # 等待所有线程结束
    for thread in threads:
        thread.join()
    # 复制d的内容到results
    results = d.copy()
    print(results)


    # # 4 生成response
    # response = response_results(new_input, results).strip()



    end     = time.time()

    print("耗时：",end-start)

# text = "2的平方是多少"
# text = "10的平方除5乘2是多少"
text = "10的3次方除5加3是多少"


messages = [{"role": "user", "content": text}]
r = chat_use_tool(messages)
print(r)