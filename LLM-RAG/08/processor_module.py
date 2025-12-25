import re
import yaml
import os
import json

class TextProcessor:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["dify"]["process_rules"]
    
    def preprocess(self, text):
        if self.config["pre_processing"][0]["enabled"]:
            text = re.sub(r"\s+", " ", text)
        if len(self.config["pre_processing"]) > 1 and self.config["pre_processing"][1]["enabled"]:
            text = re.sub(r"https?://\S+|\w+@\w+\.\w+", "", text)
        return text
    
    def segment(self, text):
        separator = self.config["segmentation"]["separator"]
        return re.split(separator, text)

    def preprocess_json(self, json_data):
        # 从 json 数据中提取 rec_texts 列表，若不存在则使用空列表
        rec_texts = json_data.get('rec_texts', [])
        processed_texts = []
        for text in rec_texts:
            processed_text = self.preprocess(str(text))
            processed_texts.append(processed_text)
        return processed_texts

    def process(self, input_data):
        """
        统一处理入口，支持字符串和 JSON 数据输入
        :param input_data: 输入数据，可以是字符串或包含 rec_texts 的 JSON 数据
        :return: 分段后的结果列表
        """
        if isinstance(input_data, dict):
            # 处理 JSON 数据
            cleaned_texts = self.preprocess_json(input_data)
            combined_text = ' '.join(cleaned_texts)
            if combined_text:
                return self.segment(combined_text)
            print("未找到可处理的文本。")
            return []
        elif isinstance(input_data, str):
            # 处理普通字符串
            cleaned_text = self.preprocess(input_data)
            return self.segment(cleaned_text)
        else:
            print("不支持的输入类型，请输入字符串或 JSON 数据。")
            return []

if __name__ == "__main__":
    processor = TextProcessor()
    with open("output/example_res.json", "r", encoding="utf-8") as f:
        jsondata = f.read()
    raw_text = json.loads(jsondata)
    print("处理 JSON 数据结果：")
    print(processor.process(raw_text))
    
    # 示例：处理普通字符串
    sample_text = "这是一个测试文本。需要进行分段。"
    print("处理普通字符串结果：")
    print(processor.process(sample_text))