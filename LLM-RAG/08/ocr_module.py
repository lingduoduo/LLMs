import yaml
import os
import json
from paddleocr import PaddleOCR

class OCREngine:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["ocr"]
        
        # 初始化PaddleOCR实例
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=self.config["use_doc_orientation_classify"],
            use_doc_unwarping=self.config["use_doc_unwarping"],
            use_textline_orientation=self.config["use_textline_orientation"],
            lang=self.config["language"],
        )
    
    def extract_text(self, file_path):
        # 执行OCR推理
        result = self.ocr.predict(file_path)
        for res in result:
            res.save_to_json("output")
        
        # 读取并显示文件内容
        with open("output/example_res.json", "r", encoding="utf-8") as f:
            jsondata =  f.read()
        return json.loads(jsondata)
        

if __name__ == "__main__":
    ocr_engine = OCREngine()
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(__file__)
    # 拼接图像文件路径
    image_path = os.path.join(base_dir, "example.png") 
    extracted_text = ocr_engine.extract_text(image_path)
    print(extracted_text.get("rec_texts", []))