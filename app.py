import gradio as gr
import torch
import requests
import os
from torchvision import transforms

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()
# response = requests.get("https://git.io/JJkYN")

# 获取当前文件的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
with open(f'{current_directory}/labels.txt', 'r', encoding='utf-8') as file:
  # 读取文件内容
  content = file.read()
  labels = content.split("\n")

def predict(inp):
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}    
  return confidences

demo = gr.Interface(fn=predict, 
             inputs=gr.inputs.Image(type="pil"),
             outputs=gr.outputs.Label(num_top_classes=3),
             examples=[["cheetah.jpg"]],
             )
             
demo.launch()
