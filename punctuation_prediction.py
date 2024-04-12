import torch
from transformers import AutoTokenizer
from helper import *
from configuration import CONFIG
CONFIG= CONFIG()

tokenizer= AutoTokenizer.from_pretrained(CONFIG.model_name)
model=NER_MODEL(model_name= CONFIG.model_name)
state_dict= torch.load(CONFIG.weight_path, map_location= CONFIG.device)
print(state_dict)

# model.load_state_dict(torch.load())
# model.to(CONFIG.device)
# model.eval()

# text= "একাধারে কবি সাহিত্যিক সংগীতজ্ঞ সাংবাদিক সম্পাদক রাজনীতিবিদ এবং সৈনিক হিসেবে অন্যায় ও অবিচারের বিরুদ্ধে নজরুল সর্বদাই ছিলেন সোচ্চার"
# result= pun_inference_fn(text, model, tokenizer, CONFIG)