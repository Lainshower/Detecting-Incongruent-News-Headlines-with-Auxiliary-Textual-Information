'''
denoising & caption concatenation
'''

import pandas as pd
from denoiser import Denoise
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--p", "--preprocess", dest="preprocess", action="store_true")     
parser.add_argument("--m", "--merge", dest="merge", action="store_true")      
parser.add_argument("--data", "--data_name", dest="data_name", type=str)
parser.add_argument("--dir", "--dir_name", dest="dir_name", type=str)     
args = parser.parse_args()

data_path = os.path.join(args.dir_name,args.data_name)
data = pd.read_csv(data_path+'.csv')
print(args.preprocess)
print(args.merge)
print(args.data_name)
print(args.dir_name)
print(data_path)

def delete(text):
    if type(text) == float : 
        text = ""
    return text

if args.preprocess:
    denoiser = Denoise()
    data["title"] = data["title"].apply(denoiser.denoise_title)
    data["subtitle"] = data["subtitle"].apply(denoiser.denoise_subtitle)
    data["body"] = data["body"].apply(denoiser.denoise_body)
    data["image_title1"] = data["image_title1"].apply(delete)
    data["image_title2"] = data["image_title2"].apply(delete)
    data["image_title3"] = data["image_title3"].apply(delete)
    data["image_caption1"] = data["image_caption1"].apply(delete)
    data["image_caption2"] = data["image_caption2"].apply(delete)
    data["image_caption3"] = data["image_caption3"].apply(delete)
    data["image_description1"] = data["image_description1"].apply(delete)
    data["image_description2"] = data["image_description2"].apply(delete)
    data["image_description3"] = data["image_description3"].apply(delete)
    data["image_title1"] = data["image_title1"].apply(denoiser.denoise_image_title)
    data["image_title2"] = data["image_title2"].apply(denoiser.denoise_image_title)
    data["image_title3"] = data["image_title3"].apply(denoiser.denoise_image_title)
    data["image_caption1"] = data["image_caption1"].apply(denoiser.denoise_image_caption)
    data["image_caption2"] = data["image_caption2"].apply(denoiser.denoise_image_caption)
    data["image_caption3"] = data["image_caption3"].apply(denoiser.denoise_image_caption)
    data["image_description1"] = data["image_description1"].apply(denoiser.denoise_image_description)
    data["image_description2"] = data["image_description2"].apply(denoiser.denoise_image_description)
    data["image_description3"] = data["image_description3"].apply(denoiser.denoise_image_description)

if args.merge:
    # concatenate multiple captions
    data["caption"] = data["image_title1"]+data["image_caption1"]+data["image_title2"]+data["image_caption2"]+data["image_title3"]+data["image_caption3"]
    del data["image_title1"],data["image_caption1"],data["image_title2"],data["image_caption2"],data["image_title3"],data["image_caption3"]
    del data["image_description1"],data["image_description2"],data["image_description3"]

# Write csv file
if args.preprocess:
    if args.merge:
        data.to_csv(data_path+"_caption_concat.csv", index=False, encoding='utf-8-sig')
    else :
        data.to_csv(data_path+"_preprocess.csv", index=False, encoding='utf-8-sig')
else:
   if args.merge:
        data.to_csv(data_path+"_merge.csv", index=False, encoding='utf-8-sig')
    else :
        continue