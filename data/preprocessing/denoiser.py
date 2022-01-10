'''
Denoising code for news articles
'''

import re

class Denoise(object):
    def __init__(self, **kwargs):
        super(Denoise,self).__init__(**kwargs)

    # Removing the square brackets
    def remove_between_square_brackets(self,text):
        return re.sub('\[[^]]*\]', '', text)
    
    # Removing the square brackets in title
    def remove_between_square_brackets_tit(self,text):
        text = re.sub('\[', '', text)
        text = re.sub('\]', '', text)
        return text
    
    # Removing the round brackets
    def remove_between_round_brackets(self,text):
        return re.sub('\([^])]*\)', '', text)
    
    # Removing the angle brackets
    def remove_between_angle_brackets(self,text):
        return  re.sub('<[^]>]*>', '', text)
    
    # Removing URL's
    def remove_url(self,text):
        text = re.sub(r"http[s]?://(?:[\t\n\r\f\v]|[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", '', text) # http로 시작되는 url
        text = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", '', text) # http로 시작되지 않는 url
        return text
    
    # Removing tags   
    def remove_tag(self,text):
        while '<YNAOBJECT ' in text:
            tag = text[text.find('<YNAOBJECT '):text.find('</YNAOBJECT>')+13]
            text = text.replace(tag, '')
        return text
    
    # Removing Email
    def remove_email(self,text):
        return re.sub(r'[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$','', text)
    
    # Removing reporter
    def remove_reporter(self,text):
        if "기자 =" in text:
            reporter = text[text.find("기자 =")-4:text.find("=")+1]
            text = text.replace(reporter,'')
        return text
    
    # Removing date
    def remove_date(self,text):
        return re.sub('\d+[.]\d+[.]\d+','', text)
        
    # Removing single quotation marks
    def remove_singlequot(self,text):
        return re.sub(r'\'','', text) # 차후년도에 작은따옴표 처리 고민하기
    
    # Removing large whitespace
    def remove_space(self,text):
        text = text.replace("\n", "")
        text = text.replace("\t", "")
        text = text.replace("\r", "")
        text = text.replace("      "," ")
        text = text.replace("    "," ")
        text = text.replace("   "," ")
        text = text.replace("  "," ")
        return text

    # Removing Special Symbol
    def remove_symbol(self,text):
        return re.sub('[<>①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮@$&#☆◇★○●◎◇◆□■△▲▽▼→←←↑↓↔〓◁◀▷▶♤♠♡♥♧♣⊙◈▣◐◑▒▤▥▨▧▦▩♨☏☎☜☞¶†‡↕↗↙↖↘]', '', text)

    # Removing noises
    def remove_etc(self,text):
        return re.sub(r'@\S+','', text)
    def remove_etc1(self,text):
        return re.sub(r'\-\s','', text)
    def remove_etc2(self,text):
        return re.sub(r'\:\s','', text)
    def remove_etc3(self,text):
        return re.sub('일러스트','', text)
    
    # Removing title~reporter name
    def remove_title2reporter(self,text):
        if " = " in text:
            text = text[text.find('=')+2:]
        return text
        
    # denoise_title
    def denoise_title(self, text, square_brack=True, round_brack=True, angle_brack=True,
                      email=True, url=True, noise=True, single=True, symbol=True, space=True):
        if square_brack == True :
            text = self.remove_between_square_brackets_tit(text)
        if round_brack == True :
            text = self.remove_between_round_brackets(text)
        if angle_brack == True :
            text = self.remove_between_angle_brackets(text)
        if email == True :
            text = self.remove_email(text)
        if url == True : 
            text = self.remove_url(text)
        if noise == True :
            text = self.remove_etc(text)
            text = self.remove_etc1(text)
            text = self.remove_etc2(text)
        if single == True:
            text = self.remove_singlequot(text)     
        if symbol == True:
            text = self.remove_symbol(text)
        if space == True:
            text = self.remove_space(text)
        return text
    
    # denoise_subtitle:
    def denoise_subtitle(self, text, square_brack=True, round_brack=True, angle_brack=True,
                         email=True, url=True, noise=True, single=True, symbol=True, space=True):
        try : 
            if square_brack == True :
                text = self.remove_between_square_brackets(text)
            if round_brack == True :
                text = self.remove_between_round_brackets(text)
            if angle_brack == True :
                text = self.remove_between_angle_brackets(text)
            if email == True :
                text = self.remove_email(text)
            if url == True : 
                text = self.remove_url(text)
            if noise == True :
                text = self.remove_etc(text)
                text = self.remove_etc1(text)
                text = self.remove_etc2(text)
            if single == True:
                text = self.remove_singlequot(text)     
            if symbol == True:
                text = self.remove_symbol(text)
            if space == True:
                text = self.remove_space(text)
        except : 
            pass
        return text

    # denoise_body
    def denoise_body(self, text, tag = True, square_brack=True, round_brack=True, angle_brack=True,
                         email=True, url=True, noise=True, single=True, 
                         symbol=True, space=True, title=True):
        if tag == True:
            text = self.remove_tag(text)
        if square_brack == True :
            text = self.remove_between_square_brackets(text)
        if round_brack == True :
            text = self.remove_between_round_brackets(text)
        if angle_brack == True :
            text = self.remove_between_angle_brackets(text)
        if email == True :
            text = self.remove_email(text)
        if url == True : 
            text = self.remove_url(text)
        if noise == True :
            text = self.remove_etc(text)
            text = self.remove_etc1(text)
            text = self.remove_etc2(text)
        if single == True:
            text = self.remove_singlequot(text)     
        if symbol == True:
            text = self.remove_symbol(text)
        if space == True:
            text = self.remove_space(text)
        if title == True:
            text = self.remove_title2reporter(text)
        return text

    # denoise_image_title
    def denoise_image_title(self, text, square_brack=True, round_brack=True, angle_brack=True,
                         email=True, url=True, noise=True, single=True, symbol=True, space=True):
        try :
            if square_brack == True :
                text = self.remove_between_square_brackets(text)
            if round_brack == True :
                text = self.remove_between_round_brackets(text)
            if angle_brack == True :
                text = self.remove_between_angle_brackets(text)
            if email == True :
                text = self.remove_email(text)
            if url == True : 
                text = self.remove_url(text)
            if noise == True :
                text = self.remove_etc(text)
                text = self.remove_etc1(text)
                text = self.remove_etc2(text)
            if single == True:
                text = self.remove_singlequot(text)     
            if symbol == True:
                text = self.remove_symbol(text)
            if space == True:
                text = self.remove_space(text)
        except :
            pass
        return text
    
    # denoise_image_description
    def denoise_image_description(self, text, square_brack=True, round_brack=True, angle_brack=True,
                         email=True, url=True, noise=True, reporter=True, date = True, single=True, symbol=True, space=True):
        try :
            if square_brack == True :
                text = self.remove_between_square_brackets(text)
            if round_brack == True :
                text = self.remove_between_round_brackets(text)
            if angle_brack == True :
                text = self.remove_between_angle_brackets(text)
            if email == True :
                text = self.remove_email(text)
            if url == True : 
                text = self.remove_url(text)
            if noise == True :
                text = self.remove_etc(text)
                text = self.remove_etc1(text)
                text = self.remove_etc2(text)
                text = self.remove_etc3(text)
            if reporter == True:
                text = self.remove_title2reporter(text)
            if date == True:
                text = self.remove_date(text)
            if single == True:
                text = self.remove_singlequot(text)     
            if symbol == True:
                text = self.remove_symbol(text)
            if space == True:
                text = self.remove_space(text)
        except :
            pass
        return text

    # denoise_image_caption
    def denoise_image_caption(self, text):
        text = self.denoise_image_description(text)
        return text