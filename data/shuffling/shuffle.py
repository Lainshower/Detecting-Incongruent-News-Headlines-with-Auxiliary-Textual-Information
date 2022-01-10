'''
Shuffling code for generating incongruent headline news
'''

from tqdm import tqdm
import pandas as pd
import numpy as np
np.random.seed(0)

# if you want T:f=7:3, ratio should be 0.3
ratio = 0.5

file_path = ''
result_csv_name = ''

data = pd.read_csv(file_path)
data = data.sample(frac=1, random_state=486)
data.reset_index(drop=True, inplace=True)
print("previous data size : ",len(data))

# remove data whose title has NAN value
data = data[-data['title'].isna()]

# excluded news articles that can be seen as incongruent news
trivial_list = []
not_using = ['[인사]', '[게시판]', '[그래픽]', '[부고]', '[표]', '[영상]', '카드뉴스', '[풀영상]', '[포토무비]', '[블랙박스]']
for i in range(len(data)):
    title = data['title'].iloc[i]
    if any(item in title for item in not_using):
        trivial_list.append(i)
print("trivial titles size : ", len(trivial_list))

data.drop(trivial_list, inplace=True)
data.reset_index(drop=True, inplace=True)
print("data size after remove trivial titles : ", len(data))

not_shuffled_data = data.iloc[int(len(data)*ratio):]
data = data.iloc[:int(len(data)*ratio)]

print("True seed :", len(not_shuffled_data))
print("False seed :", len(data))

new_title = list(not_shuffled_data['title'])
new_subtitle = list(not_shuffled_data['subtitle'])
new_body = list(not_shuffled_data['body'])

new_image_title1 = list(not_shuffled_data['image_title1'])
new_image_description1 = list(not_shuffled_data['image_description1'])
new_image_caption1 = list(not_shuffled_data['image_caption1'])

new_image_title2 = list(not_shuffled_data['image_title2'])
new_image_description2 = list(not_shuffled_data['image_description2'])
new_image_caption2 = list(not_shuffled_data['image_caption2'])

new_image_title3 = list(not_shuffled_data['image_title3'])
new_image_description3 = list(not_shuffled_data['image_description3'])
new_image_caption3 = list(not_shuffled_data['image_caption3'])

new_category = list(not_shuffled_data['category'])
new_classcode = list(not_shuffled_data['classcode'])

new_headline_id = list(not_shuffled_data['id'])
new_body_id = list(not_shuffled_data['id'])

new_label = [0] * len(new_title)

classcode_unique_list = data['classcode'].unique()
print("Unique classcode :", classcode_unique_list)

incongruent_title_list, incongruent_subtitle_list, incongruent_body_list = [], [], []
incongruent_image_title1_list, incongruent_image_title2_list, incongruent_image_title3_list = [], [], []
incongruent_image_caption1_list, incongruent_image_caption2_list, incongruent_image_caption3_list = [], [], []
incongruent_image_description1_list, incongruent_image_description2_list, incongruent_image_description3_list = [], [], []
incongruent_category_list, incongruent_classcode_list, incongruent_headline_id_list, incongruent_body_id_list = [], [], [], []
incongruent_label = []

classcode_count = []
for classcode in classcode_unique_list:
    div_data = data.loc[data['classcode']==classcode]
    classcode_count.append(len(div_data))
    flag_list = [0] * len(div_data)
    
    title_list, subtitle_list, body_list = [], [], []
    image_title1_list, image_description1_list, image_caption1_list = [], [], []
    image_title2_list, image_description2_list, image_caption2_list = [], [], []
    image_title3_list, image_description3_list, image_caption3_list = [], [], []
    categories_list, classcode_list = [],[]
    headline_id_list, body_id_list = [], []

    if len(flag_list) > 5: # do not use when classcode count is under 5
        for i in tqdm(range(len(flag_list))):
            # if data was used before, then continue
            if flag_list[i] == 1:
                continue
            flag_list[i] = 1 # flag on 

            # random sample one of the news with same classcode
            x = np.random.randint(0, len(flag_list)-1)
            
            count = 0
            switch = False

            # sample until data that is not used before with different send_date
            while (flag_list[x] != 0 or (str(div_data['send_date'].iloc[i]) == str(div_data['send_date'].iloc[x]))):
                x = random.randint(0, len(flag_list)-1)
                count += 1
                if count > 10:
                    switch = True # if we can't find the proper sample, switch on
                    break

            if switch == True:
                flag_list[i] = 0 # flag off
                continue # pass the shuffling process
        
            title_list.append(div_data['title'].iloc[i])
            subtitle_list.append(div_data['subtitle'].iloc[i])
            headline_id_list.append(div_data['id'].iloc[i])
            body_list.append(div_data['body'].iloc[x])
            image_title1_list.append(div_data['image_title1'].iloc[x])
            image_description1_list.append(div_data['image_description1'].iloc[x])
            image_caption1_list.append(div_data['image_caption1'].iloc[x])
            image_title2_list.append(div_data['image_title2'].iloc[x])
            image_description2_list.append(div_data['image_description2'].iloc[x])
            image_caption2_list.append(div_data['image_caption2'].iloc[x])
            image_title3_list.append(div_data['image_title3'].iloc[x])
            image_description3_list.append(div_data['image_description3'].iloc[x])
            image_caption3_list.append(div_data['image_caption3'].iloc[x])
            body_id_list.append(div_data['id'].iloc[x])
            categories_list.append(div_data['category'].iloc[i])
            classcode_list.append(div_data['classcode'].iloc[i])

            flag_list[x] = 1 # flag on
    
    incongruent_title_list = incongruent_title_list + title_list
    incongruent_subtitle_list = incongruent_subtitle_list + subtitle_list
    incongruent_body_list = incongruent_body_list + body_list

    incongruent_image_title1_list = incongruent_image_title1_list + image_title1_list
    incongruent_image_description1_list = incongruent_image_description1_list + image_description1_list
    incongruent_image_caption1_list = incongruent_image_caption1_list + image_caption1_list

    incongruent_image_title2_list = incongruent_image_title2_list + image_title2_list
    incongruent_image_description2_list = incongruent_image_description2_list + image_description2_list
    incongruent_image_caption2_list = incongruent_image_caption2_list + image_caption2_list

    incongruent_image_title3_list = incongruent_image_title2_list + image_title3_list
    incongruent_image_description3_list = incongruent_image_description3_list + image_description3_list
    incongruent_image_caption3_list = incongruent_image_caption3_list + image_caption3_list

    incongruent_headline_id_list = incongruent_headline_id_list + headline_id_list
    incongruent_body_id_list  = incongruent_body_id_list + body_id_list

    incongruent_category_list = incongruent_category_list + categories_list
    incongruent_classcode_list = incongruent_classcode_list + classcode_list

    incongruent_label = incongruent_label + ([1] * len(title_list))
            
incongruent_count = len(incongruent_label) 
print("Incongruent Data : ", incongruent_count)

congruent_data = {'title' : new_title, 'subtitle' : new_subtitle, 'body' : new_body, 'image_title1' : new_image_title1, 'image_description1': new_image_description1, 'image_caption1' : new_image_caption1, 'image_title2' : new_image_title2, 'image_description2': new_image_description2, 'image_caption2' : new_image_caption2, 'image_title3' : new_image_title3, 'image_description3': new_image_description3, 'image_caption3' : new_image_caption3, 'headline_id' : new_headline_id, 'body_id': new_body_id, 'category' : new_category, 'classcode': new_classcode, 'label' : new_label}
incongruent_data = {'title' : incongruent_title_list, 'subtitle' : incongruent_subtitle_list, 'body' : incongruent_body_list, 'image_title1' : incongruent_image_title1_list, 'image_description1': incongruent_image_description1_list, 'image_caption1' : incongruent_image_caption1_list, 'image_title2' : incongruent_image_title2_list, 'image_description2': incongruent_image_description2_list, 'image_caption2' : incongruent_image_caption3_list, 'image_title3' : incongruent_image_title3_list, 'image_description3': incongruent_image_description3_list, 'image_caption3' : incongruent_image_caption3_list,
 'headline_id' : incongruent_headline_id_list, 'body_id': incongruent_body_id_list, 'category' : incongruent_category_list, 'classcode': incongruent_classcode_list, 'label' : incongruent_label}

congruent_data = pd.DataFrame(congruent_data)
incongruent_data = pd.DataFrame(incongruent_data)
congruent_data = congruent_data.iloc[0:len(incongruent_data)]

incongruent_dataset = pd.concat([congruent_data,incongruent_data])
incongruent_data.reset_index(drop=True,inplace=True)

incongruent_dataset.to_csv(result_csv_name, index=False, encoding='utf-8-sig')

print("Congruent Data : ",incongruent_dataset[incongruent_dataset['label']==0].shape[0])
print("Incongruent Data : ",incongruent_dataset[incongruent_dataset['label']==1].shape[0])