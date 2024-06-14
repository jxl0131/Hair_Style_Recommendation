import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os,shutil
from functions_only_save import make_face_df_save, find_face_shape


df = pd.DataFrame(columns = ['0','1','2','3','4','5','6','7','8','9','10','11',	'12',	'13',	'14',	'15',	'16','17',
                             '18',	'19',	'20',	'21',	'22',	'23',	'24','25',	'26',	'27',	'28',	'29',
                             '30',	'31',	'32',	'33',	'34',	'35',	'36',	'37',	'38',	'39',	'40',	'41',
                             '42',	'43',	'44',	'45',	'46',	'47',	'48',	'49',	'50',	'51',	'52',	'53',
                             '54',	'55',	'56',	'57',	'58',	'59',	'60',	'61',	'62',	'63',	'64',	'65',
                             '66',	'67',	'68',	'69',	'70',	'71',	'72',	'73',	'74',	'75',	'76',	'77',
                             '78',	'79',	'80',	'81',	'82',	'83',	'84',	'85',	'86',	'87',	'88',	'89',
                             '90',	'91',	'92',	'93',	'94',	'95',	'96',	'97',	'98',	'99',	'100',	'101',
                             '102',	'103',	'104',	'105',	'106',	'107',	'108',	'109',	'110',	'111',	'112',	'113',
                             '114',	'115',	'116',	'117',	'118',	'119',	'120',	'121',	'122',	'123',	'124',	'125',
                             '126',	'127',	'128',	'129',	'130',	'131',	'132',	'133',	'134',	'135',	'136',	'137',
                             '138',	'139',	'140',	'141',	'142',	'143','A1','A2','A3','A4','A5','A6','A7','A8','A9',
                            'A10','A11','A12','A13','A14','A15','A16','Width','Height','H_W_Ratio','Jaw_width','J_F_Ratio',
                             'MJ_width','MJ_J_width'])


# test_dir = "/data/home/xinlongji/FaceswapMetrics/faces/reference copy"
test_dir = "/data/home/xinlongji/datasets/FaceShape Dataset/testing_set/Heart"
sv_dir = "/data/home/xinlongji/FaceswapMetrics/faces/refs_diff_shape_hairstyle"


def get_face_shape(img_path):

    file_num = 2035
    
    make_face_df_save(img_path,file_num,df)
    face_shape = find_face_shape(df,file_num)
    return face_shape[0]


def main():
    # 直接预测测试集中的图片的脸型
    nms = os.listdir(test_dir)
    flag = 0
    for nm in nms:

        img_path = os.path.join(test_dir, nm)
        face_shape = get_face_shape(img_path)
        if face_shape == None:
          continue
        sv_sub_dir = os.path.join(sv_dir, face_shape)
        if not os.path.exists(sv_sub_dir):
            os.mkdir(sv_sub_dir)
        # 复制图片到sv_sub_dir
        new_img_path = os.path.join(sv_sub_dir, nm)
        shutil.copy(img_path, new_img_path)

def evaluate():
    # 评估模型在测试集上的准确率
    nms = os.listdir(test_dir)
    correct = 0
    total = 0
    for nm in nms:
        img_path = os.path.join(test_dir, nm)
        face_shape = get_face_shape(img_path)
        if face_shape == None:
          continue
        if face_shape in nm:
            correct += 1
        total += 1
    print("Accuracy: ", correct/total)


        
if __name__ == '__main__':
    """Return face shape."""
    evaluate()
    # main()