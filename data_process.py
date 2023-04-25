import numpy as np
import csv
import os
import cv2
import random
def img_clip():
    path = '/data/dataset/overboard_detection/'
    save_path = '/data/dataset/overboard_classify/all'
    train_path = '/data/dataset/overboard_classify/train'
    valid_path = '/data/dataset/overboard_classify/valid'
    test_path = '/data/dataset/overboard_classify/test'
    label_path = '/data/dataset/overboard_classify'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    file_list = os.listdir(path)
    tmp = np.zeros((512,512,3))
    num = 0
    label_out = []
    for file in file_list:
        if file[-3:] == 'png' and file != '2962.png':
            img_path = os.path.join(path,file)
            lable_path = os.path.join(path,file[:-4]+'.txt')
            img = cv2.imread(img_path)
            print(file)
            label = np.loadtxt(lable_path)
            label = np.reshape(label,(-1,5))

            shape = np.shape(img)
            num_h = int(shape[0] / 512)
            num_w = int(shape[1] / 512)
            for i in range(num_h):
                for j in range(num_w):      
                    print(file)         
                    tmp[:,:,:] = img[512*i:512*i+512,512*j:512*j+512,:]
                    flag_person = 0
                    flag_ship = 0
                    for row in label:
                        if int(row[1]*shape[1]) >= 512*j and int(row[1]*shape[1]) < 512*j+512 \
                        and int(row[2]*shape[0]) >= 512*i and int(row[2]*shape[0]) < 512*i+512:
                            if int(row[0]) == 1 or int(row[0]) == 2:
                                flag_person = 1
                            else:
                                flag_ship = 1
                    label_out.append([num,flag_person,flag_ship])
                    print(label_out[num])
                    cv2.imwrite(os.path.join(save_path,'%d.png'%(num)),tmp)
                    num += 1
    np.savetxt(os.path.join(label_path,'label.csv'),label_out,fmt='%d',delimiter=',')

    for row in label_out:
        print(row)
def split_human_ship():
    path = '/data/dataset/overboard_classify/all'
    
    label_path = '/data/dataset/overboard_classify/label.csv'
    human_path = '/data/dataset/overboard_classify/human'
    ship_path = '/data/dataset/overboard_classify/ship'
    background_path = '/data/dataset/overboard_classify/background'
    if not os.path.exists(human_path):
        os.makedirs(human_path)
    if not os.path.exists(ship_path):
        os.makedirs(ship_path)
    if not os.path.exists(background_path):
        os.makedirs(background_path)
    file_list = os.listdir(path)
    label = np.loadtxt(label_path,delimiter=',')
    num_human = 0
    num_ship = 0
    num_back = 0
    for row in label:
        print(row)
        img_path = os.path.join(path,'%d.png'%(int(row[0])))
        img = cv2.imread(img_path)
        if int(row[1]) == 1:
            num_human += 1
            save_path = os.path.join(human_path,'%d.png'%(int(row[0])))
            cv2.imwrite(save_path,img)
        if int(row[2]) == 1:
            num_ship += 1
            save_path = os.path.join(ship_path,'%d.png'%(int(row[0])))
            cv2.imwrite(save_path,img)
        if int(row[1]) == 0 and int(row[2]) == 0:
            num_back += 1
            save_path = os.path.join(background_path,'%d.png'%(int(row[0])))
            cv2.imwrite(save_path,img)
    print('length:',len(label),'back:',num_back,'human:',num_human,'ship:',num_ship)

# 数据集情况  length: 38992 back: 35424 human: 1695 ship: 2328
# 数据集划分  train:        back: 4000  human: 1500 ship: 2000
# 数据集划分  valid:        back: 200   human: 95   ship: 100
# 数据集划分  test:         back: 200   human: 100  ship: 200
def split_train_valid_test():
    human_path = '/data/dataset/overboard_classify/human'
    ship_path = '/data/dataset/overboard_classify/ship'
    background_path = '/data/dataset/overboard_classify/background'

    train_path = '/data/dataset/overboard_classify/train1'
    valid_path = '/data/dataset/overboard_classify/valid1'
    test_path = '/data/dataset/overboard_classify/test1'

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    human_list = os.listdir(human_path)
    ship_list = os.listdir(ship_path)
    background_list = os.listdir(background_path)
    len_human = len(human_list)
    len_ship = len(ship_list)
    len_back = len(background_list)
    randomIndex1=random.sample(range(len_human),len_human)
    randomIndex2=random.sample(range(len_ship),len_ship)
    randomIndex3=random.sample(range(len_back),2400)
    num = 0
    for idx in randomIndex1:
        print(human_list[idx])
        img = cv2.imread(os.path.join(human_path,human_list[idx]))
        if num < 195:
            cv2.imwrite(os.path.join(valid_path,human_list[idx]),img)
            num += 1
        elif num < 395:
            cv2.imwrite(os.path.join(test_path,human_list[idx]),img)
            num += 1
        else:
            cv2.imwrite(os.path.join(train_path,human_list[idx]),img)
            num += 1
    num = 0
    for idx in randomIndex2:
        print(ship_list[idx])
        img = cv2.imread(os.path.join(ship_path,ship_list[idx]))
        if num < 200:
            cv2.imwrite(os.path.join(valid_path,ship_list[idx]),img)
            num += 1
        elif num < 400:
            cv2.imwrite(os.path.join(test_path,ship_list[idx]),img)
            num += 1
        else:
            cv2.imwrite(os.path.join(train_path,ship_list[idx]),img)
            num += 1
    num = 0
    for idx in randomIndex3:
        print(background_list[idx])
        img = cv2.imread(os.path.join(background_path,background_list[idx]))
        if num < 200:
            cv2.imwrite(os.path.join(valid_path,background_list[idx]),img)
            num += 1
        elif num < 200:
            cv2.imwrite(os.path.join(test_path,background_list[idx]),img)
            num += 1
        else:
            cv2.imwrite(os.path.join(train_path,background_list[idx]),img)
            num += 1
if __name__ == '__main__':
    # img_clip()
    # split_human_ship()   
    split_train_valid_test()