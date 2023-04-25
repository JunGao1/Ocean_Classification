from symbol import test_nocond
import torch
import cv2
import argparse
import os
from model import Net, Net_normal
from dataset import Ocean_data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from torchvision import transforms
from PIL import Image
import numpy as np
import time

def get_argparse():
    parser = argparse.ArgumentParser('OceanSearch')
    
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size.")
    parser.add_argument('--epochs',type=int,default=100,help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument('--save_path',type=str,default='./checkpoint/human',help='path of model to save.')
    parser.add_argument('--load_path',type=str,default='',help='model path to test or continue train if not None.')
    parser.add_argument('--phase',type=str,default='train',choices=['train','test'],help='runer phase.')
    parser.add_argument('--data_train',type=str,default='/data/dataset/overboard_classify/train',help='path of training dataset.')
    parser.add_argument('--data_label',type=str,default='/data/dataset/overboard_classify/label.csv',help='path of label dataset.')
    parser.add_argument('--data_valid',type=str,default='/data/dataset/overboard_classify/valid',help='path of validation dataset.')
    parser.add_argument('--data_test',type=str,default='/data/dataset/overboard_classify/test',help='path of test dataset.')
    parser.add_argument('--gpu_id',type=str,default='4',help='devices setting.-1 means use cpu')
    parser.add_argument('--start_epoch',type=int,default=0,help='number of epoch to continue')
    parser.add_argument('--flag',type=str,default='human',help='train target')
    parser.add_argument('--net',type=str,default='Net',help='train target')

    return parser.parse_args()

class Run(object):
    def __init__(self,args):
        self.args = args
        if self.args.gpu_id == '-1':
            self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:'+self.args.gpu_id)
            else:
                print('error! No cuda device.')
        if self.args.load_path:
            if self.args.gpu_id == '-1':
                self.model.load_state_dict(torch.load(self.args.load_path,map_location='cpu'))
        else:
            if self.args.net == 'Net':
                self.model = Net()
            else:
                 self.model = Net_normal()
            self.model = self.model.to(self.device)
        self.transform = transforms.Compose(
            [
                # transforms.Resize(size = (512,512)),#尺寸规范
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),   #转化为tensor
                # transforms.Normalize((0.5), (0.5)),
            ])
        
    def train(self):
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        print('parameters_count:',sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        dataset_train = Ocean_data(self.args.data_train,self.args.data_label,flag=self.args.flag)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=self.args.batch_size,shuffle=True, num_workers=8)
        step_count = len(dataloader_train)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        valid_results = []
        for epoch in range(self.args.start_epoch,self.args.epochs):
            # losses = []
            accuracy = []
            loop = tqdm((dataloader_train), total = len(dataloader_train))
            for sample in loop:
                img, label = sample
                img = img.to(self.device)
                
                label = label.to(self.device)
                self.model.train()
                self.model.zero_grad()
                optimizer.zero_grad()
                out = self.model(img)
                # print(out.shape,label.shape)
                loss = criterion(out,label)
                loss.backward()
                _,predictions = out.max(1)
                num_correct = (predictions == label).sum()
                running_train_acc = float(num_correct) / float(img.shape[0])
                accuracy.append(running_train_acc)
                optimizer.step()
                loop.set_description(f'Epoch [{epoch}/{self.args.epochs}]')
                loop.set_postfix(loss = loss.item(),acc = running_train_acc)
            
            acc = self.valid(epoch)
            valid_results.append([epoch,acc])
            print('========EPOCH:%d==ACC_VALID:%.3f======='%(epoch,acc))
            model_name = 'OceanSearch_%d.pkl' % (epoch)
            torch.save(self.model.state_dict(), os.path.join(self.args.save_path,model_name))
        log_name = self.args.flag + '_valid_results.csv'
        np.savetxt(log_name,valid_results,fmt='%d,%.3f',delimiter=',')

    def valid(self,epoch):
        dataset_valid = Ocean_data(self.args.data_valid,self.args.data_label,self.args.flag)
        dataloader_valid = DataLoader(dataset_valid,batch_size=32,shuffle=False,num_workers=8)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for step,sample in enumerate(dataloader_valid):
                img,label = sample
                img = img.to(self.device)
                label = label.to(self.device)
                
                out = self.model(img)
                _,predictions = out.max(1)
                num_correct = (predictions == label).sum()
                correct += float(num_correct) / float(img.shape[0])
                total += 1
        # print('Epoch:%d Accuracy: %.3f %%' % (epoch,100.0*correct/total))
        return correct/total
    
    def test(self,epoch):
        ckp_path = os.path.join(self.args.save_path,'OceanSearch_%d.pkl'%(epoch))
        if self.args.gpu_id == '-1':
            state = torch.load(ckp_path,map_location='cpu')
        else:
            state = torch.load(ckp_path)
        self.model.load_state_dict(state)
        self.model.eval()
        path = self.args.data_test
        list_dir = os.listdir(self.args.data_test)
        label_all = np.loadtxt(self.args.data_label,delimiter=',')
        acc = 0
        avg = 0
        # test_results = []
        torch.cuda.synchronize()
        with torch.no_grad():
            for file in list_dir:
                # time1 = time.perf_counter()
                file_path = os.path.join(path,file)
                if self.args.flag == 'human':
                    label = label_all[int(file[0:-4])][1]
                elif self.args.flag == 'ship':
                    label = label_all[int(file[0:-4])][2]
                else:
                    if label_all[int(file[0:-4])][1] + \
                        label_all[int(file[0:-4])][2]:
                        label = 1
                    else:
                        label = 0
                label = torch.as_tensor(label, dtype=torch.long)
                # label = torch.LongTensor(label)
                label = label.to(self.device)
                image = Image.open(file_path)
                image = self.transform(image)
                image = torch.unsqueeze(image,dim=0)
                image = image.to(self.device)
                time1 = time.perf_counter()
                out = self.model(image)
                time2 = time.perf_counter()
                _,predictions = out.max(1)
                if predictions == label:
                    acc += 1
                # time2 = time.perf_counter()
                avg += 1000*(time2-time1)
        acc = acc/len(list_dir)
        avg = avg/len(list_dir)
        print('=======EPOCH:%d============ACC:%.3f=======TIME:%.3fms========='%(epoch,acc,avg))
        return acc



if __name__ == '__main__':
    args = get_argparse()
    runner = Run(args)
    if args.phase == 'train':
        runner.train()
    if args.phase == 'test':
        max = 0
        max_acc = 0
        test_results = []
        for i in range(0,100):
           acc = runner.test(i)
           test_results.append([i,round(acc,3)])
           if acc > max_acc:
                max_acc = acc
                max = i
        
        print('max_epoch:%d,max_acc:%.3f'%(max,max_acc))
        # test_results.append(['max_epoch','max_acc'])
        test_results.append([max,round(max_acc,3)])
        # test_results = np.array(test_results)
        log_name = args.flag + '_test_results.csv'
        np.savetxt(log_name,test_results,fmt='%d,%.3f',delimiter=',')