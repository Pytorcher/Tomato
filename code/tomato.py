import  torch
import  os, glob
import  random, csv

from    torch.utils.data import Dataset, DataLoader

from    torchvision import transforms
from    PIL import Image


class Tomato(Dataset):

    def __init__(self, root, resize, mode):
        super(Tomato, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {} # "sq...":0
        for name in sorted(os.listdir(os.path.join(root))):  ##  将数据集中各种类用数字对应
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)

        # image, label
        self.images, self.labels = self.load_csv('images.csv')  ##注意这里load_csv()函数取得了数据集的路径与标签

        if mode=='train': # 60%
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode=='val': # 20% = 60%->80%
            self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        else: # 20% = 80%->100%
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]





    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):  ##注意，是如果过image.csv不存在才重新创建，因此如果数据集发生变化，需要把image.csv删了再重新运行
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            print(len(images), images)

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: 
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label) ## 关键哦，拿到了image和label的信息

        assert len(images) == len(labels)

        return images, labels



    def __len__(self):

        return len(self.images)


    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x


    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # label: 0
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path= > image data 路径转化为真正的图片
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))), 
            transforms.RandomRotation(15), ## 稍微旋转即可，15°
            transforms.CenterCrop(self.resize),  ##真正的resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], ## train from imagenet(标准化)
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)


        return img, label





def main():

    import  visdom
    import  time
    import  torchvision
    ##需提前在terminal中打开visdom> python -m visdom.server
    viz = visdom.Visdom()



    db = Tomato('tomato', 64, 'train')

    x,y = next(iter(db))
    print('sample:', x.shape, y.shape, y)

    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)

    for x,y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(5)

if __name__ == '__main__':
    main()