import  torch
from    torch import optim, nn
import  visdom
import  torchvision
from    torch.utils.data import DataLoader

from    tomato import Tomato
from    torchvision.models import VGG19
from    torchvision.models import resnet101
from    torchvision.models import mobilenet_v3_small
from    torchvision.models import mobilenet_v3_large
from    torchvision.models import shufflenet_v2_x0_5
from    torchvision.models import resnet101
#from    torchvision.models import
from    utils import Flatten

batchsz = 32
lr = 1e-3
epochs = 20

device = torch.device('cuda')
torch.manual_seed(1234)


train_db = Tomato('tomato', 224, mode='train')
val_db = Tomato('tomato', 224, mode='val')
test_db = Tomato('tomato', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=2)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=1)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=1)


viz = visdom.Visdom()

def evalute(model, loader):
    model.eval()
    
    correct = 0
    total = len(loader.dataset)

    for x,y in loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total

def main():

    trained_model = resnet101(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], #[b, 512, 1, 1]
                          Flatten(), # [b, 512, 1, 1] => [b, 512]
                          nn.Linear(2048, 9)
                          ).to(device)
    # x = torch.randn(2, 3, 224, 224)
    # print(model(x).shape)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()


    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):
        if epoch>10:
            optimizer = optim.Adam(model.parameters(), lr=0.1*lr)
        for step, (x,y) in enumerate(train_loader):

            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        #if epoch % 1 == 0:

            val_acc = evalute(model, val_loader)
            if val_acc> best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best.pth')

                viz.line([val_acc], [global_step], win='val_acc', update='append')


    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.pth'))
    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)





if __name__ == '__main__':
    main()
