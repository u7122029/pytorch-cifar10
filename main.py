import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import torch.nn as nn

import argparse

from pytorch_models import *
from pytorch_models.LeNet5 import LeNet5, lenet5
from pytorch_models.Linear import Linear
from pytorch_models.OBC import OBC
from misc import progress_bar


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--epoch', default=75, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--test', type=str, help="Test given model.")
    args = parser.parse_args()

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    solver = Solver(args)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.test_model_file = config.test

    def load_data(self):
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #if self.test_model_file:
        #self.model = torch.load("linear.pth")
        #self.model = self.model.to(self.device)
        #else:
        #self.model = LeNet5().to(self.device)
        self.model = OBC().to(self.device)
        #self.model = Linear().to(self.device)
            # self.model = LeNet().to(self.device)
        #self.model = AlexNet().to(self.device)
            # self.model = VGG11().to(self.device)
            # self.model = VGG13().to(self.device)
            # self.model = VGG16().to(self.device)
            # self.model = VGG19().to(self.device)
            # self.model = GoogLeNet().to(self.device)
            # self.model = resnet18().to(self.device)
            # self.model = resnet34().to(self.device)
            # self.model = resnet50().to(self.device)
            # self.model = resnet101().to(self.device)
            # self.model = resnet152().to(self.device)
            # self.model = DenseNet121().to(self.device)
            # self.model = DenseNet161().to(self.device)
            # self.model = DenseNet169().to(self.device)
            # self.model = DenseNet201().to(self.device)
            #self.model = WideResNet(depth=28, num_classes=10).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)

            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        if self.test_model_file:
            test_result = self.test()
            print(test_result)
            return

        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step()
            print("\n===> epoch: %d/200" % epoch)
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            if test_result[1] > accuracy:
                accuracy = test_result[1]
                self.save()

            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))


if __name__ == '__main__':
    main()
