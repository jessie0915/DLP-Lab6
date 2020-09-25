import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataloader import iclevrDataset
from evaluator import evaluation_model
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

channels = 3
image_size = 64
img_shape = (channels, image_size, image_size)
lr = 0.0002
b1 = 0.5
b2 = 0.999
latent_dim = 256
n_epochs = 1000
n_critic = 5
sample_interval = 400
batch_size = 32
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

cuda = True if torch.cuda.is_available() else False
n_classes = 24

eval_model = evaluation_model()

if not os.path.isdir('logs/images'):
    os.mkdir('logs/images')

if not os.path.isdir('logs/backup'):
    os.mkdir('logs/backup')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + n_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, attr):
        attr = attr.view(-1, n_classes, 1, 1)
        x = torch.cat([z, attr], 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_input = nn.Linear(n_classes, 64 * 64)
        self.main = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img, attr):
        attr = self.feature_input(attr).view(-1, 1, 64, 64)
        x = torch.cat([img, attr], 1)
        return self.main(x).view(-1, 1)


class Trainer:
    def __init__(self, path_g, path_d):
        self.generator = Generator()
        self.discriminator = Discriminator()
        if path_g != 'null':
            self.generator.load_state_dict(torch.load(path_g))
        if path_d != 'null':
            self.discriminator.load_state_dict(torch.load(path_d))

        self.loss = nn.MSELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        self.generator.cuda()
        self.discriminator.cuda()
        self.loss.cuda()

    def sample_image(self, batch_size, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        with torch.no_grad():
            # Get labels ranging from 0 to n_classes for n rows
            for i, labels in enumerate(val_loader):
                # Sample noise
                batch_size = labels.size(0)
                z = Variable(torch.FloatTensor(batch_size, latent_dim, 1, 1).normal_(0, 1).cuda())

                labels_acc = labels
                labels_acc = labels_acc.cuda()

                labels = labels.view(-1, n_classes, 1, 1)
                labels = labels.cuda()
                gen_imgs = self.generator(z, labels)

                save_image(gen_imgs.data, "logs/images/%d.png" % batches_done, nrow=8, normalize=True)
                acc = eval_model.eval(gen_imgs, labels_acc)


        return acc

    def train(self, dataloader, batch_size):
        fp = open("logs/backup/log.txt", "w")
        fp.close()
        fp = open("logs/backup/log_epoch.txt", "w")
        fp.close()
        fp = open("logs/backup/log_acc.txt", "w")
        fp.close()

        batches_done = 0
        acc_value = 0
        for epoch in range(n_epochs):
            for i, (data, attr) in enumerate(dataloader, 0):
                # train discriminator
                self.discriminator.zero_grad()

                batch_size = data.size(0)
                noise = Variable(torch.FloatTensor(batch_size, latent_dim, 1, 1).cuda())
                label_real = Variable(torch.FloatTensor(batch_size, 1).fill_(1).cuda())
                label_fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0).cuda())

                label_real.data.resize(batch_size, 1).fill_(1)
                label_fake.data.resize(batch_size, 1).fill_(0)
                noise.data.resize_(batch_size, latent_dim, 1, 1).normal_(0, 1)

                attr = Variable(attr.cuda())
                real = Variable(data.cuda())
                d_real = self.discriminator(real, attr)

                fake = self.generator(noise, attr)
                d_fake = self.discriminator(fake.detach(), attr)  # not update generator

                d_loss = self.loss(d_real, label_real) + self.loss(d_fake, label_fake)  # real label
                d_loss_value = d_loss.item()
                d_loss.backward()
                self.optimizer_d.step()

                # train generator
                # Train the generator every n_critic steps
                if i % n_critic == 0:
                    for _ in range(5):
                        self.generator.zero_grad()
                        noise.data.resize_(batch_size, latent_dim, 1, 1).normal_(0, 1)
                        fake = self.generator(noise, attr)
                        d_fake = self.discriminator(fake, attr)
                        g_loss = self.loss(d_fake, label_real)  # trick the fake into being real
                        g_loss_value = g_loss.item()
                        g_loss.backward()
                        self.optimizer_g.step()

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Accuary: %f]"
                        % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item(), acc_value)
                    )
                    fp = open("logs/backup/log.txt", "a")
                    fp.write(str(epoch) + ',' + str(i) + ',' + str(d_loss.item()) + ',' + str(g_loss.item()) + '\n')
                    fp.close()

                    if batches_done % sample_interval == 0:
                        acc = self.sample_image(batch_size, batches_done)
                        acc_value = acc
                        # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                        fp = open("logs/backup/log_acc.txt", "a")
                        fp.write(
                            str(epoch) + ',' + str(i) + ',' + str(d_loss.item()) + ',' + str(g_loss.item()) + ',' + str(
                                acc) + '\n')
                        fp.close()

                    batches_done += n_critic

            fp = open("logs/backup/log_epoch.txt", "a")
            fp.write(str(epoch) + ',' + str(d_loss_value) + ',' + str(g_loss_value) + ',' + str(acc_value) + '\n')
            fp.close()

            g_path = 'logs/backup/epoch_' + str(epoch) + '_gen.pt'
            torch.save(self.generator.state_dict(), g_path)
            d_path = 'logs/backup/epoch_' + str(epoch) + '_dis.pt'
            torch.save(self.discriminator.state_dict(), d_path)


# Configure data loader
# create train/val transforms
train_transform = transforms.Compose([
                      transforms.Resize((image_size, image_size)),
                      # transforms.RandomHorizontalFlip(p=0.2),
                      # transforms.RandomVerticalFlip(p=0.2),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5]),
                  ])
val_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5]),
                ])
# create train/val datasets
trainset = iclevrDataset(root='iclevr/',
                         mode='train',
                         transform=train_transform)
valset = iclevrDataset(root='iclevr/',
                       mode='test',
                       transform=val_transform)

# create train/val loaders
train_loader = DataLoader(dataset=trainset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
val_loader = DataLoader(dataset=valset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0)

# trainer = Trainer('logs/model_weights/epoch_77_gen.pt', 'logs/model_weights/epoch_77_dis.pt')
# max_acc = 0.0
# max_idx = 0
# for k in range(20):
#     acc = trainer.sample_image(batch_size, k+1)
#     if max_acc < acc:
#         max_acc = acc
#         max_idx = k
# print('%d Accuracu = %f' % (max_idx+1, max_acc))

trainer = Trainer('null', 'null')
trainer.train(train_loader, batch_size)