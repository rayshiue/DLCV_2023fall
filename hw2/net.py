import torch
import torch.nn as nn

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class MyBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, normalize=True):
        super(MyBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.norm1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.norm2 = nn.BatchNorm2d(out_c)
        self.activation = nn.SiLU()
        
        self.normalize = normalize

    def forward(self, x):
        # x = self.ln(x) if self.normalize else x
        x1 = self.conv1(x)
        if self.normalize:
            x1 = self.norm1(x1)
        x1 = self.activation(x1)
        x2 = self.conv2(x1)
        if self.normalize:
            x2 = self.norm2(x2)
        x2 = self.activation(x2)
        return x2
    
class MyUNet(nn.Module):
    def __init__(self, n_steps=500, n_class = 10, d=160):
        super(MyUNet, self).__init__()

        self.d = d
        # Sinusoidal embedding
        self.timeembed1 = EmbedFC(1, 2*d)
        self.timeembed2 = EmbedFC(1, 1*d)
        self.contextembed1 = EmbedFC(10, 2*d)
        self.contextembed2 = EmbedFC(10, 1*d)



        self.b1 = nn.Sequential(
            MyBlock(3, d),
            # MyBlock((d, 28, 28), d, d),
            # MyBlock((d, 28, 28), d, d)
        )
        # self.down1 = nn.MaxPool2d(2)
        self.down1 = nn.Conv2d(d, d, 4, 2, 1)

        self.b2 = nn.Sequential(
            MyBlock(d, 2*d),
            # MyBlock((2*d, 14, 14), 2*d, 2*d),
            # MyBlock((2*d, 14, 14), 2*d, 2*d)
        )
        # self.down2 = nn.MaxPool2d(2)
        self.down2 = nn.Conv2d(2*d, 2*d, 4, 2, 1)

        self.b3 = nn.Sequential(
            MyBlock(2*d, 4*d),
            # MyBlock((4*d, 7, 7), 4*d, 4*d),
            # MyBlock((4*d, 7, 7), 4*d, 4*d)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(4*d, 4*d, 2, 1),
            nn.GELU(),
            nn.Conv2d(4*d, 4*d, 4, 2, 1)
        )

        # Bottleneck
        self.b_mid = nn.Sequential(
            MyBlock(4*d, 4*d),
            # MyBlock((2*d, 3, 3), 2*d, 4*d),
            # MyBlock((2*d, 3, 3), 2*d, 4*d)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(4*d, 4*d, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(4*d, 4*d, 2, 1)
        )

        self.b4 = nn.Sequential(
            MyBlock(8*d, 4*d),
            MyBlock(4*d, 2*d),
            # MyBlock((2*d, 7, 7), 2*d, 2*d)
        )

        self.up2 = nn.ConvTranspose2d(2*d, 2*d, 4, 2, 1)
        self.b5 = nn.Sequential(
            MyBlock(4*d, 2*d),
            MyBlock(2*d, 1*d),
            # MyBlock((1*d, 14, 14), 1*d, 1*d)
        )

        self.up3 = nn.ConvTranspose2d(d, d, 4, 2, 1)
        self.b_out = nn.Sequential(
            MyBlock(2*d, 1*d),
            MyBlock(d, d, normalize=False),
            # MyBlock((d, 28, 28), d, d, normalize=False),
            nn.GroupNorm(8, d),
            nn.ReLU(),
        )

        self.conv_out = nn.Conv2d(d, 3, 3, 1, 1)

    def forward(self, x, c, t, context_mask):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        # print(t.shape)
        
        # t = self.time_embed(t.long())
        # c = torch.Tensor([0])
        # t = torch.Tensor([0])
        c = nn.functional.one_hot(c, num_classes=10).type(torch.float)
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,10)
        context_mask = -1*(1-context_mask)  # need to flip 0 <-> 1

        c = c * context_mask

        # c = self.class_embed(c.long())
        # n = len(x)


        cemb1 = self.contextembed1(c).view(-1, self.d * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.d * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.d, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.d, 1, 1)

        # print(x.shape)
        out1 = self.b1(x)  # (N, 10, 28, 28)
        # print(out1.shape)
        out2 = self.b2(self.down1(out1))  # (N, 20, 14, 14)

        out3 = self.b3(self.down2(out2))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4)  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4*cemb1+temb1)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5)  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5*cemb2+temb2)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out)  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out