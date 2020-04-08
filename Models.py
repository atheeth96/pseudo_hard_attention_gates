
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Functions to initialize Weight of network

def init_weights(model, init_type='normal', gain=0.02):
    
    '''
        Arguments :
            model       : the model whose weights have to be initialized
            init_type   : The type if initialization to perform on the weights  
    '''
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method {} is not implemented in pytorch'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialized network with {} initialization'.format(init_type))
    model.apply(init_func)

# Function to save the model, the optimizer and scheduler 

def save_model(model,optimizer,name,scheduler=None):
    '''
        Arguments :
            model       : the model whose weights have to be initialized
            optimizer   : The optimizer used 
            scheduler . : scheduler if present
    '''
    
    
    if scheduler==None:
        checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    else:
        checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()}

    torch.save(checkpoint,name)

    
# Function to load the model optimizer and scheduler state_dict 

def load_model(filename,model,optimizer=None,scheduler=None):
    
    '''
        Arguments :
            filename       : file name/path of the model weights have to be initialized
            model          : Model architecture
            optimizer      : The optimizer used 
            scheduler      : scheduler if present
    '''
    
    
    checkpoint=torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    print("Done loading")
    if  optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(optimizer.state_dict()['param_groups'][-1]['lr'],' : Learning rate')
    if  scheduler:
        scheduler.load_state_dict(checkpoint['optimizer'])
        print(scheduler.state_dict()['param_groups'][-1]['lr'],' : Learning rate')

# a single conv_block used in U-Net and its variants

class conv_block(nn.Module):
    
    '''
        Arguments :
            ch_in       : Number of channels in input tensor along dim =1
            ch_out      : Number of channels required  in out tensor along dim =1

    '''
    
    
    def __init__(self,ch_in,ch_out):
        
        super().__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    
    '''
        Arguments :
            ch_in       : Number of channels in input tensor along dim =1
            ch_out      : Number of channels required  in out tensor along dim =1

    '''
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.up = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x



class Attention_block(nn.Module):
    
    '''
        Arguments :
            F_g         : Number of channels in skip connection tensor along dim =1
            F_l         : Number of channels in the current expanding path out tensor along dim =1
            F_int       : Number of channels in the intermediate state required

    '''
    
    def __init__(self,F_g,F_l,F_int):
        
        
        super().__init__()
        self.W_g = nn.Sequential(
        nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
        nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
        nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(1),
        nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi,psi

class U_Net(nn.Module):
    
    '''
    Implementation of U-Net from "U-net:Convolutionalnetworks for biomedical image segmentation,” in International Conference on     Medicalimagecomputingandcomputer-assistedintervention. Springer, 2015
    
        Arguments :
            img_ch      : Number of channels in input image
            output_ch   : Number of channels in output image  
            dropout     : dropout rate to be applied to the connections going to latent space
    '''
    def __init__(self,img_ch=3,output_ch=1,dropout=0.45):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        self.dropout=torch.nn.Dropout(dropout)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x,x_2):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4=self.dropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1,None

class AttnUNet(nn.Module):
    
    '''
    Implementation of Attention U-Net from "Attention U-Net: Learning Where to Look for the Pancreas" 2018
    
        Arguments :
            img_ch      : Number of channels in input image
            output_ch   : Number of channels in output image  
            dropout     : dropout rate to be applied to the connections going to latent space
    '''
    
    
    def __init__(self,img_ch=1,output_ch=1,dropout=0.45):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        if dropout is not None:
            self.dropout=torch.nn.Dropout(dropout)
        else:
            self.dropout=dropout

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x,x_2):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        if self.dropout is not None:
            x5=self.dropout(x5)
        x5 = self.Conv5(x5)
        

        # decoding + concat path
        d5 = self.Up5(x5)
        x4,psi4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3,psi3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2,psi2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1,psi1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1,[psi1,psi2,psi3,psi4]
    
    
class ChannelPool(nn.Module):
    
    '''
    Implementation of Channel wise max pooling 
    
        Arguments :
            out_channels : Number of channels in output image 
    '''
    
    
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        
    def forward(self, x):
        
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])
    
    

    
# class ASM(nn.Module):
#     def __init__(self,F_ip,F_int):
#         super().__init__()
#         self.ChannelPool = ChannelPool(F_int)
#         self.W_1x1 = nn.Conv2d(F_ip, F_int, kernel_size=1,stride=1,padding=0,bias=True)
#         self.batch_norm=nn.BatchNorm2d(F_int)

#     def forward(self,map_1_fm,map_2_fm):
        
#         x=torch.cat((map_1_fm,map_2_fm),dim=1)
#         x1=self.ChannelPool(x)
#         psi=self.W_1x1(x)
#         psi=torch.sigmoid(psi)
#         psi=self.batch_norm(psi)
        

#         return psi*x1
    
# class ASM(nn.Module):
    
#     '''
#     Implementation of Attention Skip Module
    
#         Arguments :
#             F_int : Number of channels in output feature map 
#             F_ip  : Number of channels in input feature map 
#     '''
    
    
#     def __init__(self,F_int,F_ip):
#         super().__init__()
        
#         self.W_e1 = nn.Sequential(
#         nn.Conv2d(F_ip, F_int, kernel_size=1,stride=1,padding=0,bias=True),
#         nn.BatchNorm2d(F_int)
#         )

#         self.W_e2 = nn.Sequential(
#         nn.Conv2d(F_ip, F_int, kernel_size=1,stride=1,padding=0,bias=True),
#         nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#         nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
#         nn.BatchNorm2d(1),
#         nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)
        
# #         self.ChannelPool = ChannelPool(F_int)
#         self.W_1x1 = nn.Conv2d(F_int, F_ip, kernel_size=1,stride=1,padding=0,bias=True)
       

#     def forward(self,map_1_fm,map_2_fm):
#         x1 = self.W_e1(map_1_fm)
#         x2 = self.W_e2(map_2_fm)
#         x3 = self.W_1x1(x1)
#         psi = self.relu(x2+x1)
#         psi = self.psi(psi)
       
#         return psi*x3,psi
    

class ASM(nn.Module):
    
    '''
    Implementation of Attention Skip Module
    
        Arguments :
            F_int : Number of channels in output feature map 
            F_ip  : Number of channels in input feature map 
    '''
    
    
    def __init__(self,F_int,F_ip):
        super().__init__()
        
        self.W_e1 = nn.Conv2d(F_ip, F_int, kernel_size=1,stride=1,padding=0,bias=True)


        self.W_e2 =nn.Conv2d(F_ip, F_int, kernel_size=1,stride=1,padding=0,bias=True)


        self.psi = nn.Sequential(
        nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
        nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        

        self.W_1x1 = nn.Sequential(
            nn.Conv2d(F_ip, F_ip, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_ip))
       
    def forward(self,map_1_fm,map_2_fm):
        x1 = self.W_e1(map_1_fm)
        x2 = self.W_e2(map_2_fm)
        
        psi = self.relu(x2+x1)
        psi = self.psi(psi)
        result=self.W_1x1(psi*map_1_fm)
       
        return result,psi
    
class BEM(nn.Module):
    
    '''
    Implementation of Boundary Enhance Module
    
        Arguments :
            F_int : Number of channels in output feature map 
            F_ip  : Number of channels in input feature map 
    '''
    
    
    def __init__(self,F_int,F_ip):
        super().__init__()
        
        self.W_e1 = nn.Conv2d(F_ip, F_int, kernel_size=1,stride=1,padding=0,bias=True)


        self.W_e2 =nn.Conv2d(F_ip, F_int, kernel_size=1,stride=1,padding=0,bias=True)


        self.psi = nn.Sequential(
        nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
        nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        

        self.W_1x1 = nn.Sequential(
            nn.Conv2d(F_ip, F_ip, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_ip))
       
    def forward(self,map_1_fm,map_2_fm):
        x1 = self.W_e1(map_1_fm)
        x2 = self.W_e2(map_2_fm)
        
        psi = self.relu(x2+x1)
        psi = self.psi(psi)
        result=self.W_1x1(psi*map_1_fm)
       
        return result
    
class FFM(nn.Module):
    
    '''
    Implementation of Final Fussion Module
    
        Arguments :
            F_int : Number of channels in intermediate feature map 
 
    '''
    
    def __init__(self,F_int):
        super().__init__()
        self.W_x = nn.Sequential(
        nn.Conv2d(F_int, F_int, kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(F_int),
        nn.ReLU(inplace=True)
        )
#         self.ChannelPool = ChannelPool(F_int)
#         self.W_1x1 = nn.Conv2d(F_int, F_int, kernel_size=1,stride=1,padding=0,bias=True)
#         self.W_1x1 = nn.Conv2d(F_int, F_int, kernel_size=1,stride=1,padding=0,bias=True)
        self.W_1x1 = nn.Conv2d(F_int, F_int, kernel_size=1,stride=1,padding=0,bias=True)
        self.batch_norm=nn.BatchNorm2d(F_int)
      

    def forward(self,map_1_fm,map_2_fm):
        
        x=torch.cat((map_1_fm,map_2_fm),dim=1)
        x=self.W_x(x)
     
        psi=self.W_1x1(x)
        psi=torch.sigmoid(psi)
        
        x1=x*psi
        

        return x1+x
    
    
class DualEncoding_U_Net(nn.Module):
    
    '''
    Implementation of Dual encoding U-Net
    
        Arguments :
            img1_ch     : Number of channels in first image input
            img2_ch     : Number of channels in second image input
            output_ch   : Number of channels in output input
            dropout     : dropout rate to be applied to the connections going to latent space
            include_ffm : Boolean value to determine if second encoding path is to be fused with latent space
 
    '''

    def __init__(self,img1_ch=3,img2_ch=1,output_ch=2,dropout=0.25,include_ffm=True):
        super().__init__()
        self.include_ffm=include_ffm

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        if dropout is not None:
            self.dropout=torch.nn.Dropout(dropout)
        else:
            self.dropout=dropout

        self.Conv1_encoding_1 = conv_block(ch_in=img1_ch,ch_out=64)
        self.Conv2_encoding_1 = conv_block(ch_in=64,ch_out=128)
        self.Conv3_encoding_1 = conv_block(ch_in=128,ch_out=256)
        self.Conv4_encoding_1 = conv_block(ch_in=256,ch_out=512)

        self.Conv1_encoding_2 = conv_block(ch_in=img2_ch,ch_out=64)
        self.Conv2_encoding_2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3_encoding_2 = conv_block(ch_in=128,ch_out=256)
        self.Conv4_encoding_2 = conv_block(ch_in=256,ch_out=512)
        
        
        if self.include_ffm:
            self.ffm=FFM(1024)
        else:
            self.Conv5_encoding_1 = conv_block(ch_in=512,ch_out=1024)
        
        
        self.asm4=ASM(128,64)
        self.asm3=ASM(256,128)
        self.asm2=ASM(512,256)
        self.asm1=ASM(1024,512)
      

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = nn.Sequential(nn.Conv2d(1024,512,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True))

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = nn.Sequential(nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x_h,x_e):
        # encoding path
        x_h1 = self.Conv1_encoding_1(x_h)
        # N*64*128*128

        x_h2 = self.Maxpool(x_h1)
        x_h2 = self.Conv2_encoding_1(x_h2)
        # N*128*64*64

        x_h3 = self.Maxpool(x_h2)
        x_h3 = self.Conv3_encoding_1(x_h3)
        # N*256*32*32

        x_h4 = self.Maxpool(x_h3)
        x_h4 = self.Conv4_encoding_1(x_h4)
        # N*512*16*16
        
        x_e1 = self.Conv1_encoding_2(x_e)
        # N*64*128*128

        x_e2 = self.Maxpool(x_e1)
        x_e2 = self.Conv2_encoding_2(x_e2)
        # N*128*64*64

        x_e3 = self.Maxpool(x_e2)
        x_e3 = self.Conv3_encoding_2(x_e3)
    
        # N*256*32*32

        x_e4 = self.Maxpool(x_e3)
        x_e4 = self.Conv4_encoding_2(x_e4)
        # N*512*16*16
        
        if self.dropout is not None:
            lat_spc=self.dropout(x_h4)
            if self.include_ffm:
                lat_spc2=self.dropout(x_e4)
                lat_spc=self.ffm(lat_spc,lat_spc2)
                
            else:
                lat_spc=self.Conv5_encoding_1(lat_spc)
        else:
            if self.include_ffm:
                lat_spc=self.ffm(x_h4,x_e4)
            else:
                lat_spc=self.Conv5_encoding_1(x_h4)
            
        lat_spc=self.Maxpool(lat_spc)
        
        # N*1024*8*8
      
        
        # decoding + concat path
        
        d5 = self.Up5(lat_spc)
        # N*512*16*16
        
        x4,psi1=self.asm1(x_h4,x_e4)
        # N*512*16*16
        
        d5 = torch.cat((x4,d5),dim=1)
        # N*1024*16*16
        d5=self.Up_conv5(d5)
        # N*512*16*16
        
        d4 = self.Up4(d5)
        # N*256*32*32
        x3,psi2=self.asm2(x_h3,x_e3)
        # N*256*32*32
        d4 = torch.cat((x3,d4),dim=1)
        # N*512*32*32
        d4 = self.Up_conv4(d4)
        # N*256*32*32

        d3 = self.Up3(d4)
        # N*128*64*64
        x2,psi3=self.asm3(x_h2,x_e2)
        # N*128*64*64
        
        d3 = torch.cat((x2,d3),dim=1)
        # N*256*64*64
        d3 = self.Up_conv3(d3)
        # N*128*64*64

        d2 = self.Up2(d3)
        # N*64*128*128
        x1,psi4=self.asm4(x_h1,x_e1)
        # N*64*128*128
        d2 = torch.cat((x1,d2),dim=1)
        # N*128*128*128
        d2 = self.Up_conv2(d2)
        # N*64*128*128

        d1 = self.Conv_1x1(d2)
        attention_maps=[psi4,psi3,psi2,psi1]
        # N*2*128*128

        return d1,attention_maps
    
    
class DualEncoding_U_Net_without_asm(nn.Module):
    
    '''
    Implementation of Dual encoding U-Net
    
        Arguments :
            img1_ch     : Number of channels in first image input
            img2_ch     : Number of channels in second image input
            output_ch   : Number of channels in output input
            dropout     : dropout rate to be applied to the connections going to latent space
            include_ffm : Boolean value to determine if second encoding path is to be fused with latent space
 
    '''

    def __init__(self,img1_ch=3,img2_ch=1,output_ch=2,dropout=0.25):
        super().__init__()
        

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        if dropout is not None:
            self.dropout=torch.nn.Dropout(dropout)
        else:
            self.dropout=dropout

        self.Conv1_encoding_1 = conv_block(ch_in=img1_ch,ch_out=64)
        self.Conv2_encoding_1 = conv_block(ch_in=64,ch_out=128)
        self.Conv3_encoding_1 = conv_block(ch_in=128,ch_out=256)
        self.Conv4_encoding_1 = conv_block(ch_in=256,ch_out=512)

        self.Conv5_encoding_1 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = nn.Sequential(nn.Conv2d(1024,512,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True))

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = nn.Sequential(nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x_h,x_e):
        # encoding path
        x_h1 = self.Conv1_encoding_1(x_h)
        # N*64*128*128

        x_h2 = self.Maxpool(x_h1)
        x_h2 = self.Conv2_encoding_1(x_h2)
        # N*128*64*64

        x_h3 = self.Maxpool(x_h2)
        x_h3 = self.Conv3_encoding_1(x_h3)
        # N*256*32*32

        x_h4 = self.Maxpool(x_h3)
        x_h4 = self.Conv4_encoding_1(x_h4)
        # N*512*16*16
        
      
        if self.dropout is not None:
            lat_spc=self.dropout(x_h4)
            lat_spc=self.Conv5_encoding_1(lat_spc)
        else:
            lat_spc=self.Conv5_encoding_1(x_h4)
            
        lat_spc=self.Maxpool(lat_spc)
              
        d5 = self.Up5(lat_spc)
        # N*512*16*16
        
        x4=x_h4*F.interpolate(x_e, size=d5.size()[-2:])
        # N*512*16*16
        
        d5 = torch.cat((x4,d5),dim=1)
        # N*1024*16*16
        d5=self.Up_conv5(d5)
        # N*512*16*16
        
        d4 = self.Up4(d5)
        # N*256*32*32
        x3=x_h3*F.interpolate(x_e, size=d4.size()[-2:])
        # N*256*32*32
        d4 = torch.cat((x3,d4),dim=1)
        # N*512*32*32
        d4 = self.Up_conv4(d4)
        # N*256*32*32

        d3 = self.Up3(d4)
        # N*128*64*64
        x2=x_h2*F.interpolate(x_e, size=d3.size()[-2:])
        # N*128*64*64
        
        d3 = torch.cat((x2,d3),dim=1)
        # N*256*64*64
        d3 = self.Up_conv3(d3)
        # N*128*64*64

        d2 = self.Up2(d3)
        # N*64*128*128
        x1=x_h1*F.interpolate(x_e, size=d2.size()[-2:])
        # N*64*128*128
        d2 = torch.cat((x1,d2),dim=1)
        # N*128*128*128
        d2 = self.Up_conv2(d2)
        # N*64*128*128

        d1 = self.Conv_1x1(d2)
        # N*2*128*128

        return d1,[F.interpolate(x_e, size=d2.size()[-2:])\
                   ,F.interpolate(x_e, size=d3.size()[-2:])\
                   ,F.interpolate(x_e, size=d4.size()[-2:])\
                   ,F.interpolate(x_e, size=d5.size()[-2:])]
    
    
class DualEncodingDecoding_U_Net(nn.Module):
    
    '''
    Implementation of Dual encoding and Dual decoding U-Net
    
        Arguments :
            img1_ch     : Number of channels in first image input
            img2_ch     : Number of channels in second image input
            dropout     : dropout rate to be applied to the connections going to latent space
           
    '''

    
    def __init__(self,img_ch1=3,img_ch2=1,output_ch1=1,output_ch2=1,dropout=0.45):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1_encoding_1 = conv_block(ch_in=img_ch1,ch_out=64)
        self.Conv2_encoding_1 = conv_block(ch_in=64,ch_out=128)
        self.Conv3_encoding_1 = conv_block(ch_in=128,ch_out=256)
        self.Conv4_encoding_1 = conv_block(ch_in=256,ch_out=512)
        self.Conv5_encoding_1 = conv_block(ch_in=512,ch_out=1024)
        
        
        
        self.Conv1_encoding_2 = conv_block(ch_in=img_ch2,ch_out=64)
        self.Conv2_encoding_2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3_encoding_2 = conv_block(ch_in=128,ch_out=256)
        self.Conv4_encoding_2 = conv_block(ch_in=256,ch_out=512)
        
#         self.ffm=FFM(1024)
        
        self.asm4=ASM(128,64)
        self.asm3=ASM(256,128)
        self.asm2=ASM(512,256)
        self.asm1=ASM(1024,512)
        
        
        self.bem4=BEM(128,64)
        self.bem3=BEM(256,128)
        self.bem2=BEM(512,256)
        self.bem1=BEM(1024,512)
        
        
      
        self.dropout=nn.Dropout(dropout)

        self.Up_sample5_decod_1 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5_decod_1 = nn.Sequential(nn.Conv2d(1024,512,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True))

        self.Up_sample4_decod_1 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4_decod_1 = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))

        self.Up_sample3_decod_1 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3_decod_1 = nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))

        self.Up_sample2_decod_1 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2_decod_1 = nn.Sequential(nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        
       
        self.Up_sample5_decod_2 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5_decod_2 = nn.Sequential(nn.Conv2d(1024,512,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True))

        self.Up_sample4_decod_2 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4_decod_2 = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))

        self.Up_sample3_decod_2 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3_decod_2 = nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))

        self.Up_sample2_decod_2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2_decod_2 = nn.Sequential(nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        
        self.Conv_1x1_decod_1 = nn.Conv2d(64,output_ch1,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_decod_2 = nn.Conv2d(64,output_ch2,kernel_size=1,stride=1,padding=0)

        


    def forward(self,x_1,x_2):
    # encoding path
        x1_encode_1 = self.Conv1_encoding_1(x_1)
        # N*64*512*512

        x2_encode_1 = self.Maxpool(x1_encode_1)
        x2_encode_1 = self.Conv2_encoding_1(x2_encode_1)
        # N*128*256*256

        x3_encode_1 = self.Maxpool(x2_encode_1)
        x3_encode_1 = self.Conv3_encoding_1(x3_encode_1)
        # N*256*128*128

        x4_encode_1 = self.Maxpool(x3_encode_1)
        x4_encode_1 = self.Conv4_encoding_1(x4_encode_1)
        # N*512*64*64

        x1_encode_2 = self.Conv1_encoding_2(x_2)
        # N*64*512*512

        x2_encode_2 = self.Maxpool(x1_encode_2)
        x2_encode_2 = self.Conv2_encoding_2(x2_encode_2)
        # N*128*256*256

        x3_encode_2 = self.Maxpool(x2_encode_2)
        x3_encode_2 = self.Conv3_encoding_2(x3_encode_2)

        # N*256*128*128

        x4_encode_2 = self.Maxpool(x3_encode_2)
        x4_encode_2 = self.Conv4_encoding_2(x4_encode_2)
        # N*512*64*64

        lat_spc=self.dropout(x4_encode_1)
        lat_spc=self.Conv5_encoding_1(lat_spc)
        lat_spc=self.Maxpool(lat_spc)
        # N*1024*32*32


        # decoding + concat path

        d5_decod_1 = self.Up_sample5_decod_1(lat_spc)
        # N*512*64*64

        x4_decod_1,psi1=self.asm1(x4_encode_1,x4_encode_2)
        # N*512*64*64

        d5_decod_1 = torch.cat((x4_decod_1,d5_decod_1),dim=1)
        # N*1024*64*64
        d5_decod_1=self.Up_conv5_decod_1(d5_decod_1)
        # N*512*64*64

        d5_decod_2 = self.Up_sample5_decod_2(lat_spc)
        
        # N*512*64*64
    
        x4_decod_2=self.bem1(x4_decod_1,d5_decod_2)
        # N*512*64*64

        d5_decod_2 = torch.cat((x4_decod_2,d5_decod_2),dim=1)
        
        # N*1024*64*64
        d5_decod_2=self.Up_conv5_decod_2(d5_decod_2)


        ###
        d4_decod_1 = self.Up_sample4_decod_1(d5_decod_1)
        # N*512*64*64

        x3_decod_1,psi2=self.asm2(x3_encode_1,x3_encode_2)
        # N*512*64*64

        d4_decod_1 = torch.cat((x3_decod_1,d4_decod_1),dim=1)
        # N*1024*64*64
        d4_decod_1=self.Up_conv4_decod_1(d4_decod_1)
        # N*512*64*64

        d4_decod_2 = self.Up_sample4_decod_2(d5_decod_2)
        # N*512*64*64

        x3_decod_2=self.bem2(x3_decod_1,d4_decod_2)
        # N*512*64*64

        d4_decod_2 = torch.cat((x3_decod_2,d4_decod_2),dim=1)
        # N*1024*64*64
        d4_decod_2=self.Up_conv4_decod_2(d4_decod_2)
        ###

        d3_decod_1 = self.Up_sample3_decod_1(d4_decod_1)
        # N*512*64*64

        x2_decod_1,psi3=self.asm3(x2_encode_1,x2_encode_2)
        # N*512*64*64

        d3_decod_1 = torch.cat((x2_decod_1,d3_decod_1),dim=1)
        # N*1024*64*64
        d3_decod_1=self.Up_conv3_decod_1(d3_decod_1)
        
        # N*512*64*64

        d3_decod_2 = self.Up_sample3_decod_2(d4_decod_2)
        # N*512*64*64

        x2_decod_2=self.bem3(x2_decod_1,d3_decod_2)
        # N*512*64*64

        d3_decod_2 = torch.cat((x2_decod_2,d3_decod_2),dim=1)
        # N*1024*64*64
        d3_decod_2=self.Up_conv3_decod_2(d3_decod_2)

        ###

        d2_decod_1 = self.Up_sample2_decod_1(d3_decod_1)
        # N*512*64*64

        x1_decod_1,psi4=self.asm4(x1_encode_1,x1_encode_2)
        # N*512*64*64

        d2_decod_1 = torch.cat((x1_decod_1,d2_decod_1),dim=1)
        # N*1024*64*64
        d2_decod_1=self.Up_conv2_decod_1(d2_decod_1)
        # N*512*64*64

        d2_decod_2 = self.Up_sample2_decod_2(d3_decod_2)
        # N*512*64*64

        x1_decod_2=self.bem4(x1_decod_1,d2_decod_2)
        # N*512*64*64

        d2_decod_2 = torch.cat((x1_decod_2,d2_decod_2),dim=1)
        # N*1024*64*64
        d2_decod_2=self.Up_conv2_decod_2(d2_decod_2)

        d1_encod_1=self.Conv_1x1_decod_1(d2_decod_1)
        d1_encod_2=self.Conv_1x1_decod_2(d2_decod_2)
        attention_maps=[psi4,psi3,psi1,psi1]


        return torch.cat((d1_encod_1,d1_encod_2),dim=1),attention_maps

