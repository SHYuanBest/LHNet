import torch
import torch.nn as nn
from utils_modules import SwinTransformer, FIF_Module, FMI_Module, PPM, Downsample, Upsample
import torch.nn.functional as F
import torch
class LHNet(nn.Module):
    """Implementation of LHNet from Shenghai Yuan et al. (ACM MM 2023)."""
    def __init__(self, in_channels=3, out_channels=3,bias=False):

        super(LHNet, self).__init__()
        self.embedding_dim = 3
        self.conv1 = nn.Conv2d(256*3, 256, 3, stride=1, padding=1)
        self.FIF_1 = FIF_Module(384,256)
        self.conv2 = nn.Conv2d(128*3, 128, 3, stride=1, padding=1)
        self.FIF_2 = FIF_Module(192,128)
        self.conv3 = nn.Conv2d(64*3, 64, 3, stride=1, padding=1)
        self.FIF_3 = FIF_Module(96, 64)

        self.in_chans = 3
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ReLU=nn.LeakyReLU(inplace=True)
        self.IN_1=nn.InstanceNorm2d(48, affine=False)
        self.IN_2=nn.InstanceNorm2d(96, affine=False)
        self.IN_3=nn.InstanceNorm2d(192, affine=False)

        # PPM is used to expand the amount of model parameters
        self.PPM1 = PPM(32, 8, bins=(1, 2, 3, 4))
        self.PPM2 = PPM(64, 16, bins=(1, 2, 3, 4))
        self.PPM3 = PPM(128, 32, bins=(1, 2, 3, 4))
        self.PPM4 = PPM(256, 64, bins=(1, 2, 3, 4))

        # 27,565,242
        self.swin_1 = SwinTransformer(pretrain_img_size=224,
                                    patch_size=2,
                                    in_chans=3,
                                    embed_dim=96,
                                    depths=[2, 2, 2],
                                    num_heads=[3, 6, 12], 
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True, 
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0., 
                                    drop_path_rate=0.2,
                                    norm_layer=nn.LayerNorm, 
                                    ape=False,
                                    patch_norm=True,
                                    out_indices=(0, 1, 2),
                                    frozen_stages=-1,
                                    use_checkpoint=False)

        self.E_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(32, affine=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            Downsample(32))

        self.E_block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(64, affine=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            Downsample(64))

        self.E_block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(128, affine=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            Downsample(128))
        
        self.E_block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(256, affine=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            Downsample(256))
        
        self.E_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(512, affine=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            Downsample(512))


        self._block1 = Upsample(512)
        # nn.Sequential(
        #     nn.Conv2d(512, 256, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingBilinear2d(scale_factor=2))

        self._block2 = Upsample(512)
        # nn.Sequential(
        #     nn.Conv2d(512, 256, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingBilinear2d(scale_factor=2))

        self._block3 = Upsample(256)
        # nn.Sequential(
        #     nn.Conv2d(256, 128, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingBilinear2d(scale_factor=2))
        
        self._block4 = Upsample(128)
        # nn.Sequential(
        #     nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingBilinear2d(scale_factor=2))
        
        self._block5= Upsample(64)
        # nn.Sequential(
        #     nn.Conv2d(64, 32, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     #nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(32, 32, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingBilinear2d(scale_factor=2))
            
        self._block6= Upsample(46)
        # nn.Sequential(
        #     nn.Conv2d(46, 23, 3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     #nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(23, 23, 3, stride=1, padding=1),
        #     nn.UpsamplingBilinear2d(scale_factor=2))
            
        # self._block7= nn.Sequential(
        #     nn.Conv2d(32, 32, 3, stride=1, padding=1),
        #     nn.LeakyReLU(inplace=True),
        #     #nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
        #     )

        self.Multi_fussion_en_1 = FMI_Module( c=64)
        self.Multi_fussion_en_2 = FMI_Module(c=128)
        self.Multi_fussion_en_3 = FMI_Module(c=256)

        self.Multi_fussion_dn_1 = FMI_Module(c=128)
        self.Multi_fussion_dn_2 = FMI_Module(c=64)
        self.Multi_fussion_dn_3 = FMI_Module(c=32)
        
        self.Up8 = nn.Sequential(Upsample(256),
                                Upsample(128),
                                Upsample(64))
        self.Up4 = nn.Sequential(Upsample(128),
                                Upsample(64))
        self.Up2 = nn.Sequential(Upsample(64))
        self.fina_block = nn.Sequential(
                    nn.Conv2d(32 * 4, 32, 3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
                    )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        swin_in = x    #96,192,384,768
        swin_out_1=self.swin_1(swin_in)    # [B, 96, 112, 112]、[B, 192, 56, 56]、[B, 384, 28, 28]
        # Encoder
        swin_input_1 = self.E_block1(swin_in)  # 32
        swin_input_1 = self.PPM1(swin_input_1)
        swin_input_2 = self.E_block2(swin_input_1)  # 64
        swin_input_2 = self.PPM2(swin_input_2)
        swin_input_3 = self.E_block3(swin_input_2)  # 128
        swin_input_3 = self.PPM3(swin_input_3)
        swin_input_4 = self.E_block4(swin_input_3)  # 256
        swin_input_4 = self.PPM4(swin_input_4)

        # Multi_scale_fussion_en
        fussion_en_1, fussion_en_2 = self.Multi_fussion_en_1([swin_input_1, swin_input_2])
        fussion_en_2, fussion_en_3 = self.Multi_fussion_en_2([fussion_en_2, swin_input_3])
        fussion_en_3, fussion_en_4 = self.Multi_fussion_en_3([fussion_en_3, swin_input_4])

        # Decoder
        upsample1 = self._block1(swin_input_4)#256

        swin_input_3_refine = self.FIF_1(self.IN_3(swin_input_3),swin_out_1[2])
        swin_input_3 = fussion_en_3
        concat3 = torch.cat((swin_input_3,swin_input_3_refine,upsample1), dim=1)#256+256+256=768
        decoder_3 = self.ReLU(self.conv1(concat3)) #256
        upsample3 = self._block3(decoder_3)#128

        swin_input_2_refine = self.FIF_2(self.IN_2(swin_input_2),swin_out_1[1])
        swin_input_2 = fussion_en_2
        concat2 = torch.cat((swin_input_2,swin_input_2_refine,upsample3), dim=1)#128+128+128=384
        decoder_2 = self.ReLU(self.conv2(concat2))#128
        upsample4 = self._block4(decoder_2)#64

        swin_input_1_refine = self.FIF_3(self.IN_3(swin_input_1),swin_out_1[0])
        swin_input_1 = fussion_en_1
        concat1 = torch.cat((swin_input_1,swin_input_1_refine,upsample4), dim=1)#64+64+64=192
        decoder_1 = self.ReLU(self.conv3(concat1))#64
        upsample5 = self._block5(decoder_1)#32

        # Multi_scale_fussion_de
        upsample3, upsample1 = self.Multi_fussion_dn_1([upsample3, upsample1])
        upsample4, upsample3 = self.Multi_fussion_dn_2([upsample4, upsample3])
        upsample5, upsample4 = self.Multi_fussion_dn_3([upsample5, upsample4])
        out_8x = self.Up8(upsample1)
        out_4x = self.Up4(upsample3)
        out_2x = self.Up2(upsample4)
        result = torch.cat((upsample5, out_2x, out_4x, out_8x), dim=1)
        result = self.fina_block(result)
        
        return result

# net = LHNet(3,3)
# net(torch.rand((2,3,224,224)))

# from torchinfo import summary
# net = LHNet(3,3)
# summary(net,col_names = ("input_size", "output_size", "num_params"),input_size=(1,3,224,224),depth=9,device='cpu')