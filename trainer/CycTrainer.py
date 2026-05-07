import itertools
from functools import partial

import wandb
import torch.cuda
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import ReplayBuffer, pad_to_size
from .datasets import ImageDataset,ValDataset
from Model.CycleGan import *
from .utils import smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine,ToPILImage, Normalize, ToTensor, Resize, Lambda
from .transformer import Transformer_2D
from skimage import measure
from skimage.metrics import normalized_mutual_information
import numpy as np
import cv2

class Cyc_Trainer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ## def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).to(self.device)
        self.netD_B = Discriminator(config['input_nc']).to(self.device)
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'],
                                              betas=(0.5, 0.999))

        if config['regist']:
            self.R_A = Reg(config['size'], config['size'], config['input_nc'],
                           config['input_nc']).to(self.device)
            self.spatial_transform = Transformer_2D().to(self.device)
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'],
                                                  betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = Generator(config['input_nc'], config['output_nc']).to(self.device)
            self.netD_A = Discriminator(config['input_nc']).to(self.device)
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'],
                                                  betas=(0.5, 0.999))

        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'],
                                                betas=(0.5, 0.999))

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        self.input_A = torch.empty(config['batchSize'], config['input_nc'], config['size'],
                                   config['size'], device=self.device)
        self.input_B = torch.empty(config['batchSize'], config['output_nc'], config['size'],
                                   config['size'], device=self.device)
        self.target_real = Variable(torch.ones(1, 1, device=self.device), requires_grad=False)
        self.target_fake = Variable(torch.zeros(1, 1, device=self.device), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loader
        level = config['noise_level']  # set noise level
        
        transforms_1 = [
            ToPILImage(),
            RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level]),
            ToTensor(),
            Normalize(mean=(0.5,), std=(0.5,)),
            Lambda(partial(pad_to_size, size=config['size'], fill=-1)),
            Resize(size=(config['size'], config['size']))
        ]
    
        transforms_2 = [
            ToPILImage(),
            RandomAffine(degrees=1,translate=[0.02, 0.02],scale=[0.98, 1.02]),
            ToTensor(),
            Normalize(mean=(0.5,), std=(0.5,)),
            Lambda(partial(pad_to_size, size=config['size'], fill=-1)),
            Resize(size=(config['size'], config['size']))
        ]

        self.dataloader = DataLoader(ImageDataset(config['dataroot'], level, transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False,),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])

        val_transforms = [ToTensor(),
                          Resize(size = (config['size'], config['size']))]
        
        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_ =val_transforms, unaligned=False),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

 
       # Loss plot
        self.logger = Logger(config['n_epochs'], len(self.dataloader))
        
    def train(self):
        ###### Training ######
        start_epoch = self.config['epoch']
        steps_per_epoch = len(self.dataloader)
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                global_step = (epoch - start_epoch) * steps_per_epoch + i
                batch['A'] = batch['A'].to(self.device)
                batch['B'] = batch['B'].to(self.device)
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                if self.config['bidirect']:   # C dir
                    if self.config['regist']:    #C + R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        
                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        
                        Trans = self.R_A(fake_B,real_B) 
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        
                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss +SM_loss
                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()
                        
                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ################################### 
                    
                    else: #only  dir:  C
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda']*self.MSE_loss(pred_fake, self.target_real)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                        loss_Total.backward()
                        self.optimizer_G.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################
                        
                        
                        
                else:                  # s dir :NC
                    if self.config['regist']:    # NC+R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        fake_B = self.netG_A2B(real_A)
                        Trans = self.R_A(fake_B,real_B) 
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        toal_loss = SM_loss+adv_loss+SR_loss
                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B = self.netG_A2B(real_A)
                        pred_fake0 = self.netD_B(fake_B)
                        pred_real = self.netD_B(real_B)
                        loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake)+self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)


                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        
                        
                        
                    else:        # only NC
                        self.optimizer_G.zero_grad()
                        fake_B = self.netG_A2B(real_A)
                        #### GAN aligin loss
                        pred_fake = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        adv_loss.backward()
                        self.optimizer_G.step()
                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()
                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        ###################################


                self.logger.log(losses={'loss_D_B': loss_D_B, 'SR_loss': SR_loss},
                                iteration=global_step,
                                )

    #         # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            ckpt_path = self.config['save_root'] + 'netG_A2B.pth'
            torch.save(self.netG_A2B.state_dict(), ckpt_path)
            if wandb.run is not None:
                wandb.save(ckpt_path, base_path=self.config['save_root'], policy='now')

            #############val###############
            val_step = (epoch - start_epoch + 1) * steps_per_epoch
            with torch.no_grad():
                NMI_sum = 0
                num = 0
                val_images = {'real_A': [], 'real_B': [], 'fake_B': []}
                for i, batch in enumerate(self.val_data):
                    batch['A'] = batch['A'].to(self.device)
                    batch['B'] = batch['B'].to(self.device)
                    real_A_t = Variable(self.input_A.copy_(batch['A']))
                    real_B_t = Variable(self.input_B.copy_(batch['B']))
                    fake_B_t = self.netG_A2B(real_A_t)

                    real_B = real_B_t.detach().cpu().numpy().squeeze()
                    fake_B = fake_B_t.detach().cpu().numpy().squeeze()
                    nmi = normalized_mutual_information(real_B, fake_B)
                    NMI_sum += nmi
                    num += 1

                    val_images['real_A'].append(real_A_t.detach().cpu())
                    val_images['real_B'].append(real_B_t.detach().cpu())
                    val_images['fake_B'].append(fake_B_t.detach().cpu())

                val_nmi = NMI_sum / num
                print('Val NMI:', val_nmi)

                if wandb.run is not None:
                    log_dict = {'val/NMI': val_nmi, 'epoch': epoch}
                    for i, (name, tensors) in enumerate(val_images.items()):
                        imgs = []
                        for t in tensors:
                            for sample_idx in range(t.shape[0]):
                                arr = (((t[sample_idx] + 1) / 2) * 255).numpy().astype('uint8')
                                imgs.append(wandb.Image(arr))
                        log_dict[f'val/{val_step}/{name}_{i}'] = imgs
                    wandb.log(log_dict, step=val_step)
                
                    
                         
    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        #self.R_A.load_state_dict(torch.load(self.config['save_root'] + 'Regist.pth'))
        with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    
                    fake_B = self.netG_A2B(real_A)
                    fake_B = fake_B.detach().cpu().numpy().squeeze()                                                 
                    mae = self.MAE(fake_B,real_B)
                    psnr = self.PSNR(fake_B,real_B)
                    ssim = measure.compare_ssim(fake_B,real_B)
                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim 
                    num += 1
                print ('MAE:',MAE/num)
                print ('PSNR:',PSNR/num)
                print ('SSIM:',SSIM/num)
    
    def PSNR(self,fake,real):
       x,y = np.where(real!= -1)# Exclude background
       mse = np.mean(((fake[x,y]+1)/2. - (real[x,y]+1)/2.) ** 2 )
       if mse < 1.0e-10:
          return 100
       else:
           PIXEL_MAX = 1
           return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
            
    def MAE(self,fake,real):
        x,y = np.where(real!= -1)  # Exclude background
        mae = np.abs(fake[x,y]-real[x,y]).mean()
        return mae/2     #from (-1,1) normaliz  to (0,1)            

    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy) 
