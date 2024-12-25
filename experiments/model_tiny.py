
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
import math
import numpy as np
import random 

def kde(x, std = 0.1):
    # use a gaussian kernel to estimate density
    x = x.half() # Do it in half precision TODO: remove hardcoding
    scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

class BasicLayer(nn.Module):
    """
        Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, relu = True):
        super().__init__()
        self.layer = nn.Sequential(
                                        nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
                                        nn.BatchNorm2d(out_channels, affine=False),
                                        nn.ReLU(inplace = True) if relu else nn.Identity()
                                    )

    def forward(self, x):
        return self.layer(x)

def gather_sample(input, grid):
    _,C,Hi,Wi = input.shape
    N,H,W,_ = grid.shape
    grid[..., 0] *= Wi/2
    grid[..., 0] += Wi/2
    grid[..., 1] *= Hi/2
    grid[..., 1] += Hi/2

    grid[...,0] = torch.clamp(grid[...,0], min=0, max=Wi-1)
    grid[...,1] = torch.clamp(grid[...,1], min=0, max=Hi-1)
    grid = grid.long()
    idx = grid[..., 1]*Wi + grid[...,0]      # N,H,W
    
    idx = idx.view(N,1,H*W).expand(N,C,H*W).long()
    input = input.view(N,C,-1)
    output = torch.gather(input, 2, idx)
    output = output.view(N,C,H,W)
    return output


def gather(input, idx):
    N,H,W = idx.shape
    _,C,_,_ = input.shape
    idx = idx.view(N,1,H*W).expand(N,C,H*W).long()
    input = input.view(N,C,-1)
    output = torch.gather(input, 2, idx)
    output = output.view(N,C,H,W)

    return output

def gather_bilinear(input, grid):
    _,C,Hi,Wi = input.shape
    N,H,W,_ = grid.shape

    pad = (grid<-1)|(grid>1)
    grid[pad] = 0

    grid[..., 0] *= Wi/2
    grid[..., 0] += Wi/2
    grid[..., 1] *= Hi/2
    grid[..., 1] += Hi/2

    grid_x = grid[...,0]
    grid_y = grid[...,1]

    
    grid_x0 = torch.floor(grid_x)
    grid_x1 = grid_x0+1
    

    grid_y0 = torch.floor(grid_y)
    grid_y1 = grid_y0+1

    grid_x = torch.clamp(grid_x, min=0, max=Wi-1).long()
    grid_y = torch.clamp(grid_y, min=0, max=Wi-1).long()
    grid_x0 = torch.clamp(grid_x0, min=0, max=Wi-1).long()
    grid_x1 = torch.clamp(grid_x1, min=0, max=Wi-1).long()
    grid_y0 = torch.clamp(grid_y0, min=0, max=Hi-1).long()
    grid_y1 = torch.clamp(grid_y1, min=0, max=Hi-1).long()

    idx00 = grid_x0 + grid_y0*Wi        # N,H,W
    idx01 = grid_x0 + grid_y1*Wi 
    idx10 = grid_x1 + grid_y0*Wi 
    idx11 = grid_x1 + grid_y1*Wi 
    
    out00 = gather(input, idx00)
    out01 = gather(input, idx01)
    out10 = gather(input, idx10)
    out11 = gather(input, idx11)

    w0 = (grid_x1 - grid_x).view(N,1,H,W)
    w1 = (grid_x - grid_x0).view(N,1,H,W)

    w2 = (grid_y1 - grid_y).view(N,1,H,W)
    w3 = (grid_y - grid_y0)

    value0 = w0*out00 + w1*out10
    value1 = w0*out01 + w1*out11 

    output = w2*value0 + w3*value1

    return output


class TinyRoma(nn.Module):
    """
        Implementation of architecture described in 
        "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self, xfeat = None, 
                 freeze_xfeat = True, 
                 sample_mode = "threshold_balanced", 
                 symmetric = False, 
                 exact_softmax = False):
        super().__init__()
        if xfeat is None:
            xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096).net
            del xfeat.heatmap_head, xfeat.keypoint_head, xfeat.fine_matcher
        if freeze_xfeat:
            xfeat.train(False)
            self.xfeat = [xfeat]# hide params from ddp
        else:
            self.xfeat = nn.ModuleList([xfeat])
        self.freeze_xfeat = freeze_xfeat
        match_dim = 256
        self.coarse_matcher = nn.Sequential(
            BasicLayer(64+64+2, match_dim,),
            BasicLayer(match_dim, match_dim,), 
            BasicLayer(match_dim, match_dim,), 
            BasicLayer(match_dim, match_dim,), 
            nn.Conv2d(match_dim, 3, kernel_size=1, bias=True, padding=0))
        fine_match_dim = 64
        self.fine_matcher = nn.Sequential(
            BasicLayer(24+24+2, fine_match_dim,),
            BasicLayer(fine_match_dim, fine_match_dim,), 
            BasicLayer(fine_match_dim, fine_match_dim,), 
            BasicLayer(fine_match_dim, fine_match_dim,), 
            nn.Conv2d(fine_match_dim, 3, kernel_size=1, bias=True, padding=0),)
        self.sample_mode = sample_mode
        self.sample_thresh = 0.2
        self.symmetric = symmetric
        self.exact_softmax = exact_softmax

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv2_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv2_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    @property
    def device(self):
        return self.fine_matcher[-1].weight.device
    
    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        H, W = x.shape[-2:]
        _H, _W = (H//32) * 32, (W//32) * 32
        rh, rw = H/_H, W/_W

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw        
    
    def forward_single(self, x):
        with torch.inference_mode(self.freeze_xfeat or not self.training):
            xfeat = self.xfeat[0]
            with torch.no_grad():
                x = x.mean(dim=1, keepdim = True)
                x = xfeat.norm(x)
                # x = (x-x.mean())/x.std()

            #main backbone
            x1 = xfeat.block1(x)
            x2 = xfeat.block2(x1 + xfeat.skip1(x))
            x3 = xfeat.block3(x2)
            x4 = xfeat.block4(x3)
            x5 = xfeat.block5(x4)
            # x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
            x4 = self.upconv1(x4)
            # x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
            x5 = self.upconv2_1(x5)
            x5 = self.upconv2_2(x5)
            feats = xfeat.block_fusion( x3 + x4 + x5 )
        if self.freeze_xfeat:
            return x2.clone(), feats.clone()
        return x2, feats

    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
    
    def pos_embed(self, corr_volume: torch.Tensor):
        B, H1, W1, H0, W0 = corr_volume.shape 
        grid = torch.stack(
                torch.meshgrid(
                    torch.linspace(-1+1/W1,1-1/W1, W1), 
                    torch.linspace(-1+1/H1,1-1/H1, H1), 
                    indexing = "xy"), 
                dim = -1).float().to(corr_volume).reshape(H1*W1, 2)
        down = 4
        # if not self.training and not self.exact_softmax:
        if False:
            grid_lr = torch.stack(
                torch.meshgrid(
                    torch.linspace(-1+down/W1,1-down/W1, W1//down), 
                    torch.linspace(-1+down/H1,1-down/H1, H1//down), 
                    indexing = "xy"), 
                dim = -1).float().to(corr_volume).reshape(H1*W1 //down**2, 2)
            cv = corr_volume
            best_match = cv.reshape(B,H1*W1,H0,W0).argmax(dim=1) # B, HW, H, W
            P_lowres = torch.cat((cv[:,::down,::down].reshape(B,H1*W1 // down**2,H0,W0), best_match[:,None]),dim=1).softmax(dim=1)
            pos_embeddings = torch.einsum('bchw,cd->bdhw', P_lowres[:,:-1], grid_lr)
            pos_embeddings += P_lowres[:,-1] * grid[best_match].permute(0,3,1,2)
        else:
            # P = corr_volume.reshape(B,H1*W1,H0,W0).softmax(dim=1) # B, HW, H, W
            # pos_embeddings = torch.einsum('bchw,cd->bdhw', P, grid)

            P = corr_volume.reshape(B,H1*W1,H0,W0).softmax(dim=1) # B, HW, H, W
            # pos_embeddings = torch.einsum('bchw,cd->bdhw', P, grid)
            P = P.permute(0,2,3,1)
            # pos_embeddings = torch.matmul(P, grid).permute(0,3,1,2)

            a=grid[:,0:1].view(-1)
            b=grid[:,1:2].view(-1)
            p0 = torch.sum(P*a, -1, keepdim=True)
            p1 = torch.sum(P*b, -1, keepdim=True)
            pos_embeddings = torch.cat((p0,p1), -1)
            pos_embeddings = pos_embeddings.permute(0,3,1,2)

        return pos_embeddings
    
    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                       im_A_path = None, im_B_path = None, symmetric = True, save_path = None, unnormalize = False):
        device = warp.device
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = F.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im
     
    def corr_volume(self, feat0, feat1):
        """
            input:
                feat0 -> torch.Tensor(B, C, H, W)
                feat1 -> torch.Tensor(B, C, H, W)
            return:
                corr_volume -> torch.Tensor(B, H, W, H, W)
        """
        B, C, H0, W0 = feat0.shape
        B, C, H1, W1 = feat1.shape
        feat0 = feat0.view(B, C, H0*W0)
        feat1 = feat1.view(B, C, H1*W1)
        corr_volume = torch.einsum('bci,bcj->bji', feat0, feat1).reshape(B, H1, W1, H0 , W0)/math.sqrt(C) #16*16*16
        return corr_volume
    
    @torch.inference_mode()
    def match_from_path(self, im0_path, im1_path):
        device = self.device
        im0 = ToTensor()(Image.open(im0_path))[None].to(device)
        im1 = ToTensor()(Image.open(im1_path))[None].to(device)
        return self.match(im0, im1, batched = False)
    
    @torch.inference_mode()
    def match(self, im0, im1, *args, batched = True):
        # stupid
        if isinstance(im0, (str, Path)):
            return self.match_from_path(im0, im1)
        elif isinstance(im0, Image.Image):
            batched = False
            device = self.device
            im0 = ToTensor()(im0)[None].to(device)
            im1 = ToTensor()(im1)[None].to(device)
 
        B,C,H0,W0 = im0.shape
        B,C,H1,W1 = im1.shape
        self.train(False)
        corresps = self.forward({"im_A":im0, "im_B":im1})
        #return 1,1
        flow = F.interpolate(
            corresps[8]["flow"], 
            size = (H0, W0), 
            mode = "bilinear", align_corners = False).permute(0,2,3,1).reshape(B,H0,W0,2)
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1+1/W0,1-1/W0, W0), 
                torch.linspace(-1+1/H0,1-1/H0, H0), 
                indexing = "xy"), 
            dim = -1).float().to(flow.device).expand(B, H0, W0, 2)
        
        certainty = F.interpolate(corresps[8]["certainty"], size = (H0,W0), mode = "bilinear", align_corners = False)
        warp, cert = torch.cat((grid, flow), dim = -1), certainty[:,0].sigmoid()
        if batched:
            return warp, cert
        else:
            return warp[0], cert[0]

    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]
            
    def forward(self, batch):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:

        """
        im0 = batch["im_A"]
        im1 = batch["im_B"]
        corresps = {}
        im0, rh0, rw0 = self.preprocess_tensor(im0)
        im1, rh1, rw1 = self.preprocess_tensor(im1)
        B, C, H0, W0 = im0.shape
        B, C, H1, W1 = im1.shape
        to_normalized = torch.tensor((2/W1, 2/H1, 1)).to(im0.device)[None,:,None,None]
 
        if im0.shape[-2:] == im1.shape[-2:]:
            x = torch.cat([im0, im1], dim=0)
            x = self.forward_single(x)
            feats_x0_c, feats_x1_c = x[1].chunk(2)
            feats_x0_f, feats_x1_f = x[0].chunk(2)
        else:
            feats_x0_f, feats_x0_c = self.forward_single(im0)
            feats_x1_f, feats_x1_c = self.forward_single(im1)
        corr_volume = self.corr_volume(feats_x0_c, feats_x1_c)
        coarse_warp = self.pos_embed(corr_volume)
        coarse_matches = torch.cat((coarse_warp, torch.zeros_like(coarse_warp[:,-1:])), dim=1)
        feats_x1_c_warped = F.grid_sample(feats_x1_c, coarse_matches.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        coarse_matches_delta = self.coarse_matcher(torch.cat((feats_x0_c, feats_x1_c_warped, coarse_warp), dim=1))
        coarse_matches = coarse_matches + coarse_matches_delta * to_normalized
        corresps[8] = {"flow": coarse_matches[:,:2], "certainty": coarse_matches[:,2:]}
        return corresps
        # coarse_matches_up = F.interpolate(coarse_matches, size = feats_x0_f.shape[-2:], mode = "bilinear", align_corners = False)
        # # coarse_matches_up = self.upconv3(coarse_matches)

        # coarse_matches_up_detach = coarse_matches_up.detach()#note the detach
        # feats_x1_f_warped = F.grid_sample(feats_x1_f, coarse_matches_up_detach.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        # fine_matches_delta = self.fine_matcher(torch.cat((feats_x0_f, feats_x1_f_warped, coarse_matches_up_detach[:,:2]), dim=1))
        # # fine_matches_delta = self.fine_matcher(torch.cat((feats_x0_f, feats_x1_f, coarse_matches_up_detach[:,:2]), dim=1))
        # fine_matches = coarse_matches_up_detach+fine_matches_delta * to_normalized
        # corresps[4] = {"flow": fine_matches[:,:2], "certainty": fine_matches[:,2:]}
        # return corresps
    

class TinyRoMaExport(nn.Module):
    """
        Implementation of architecture described in 
        "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self, xfeat = None, 
                 freeze_xfeat = True, 
                 sample_mode = "threshold_balanced", 
                 symmetric = False, 
                 exact_softmax = False):
        super().__init__()
        if xfeat is None:
            xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096).net
            del xfeat.heatmap_head, xfeat.keypoint_head, xfeat.fine_matcher
        if freeze_xfeat:
            xfeat.train(False)
            self.xfeat = [xfeat]# hide params from ddp
        else:
            self.xfeat = nn.ModuleList([xfeat])
        self.freeze_xfeat = freeze_xfeat
        match_dim = 256
        self.coarse_matcher = nn.Sequential(
            BasicLayer(64+64+2, match_dim,),
            BasicLayer(match_dim, match_dim,), 
            BasicLayer(match_dim, match_dim,), 
            BasicLayer(match_dim, match_dim,), 
            nn.Conv2d(match_dim, 3, kernel_size=1, bias=True, padding=0))
        fine_match_dim = 64
        self.fine_matcher = nn.Sequential(
            BasicLayer(24+24+2, fine_match_dim,),
            BasicLayer(fine_match_dim, fine_match_dim,), 
            BasicLayer(fine_match_dim, fine_match_dim,), 
            BasicLayer(fine_match_dim, fine_match_dim,), 
            nn.Conv2d(fine_match_dim, 3, kernel_size=1, bias=True, padding=0),)
        self.sample_mode = sample_mode
        self.sample_thresh = 0.05
        self.symmetric = symmetric
        self.exact_softmax = exact_softmax

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv2_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv2_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(1)

    def forward_single(self, x):
        with torch.inference_mode(self.freeze_xfeat or not self.training):
            xfeat = self.xfeat[0]
            with torch.no_grad():
                x = x.mean(dim=1, keepdim = True)
                # x = xfeat.norm(x)
                # x = (x-x.mean())/x.std()
                x = self.bn(x)

            #main backbone
            x1 = xfeat.block1(x)
            x2 = xfeat.block2(x1 + xfeat.skip1(x))
            x3 = xfeat.block3(x2)
            x4 = xfeat.block4(x3)
            x5 = xfeat.block5(x4)
            print("x3.shape",x3.shape)
            print("x4",x4.shape)
            print("x5",x5.shape)
            # x3_shape = [1, 64, 120, 28]
            # x4 = F.interpolate(x4, size=(x3.shape[-2], x3.shape[-1]), mode='bilinear')
            # x5 = F.interpolate(x5, size=(x3.shape[-2], x3.shape[-1]), mode='bilinear')
            x4 = self.upconv1(x4)
            x5 = self.upconv2_1(x5)
            x5 = self.upconv2_2(x5)
            print("x4",x4.shape)
            print("x5",x5.shape)
            feats = xfeat.block_fusion( x3 + x4 + x5 )
        if self.freeze_xfeat:
            return x2.clone(), feats.clone()
        return x2, feats

    def corr_volume(self, feat0, feat1):
        """
            input:
                feat0 -> torch.Tensor(B, C, H, W)
                feat1 -> torch.Tensor(B, C, H, W)
            return:
                corr_volume -> torch.Tensor(B, H, W, H, W)
        """
        B, C, H0, W0 = feat0.shape
        B, C, H1, W1 = feat1.shape
        feat0 = feat0.view(B, C, H0*W0)
        feat1 = feat1.view(B, C, H1*W1)
        # corr_volume = torch.einsum('bci,bcj->bji', feat0, feat1).reshape(B, H1, W1, H0 , W0)/math.sqrt(C) #16*16*16
        feat1 = feat1.permute(0,2,1)  # bjc
        corr_volume = torch.matmul(feat1, feat0).reshape(B, H1, W1, H0 , W0)/math.sqrt(C)  # bjc matmul bci -> bji
        return corr_volume
    
    def pos_embed(self, corr_volume, grid):
        B, H1, W1, H0, W0 = corr_volume.shape 
        print("corr_volume",corr_volume.shape)
        # grid = torch.stack(
        #         torch.meshgrid(
        #             torch.linspace(-1+1/W1,1-1/W1, W1), 
        #             torch.linspace(-1+1/H1,1-1/H1, H1), 
        #             indexing = "xy"), 
        #         dim = -1).float().to(corr_volume).reshape(H1*W1, 2)

        down = 4
        # if not self.training and not self.exact_softmax:
        if False:
            # grid_lr = torch.stack(
            #     torch.meshgrid(
            #         torch.linspace(-1+down/W1,1-down/W1, W1//down), 
            #         torch.linspace(-1+down/H1,1-down/H1, H1//down), 
            #         indexing = "xy"), 
            #     dim = -1).float().to(corr_volume).reshape(H1*W1 //down**2, 2)
            cv = corr_volume
            best_match = cv.reshape(B,H1*W1,H0,W0).argmax(dim=1) # B, HW, H, W
            cv_ = cv.reshape(B,H1*W1,H0,W0)
            print("best_match[:,None]",best_match[:,None].shape)
            # torch.save(best_match, "best_match.pth")
            # print("cv[best_match[:,None]]",cv[best_match[:,None]].shape)
            P_lowres = torch.cat((cv[:,::down,::down].reshape(B,H1*W1 // down**2,H0,W0), torch.zeros_like(best_match[:,None])),dim=1).softmax(dim=1)
            # pos_embeddings = torch.einsum('bchw,cd->bdhw', P_lowres[:,:-1], grid_lr)

            # a = P_lowres[:,:-1]   # bchw 
            # a = a.permute(0,2,3,1) # bhwc
            a = P_lowres.permute(0,2,3,1)[..., :-1]
            pos_embeddings = torch.matmul(a, grid_lr)   # bhwc matmul cd -> bhwd
            pos_embeddings = pos_embeddings.permute(0,3,1,2)
            print("grid[best_match]",grid[best_match].shape)
            print("P_lowres[:,-1]",P_lowres[:,-1].shape)
            tmp = P_lowres[:,-1] * grid[best_match].permute(0,3,1,2)
            print("tmp",tmp.shape)
            pos_embeddings += tmp
            #print("hej")
        else:
            P = corr_volume.reshape(B,H1*W1,H0,W0).softmax(dim=1) # B, HW, H, W
            # pos_embeddings = torch.einsum('bchw,cd->bdhw', P, grid)
            P = P.permute(0,2,3,1)                                  # B,H,W,HW
            # pos_embeddings = torch.matmul(P, grid).permute(0,3,1,2)     # build RecursionError: maximum recursion depth exceeded while calling a Python object

            a=grid[:,0:1].view(-1)
            b=grid[:,1:2].view(-1)
            p0 = torch.sum(P*a, -1, keepdim=True)
            p1 = torch.sum(P*b, -1, keepdim=True)
            pos_embeddings = torch.cat((p0,p1), -1)
            pos_embeddings = pos_embeddings.permute(0,3,1,2)
        return pos_embeddings
    
    def forward(self, im0, im1, grid, to_normalized):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:

        """
        
        corresps = {}

        
        feats_x0_f, feats_x0_c = self.forward_single(im0)
        feats_x1_f, feats_x1_c = self.forward_single(im1)
        
        corr_volume = self.corr_volume(feats_x0_c, feats_x1_c)
        
        coarse_warp = self.pos_embed(corr_volume, grid)
        
        coarse_matches = torch.cat((coarse_warp, torch.zeros_like(coarse_warp[:,-1:])), dim=1)
        
        feats_x1_c_warped = F.grid_sample(feats_x1_c, coarse_matches.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        # feats_x1_c_warped = bilinear_grid_sample(feats_x1_c, coarse_matches.permute(0, 2, 3, 1)[...,:2])
        coarse_matches_delta = self.coarse_matcher(torch.cat((feats_x0_c, feats_x1_c_warped, coarse_warp), dim=1))
        coarse_matches = coarse_matches + coarse_matches_delta * to_normalized
        corresps[8] = {"flow": coarse_matches[:,:2], "certainty": coarse_matches[:,2:]}
        return coarse_matches
        # coarse_matches_up = F.interpolate(coarse_matches, size = feats_x0_f.shape[-2:], mode = "bilinear", align_corners = False)        
        coarse_matches_up = self.upconv3(coarse_matches)
        coarse_matches_up_detach = coarse_matches_up.detach()#note the detach
        # feats_x1_f_warped = F.grid_sample(feats_x1_f, coarse_matches_up_detach.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        # feats_x1_f_warped = bilinear_grid_sample(feats_x1_f, coarse_matches_up_detach.permute(0, 2, 3, 1)[...,:2])
        # fine_matches_delta = self.fine_matcher(torch.cat((feats_x0_f, feats_x1_f_warped, coarse_matches_up_detach[:,:2]), dim=1))
        fine_matches_delta = self.fine_matcher(torch.cat((feats_x0_f, feats_x1_f, coarse_matches_up_detach[:,:2]), dim=1))
        fine_matches = coarse_matches_up_detach+fine_matches_delta * to_normalized
        # corresps[4] = {"flow": fine_matches[:,:2], "certainty": fine_matches[:,2:]}
        # return corresps
        return fine_matches



class TinyRoMaExport1(TinyRoMaExport):
    def __init__(self, xfeat = None, 
                 freeze_xfeat = False, 
                 sample_mode = "threshold_balanced", 
                 symmetric = False, 
                 exact_softmax = False):
        super().__init__(xfeat=xfeat, freeze_xfeat=freeze_xfeat, sample_mode=sample_mode, symmetric=symmetric,exact_softmax=exact_softmax)

        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(24, 24, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, im0, im1, grid, to_normalized):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:

        """
        
        corresps = {}

        
        feats_x0_f, feats_x0_c = self.forward_single(im0)
        feats_x1_f, feats_x1_c = self.forward_single(im1)
        
        corr_volume = self.corr_volume(feats_x0_c, feats_x1_c)
        
        coarse_warp = self.pos_embed(corr_volume, grid)
        
        coarse_matches = torch.cat((coarse_warp, torch.zeros_like(coarse_warp[:,-1:])), dim=1)
        
        feats_x1_c_warped = F.grid_sample(feats_x1_c, coarse_matches.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        # feats_x1_c_warped = bilinear_grid_sample(feats_x1_c, coarse_matches.permute(0, 2, 3, 1)[...,:2])
        coarse_matches_delta = self.coarse_matcher(torch.cat((feats_x0_c, feats_x1_c_warped, coarse_warp), dim=1))
        coarse_matches = coarse_matches + coarse_matches_delta * to_normalized
        # corresps[8] = {"flow": coarse_matches[:,:2], "certainty": coarse_matches[:,2:]}
        # return coarse_matches
               
        # coarse_matches_up = self.upconv3(coarse_matches)
        # coarse_matches_up_detach = coarse_matches_up.detach()#note the detach
        # feats_x1_f_warped = F.grid_sample(feats_x1_f, coarse_matches_up_detach.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        # fine_matches_delta = self.fine_matcher(torch.cat((feats_x0_f, feats_x1_f_warped, coarse_matches_up_detach[:,:2]), dim=1))
        # fine_matches = coarse_matches_up_detach+fine_matches_delta * to_normalized

        # coarse_matches_detach = coarse_matches.detach()
        # feats_x1_f_warped = F.grid_sample(feats_x1_f, coarse_matches_detach.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        # coarse_matches_up_detach = self.upconv3(coarse_matches_detach)
        # feats_x1_f_warped = self.upconv4(feats_x1_f_warped)
        # fine_matches_delta = self.fine_matcher(torch.cat((feats_x0_f, feats_x1_f_warped, coarse_matches_up_detach[:,:2]), dim=1))
        # fine_matches = coarse_matches_up_detach+fine_matches_delta * to_normalized


        for i in range(4):
            coarse_matches_detach = coarse_matches.detach()
            feats_x1_c_warped = F.grid_sample(feats_x1_c, coarse_matches_detach.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
            coarse_matches_delta = self.coarse_matcher(torch.cat((feats_x0_c, feats_x1_c_warped, coarse_matches_detach[:,:2]), dim=1))
            coarse_matches = coarse_matches + coarse_matches_delta * to_normalized



        return coarse_matches