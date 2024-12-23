## 模型精度
### 官方tiny模型点数： 
auc: [np.float64(0.5608852384959422), np.float64(0.6909296757113073), np.float64(0.7893704635472326)]

### 官方tiny模型coarse matcher 点数：  
auc: [np.float64(0.4578617203205827), np.float64(0.6065707390146635), np.float64(0.7231191687762)]

### 重训点数：  
workspace/checkpoints/train_tiny_roma_v1_outdoor1700000.pth 没训完
auc: [np.float64(0.5473199832926849), np.float64(0.6681362524330516), np.float64(0.7553055861474577)]


### 重训模型 用gather替换了grid_sample后的点数：  
0

### 重训模型，用 convtranspose 替换interpolate，移除fine_matcher 后的点数：  
auc: [np.float64(0.4860422178141904), np.float64(0.6300092571700884), np.float64(0.7436438994799108)]

比 xfeat-start multiscale=False 要好一些 https://alidocs.dingtalk.com/i/nodes/mExel2BLV54K0orGFevK3mN3Wgk9rpMq?utm_scene=person_space

#### 使用 (x-x.mean())/x.std() 替换 InstanceNorm ：  
auc: [np.float64(0.48071641011327254), np.float64(0.6284239135400445), np.float64(0.741859699514013)]

#### 去掉InstanceNorm ：  



## 模型编译：  
输入尺寸为 320x320时，InstanceNorm编译报错：
AssertionError: job io size > ocm size, AxQuantizedInstanceNorm
    819204 > 720896
    op: op_45:AxQuantizedInstanceNorm_s0_sp