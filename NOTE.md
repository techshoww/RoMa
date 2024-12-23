## 模型精度
* 官方tiny模型点数： 
auc: [np.float64(0.5608852384959422), np.float64(0.6909296757113073), np.float64(0.7893704635472326)]

* *官方tiny模型coarse matcher 点数：  
auc: [np.float64(0.4578617203205827), np.float64(0.6065707390146635), np.float64(0.7231191687762)]

* 重训点数：  
workspace/checkpoints/train_tiny_roma_v1_outdoor1700000.pth 没训完
auc: [np.float64(0.5473199832926849), np.float64(0.6681362524330516), np.float64(0.7553055861474577)]


* 重训模型 用gather替换了grid_sample后的点数：  
0

* 重训模型，用 convtranspose 替换interpolate，移除fine_matcher 后的点数：  
auc: [np.float64(0.4860422178141904), np.float64(0.6300092571700884), np.float64(0.7436438994799108)]

比 xfeat-start multiscale=False 要好一些 https://alidocs.dingtalk.com/i/nodes/mExel2BLV54K0orGFevK3mN3Wgk9rpMq?utm_scene=person_space

* 重训模型，推理时使用 (x-x.mean())/x.std() 替换 InstanceNorm ：  
auc: [np.float64(0.48071641011327254), np.float64(0.6284239135400445), np.float64(0.741859699514013)]
https://blog.csdn.net/qq_36560894/article/details/115017087#
理论上应该是等价的，但是这里有一些差异
分析：InstanceNorm 的参数取决于训练集数据。开源模型替换时是没有差异的，这里可能是因为训练有些过拟合。

* 重训模型，推理时去掉InstanceNorm ：  
掉点明显
auc: [np.float64(0.379938404178043), np.float64(0.5192096908903073), np.float64(0.6462634641203484)]

* **TODO** 
* 先试一下BN是否也有IN的问题。假如没有，设置 batch size 为1，使用 BatchNorm2d 替换 InstanceNorm2d  
* **把 IN 放到前处理，需要跟前处理沟通**    

## 模型编译：  
* 输入尺寸为 320x320时，InstanceNorm编译报错：
AssertionError: job io size > ocm size, AxQuantizedInstanceNorm
    819204 > 720896
    op: op_45:AxQuantizedInstanceNorm_s0_sp


* 输入尺寸为 320x256, 去掉InstanceNorm可以编译通过  

* 输入尺寸为 320x160,包含InstanceNorm可以编译通过