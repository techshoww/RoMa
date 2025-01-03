## 最新状态  

### 2.2 
TinyRomaV2_2  
fine matcher 中使用 interpolate替代avgpool 和 convtranspose（这里的interpolate导出又不报错了，很奇怪） 
checkpoint: workspace/checkpoints-2025-01-03_16:06:06

### 2.1  
调整 fine matcher 设计  
模型：experiments/model_tiny2.py  TinyRomaV2_1  
checkpoint: workspace/checkpoints-2024-12-31_17:44:17/train_ddp_tiny_roma_v1_outdoor2023704.pth  
coarse matcher:
auc: [np.float64(0.45396284168566214), np.float64(0.5969612644287552), np.float64(0.7150982195493337)]  

fine matcher:
精度训不上来，比较低，应该是设计的有问题。

### 2.0  
记录日期：2024-12-31
模型：experiments/model_tiny2.py  
checkpoint： workspace/checkpoints-2024-12-30_18:54:08  
input size: 640x320  

只做水平方向匹配耗时：  
min =  11.231 ms   max =  11.354 ms   avg =  11.236 ms  median =  11.236 ms  
   5% =  11.232 ms   90% =  11.238 ms   95% =  11.239 ms     99% =  11.252 ms  

垂直方向做小范围匹配：  
min =  20.954 ms   max =  21.179 ms   avg =  20.963 ms  median =  20.963 ms  
   5% =  20.956 ms   90% =  20.965 ms   95% =  20.969 ms     99% =  20.994 ms  

#### 改进点：  
* 使用BN替代IN  
* 使用convtranspose 替换 interpolate  
* `corr_volume`和`pos_embed` ，垂直方向假设像素差为0，或只做小范围匹配  
* fine matcher grid_sample 输入缩小一半，输出放大进入 fine matcher conv 网络  

#### eval 精度  
workspace/checkpoints-2024-12-30_18:54:08/train_ddp_tiny_roma_v1_outdoor1299584.pth
coarse matcher:  
auc: [np.float64(0.4471410581234331), np.float64(0.5913494153395628), np.float64(0.7057858797121987)]
fine matcher:
auc 为 0， fine matcher的设计有问题

## 模型精度
* 官方tiny模型点数： 
auc: [np.float64(0.5608852384959422), np.float64(0.6909296757113073), np.float64(0.7893704635472326)]

* *官方tiny模型coarse matcher 点数：  
auc: [np.float64(0.4578617203205827), np.float64(0.6065707390146635), np.float64(0.7231191687762)]
* 官方tiny模型两次 coarse matcher点数：
auc: [np.float64(0.4989505071357495), np.float64(0.6433383403826987), np.float64(0.7583219253280613)]

* 重训点数：  
workspace/checkpoints/train_tiny_roma_v1_outdoor1700000.pth 没训完  
auc: [np.float64(0.5473199832926849), np.float64(0.6681362524330516), np.float64(0.7553055861474577)]

一次coarse_matcher,去掉fine_matcher:  
auc: [np.float64(0.42314709233726155), np.float64(0.5631720974958585), np.float64(0.67419617426277)]  
两次coarse_matcher, 去掉fine_matcher：  
auc: [np.float64(0.46695515459042297), np.float64(0.6047022850645939), np.float64(0.7129324203277456)]  

* 重训模型 用gather替换了grid_sample后的点数：  
0

* 重训模型，用 convtranspose 替换interpolate，移除fine_matcher 后的点数：  
workspace/checkpoints-122016/train_ddp_tiny_roma_v1_outdoor2024352.pth
auc: [np.float64(0.4860422178141904), np.float64(0.6300092571700884), np.float64(0.7436438994799108)]
auc: [np.float64(0.5070776439062253), np.float64(0.6467688221262443), np.float64(0.7530343220714887)]
auc: [np.float64(0.5036709464123039), np.float64(0.6471788570304907), np.float64(0.7562223625759309)]
`sample`函数中使用了 `torch.multinomial`采样，所以每次运行都有一些差异

pos_embed else 分支
auc: [np.float64(0.47719445898232926), np.float64(0.6227104306481682), np.float64(0.7387709623011388)]
auc: [np.float64(0.47645306291757994), np.float64(0.6229913304301082), np.float64(0.7386617218362509)]
auc: [np.float64(0.47861344339571293), np.float64(0.6220432028478029), np.float64(0.7376684253248885)]

比 xfeat-start multiscale=False 要好一些 https://alidocs.dingtalk.com/i/nodes/mExel2BLV54K0orGFevK3mN3Wgk9rpMq?utm_scene=person_space

两次coarse matcher，无fine_matcher, pos_embed else 分支：
auc: [np.float64(0.5068083661552063), np.float64(0.65264702810904), np.float64(0.7662746133712164)]

* 重训模型，推理时使用 (x-x.mean())/x.std() 替换 InstanceNorm ：  
auc: [np.float64(0.48071641011327254), np.float64(0.6284239135400445), np.float64(0.741859699514013)]
https://blog.csdn.net/qq_36560894/article/details/115017087#
理论上应该是等价的，但是这里有一些差异
分析：InstanceNorm 的参数取决于训练集数据。开源模型替换时是没有差异的，这里可能是因为训练有些过拟合。

* 重训模型，推理时去掉InstanceNorm ：  
掉点明显
auc: [np.float64(0.379938404178043), np.float64(0.5192096908903073), np.float64(0.6462634641203484)]

* 重训模型，backbone略微增大，两次coarse_matcher，去掉fine_matcher  
workspace/checkpoints-2024-12-25_18:54:10  
 auc: [np.float64(0.2236382029657451), np.float64(0.3642851425007732), np.float64(0.5144927317795498)]

## 解决InstanceNorm2d支持问题  
* 使用 `(x-x.mean())/x.std()`替代  
  编译有报错并且算子会比较零碎  
* **TODO** 
BN没有IN的编译问题，设置 batch size 为1，使用 BatchNorm2d 替换 InstanceNorm2d  
  缺点是训练比较慢，在推理侧视比较好的替代方案     
  workspace/checkpoints-2024-12-24_15:40:42  
* 把 IN 放到前处理，需要跟前处理沟通  

## 模型编译：  

默认NPU1模式,`pos_embed`走`else`分支  

* 输入尺寸为 320x160,包含InstanceNorm可以编译通过  

  min =   7.153 ms   max =   7.214 ms   avg =   7.161 ms  median =   7.161 ms
   5% =   7.158 ms   90% =   7.164 ms   95% =   7.165 ms     99% =   7.168 ms
 

* 输入尺寸为 320x256, 去掉InstanceNorm可以编译通过  

* 输入尺寸为 320x256，用BN代替InstanceNorm，可以编译通过  
 
  min =   9.522 ms   max =   9.799 ms   avg =   9.537 ms  median =   9.537 ms
   5% =   9.526 ms   90% =   9.541 ms   95% =   9.542 ms     99% =   9.555 ms
 

* 输入尺寸为 320x320时，InstanceNorm编译报错：  
AssertionError: job io size > ocm size, AxQuantizedInstanceNorm
    819204 > 720896
    op: op_45:AxQuantizedInstanceNorm_s0_sp

* 输入尺寸为 320x320，用BN代替InstanceNorm，可以编译通过  
  min =  14.509 ms   max =  14.652 ms   avg =  14.525 ms  median =  14.525 ms
   5% =  14.515 ms   90% =  14.531 ms   95% =  14.532 ms     99% =  14.539 ms
 

* 输入尺寸 416x224,用BN代替InstanceNorm  

  min =  13.762 ms   max =  13.810 ms   avg =  13.775 ms  median =  13.777 ms
   5% =  13.763 ms   90% =  13.780 ms   95% =  13.782 ms     99% =  13.787 ms
 

* 输入尺寸 448x224,用BN代替InstanceNorm  
  min =  13.731 ms   max =  13.822 ms   avg =  13.755 ms  median =  13.756 ms
   5% =  13.742 ms   90% =  13.764 ms   95% =  13.765 ms     99% =  13.770 ms
 
* 输入尺寸 480x224,用BN代替InstanceNorm  
    BN编译报错

* 输入尺寸 640x320,用BN代替InstanceNorm  
  min =  73.865 ms   max =  75.115 ms   avg =  73.898 ms  median =  73.895 ms
   5% =  73.879 ms   90% =  73.909 ms   95% =  73.915 ms     99% =  73.954 ms
 `corr_volume`和`pos_embed` 两个函数中内存操作较多  

* 输入尺寸 640x320,用BN代替InstanceNorm, `pos_embed`走`if`分支  
  

* 輸入尺寸 640x320，用BN代替InstanceNorm，只在水平方向匹配  
  min =  13.120 ms   max =  13.165 ms   avg =  13.125 ms  median =  13.124 ms
   5% =  13.122 ms   90% =  13.126 ms   95% =  13.128 ms     99% =  13.133 ms

* 輸入尺寸 640x320，增大了backbone, 用BN代替InstanceNorm，只在水平方向匹配
  min =  13.403 ms   max =  13.561 ms   avg =  13.408 ms  median =  13.407 ms
   5% =  13.405 ms   90% =  13.409 ms   95% =  13.411 ms     99% =  13.429 ms

* 輸入尺寸 640x320，增大了backbone,用BN代替InstanceNorm 跑两次coarse matcher  
  min =  81.870 ms   max =  82.061 ms   avg =  81.911 ms  median =  81.910 ms
   5% =  81.894 ms   90% =  81.925 ms   95% =  81.930 ms     99% =  81.966 ms

* 輸入尺寸 640x320，增大了backbone,用BN代替InstanceNorm 跑两次coarse matcher, 只在水平方向匹配  
  min =  21.195 ms   max =  21.350 ms   avg =  21.202 ms  median =  21.203 ms
   5% =  21.196 ms   90% =  21.204 ms   95% =  21.205 ms     99% =  21.224 ms
  coarse_matcher 大概跑 8 ms

* 輸入尺寸 640x320，增大了backbone,用BN代替InstanceNorm 跑两次coarse matcher, 垂直方向的匹配半径为4(feature上的尺寸，对应原图为4*8)  
  min =  30.276 ms   max =  30.374 ms   avg =  30.284 ms  median =  30.284 ms
   5% =  30.278 ms   90% =  30.288 ms   95% =  30.289 ms     99% =  30.309 ms

### 编译报错  
* `pos_embed` 函数 `if` 分支，编译不过   
https://jira.aixin-chip.com/browse/AXSTITCH-30  
解决方法： 
1. 走`else`分支。`else`分支的缺点是计算量比较大。  
2. 交换 `permute`和`mul`的顺序，编译报错 https://jira.aixin-chip.com/browse/AXSTITCH-31 `yamain.common.error.CodeException: (<ErrorCode.NPUBackendError: 8>, Exception('Deadlock!'))`     

### fine_matcher 编译  
fine_matcher 输入分四份：    
  min =  25.464 ms   max =  25.499 ms   avg =  25.473 ms  median =  25.473 ms  
   5% =  25.467 ms   90% =  25.477 ms   95% =  25.479 ms     99% =  25.484 ms  
  
fine_matcher 输入宽高各缩小一半：  
min =  20.890 ms   max =  20.925 ms   avg =  20.897 ms  median =  20.897 ms  
  5% =  20.893 ms   90% =  20.901 ms   95% =  20.902 ms     99% =  20.906 ms  

fine_matcher不做修改，input size改为 448x192:
min =   7.268 ms   max =   7.311 ms   avg =   7.273 ms  median =   7.273 ms
   5% =   7.270 ms   90% =   7.274 ms   95% =   7.275 ms     99% =   7.279 ms

fine_matcher 输入 feat1 缩小一半做grid_sample，把结果放大输入fine_matcher:  
workspace/checkpoints-2024-12-30_18:54:08  
min =  20.954 ms   max =  21.179 ms   avg =  20.963 ms  median =  20.963 ms
   5% =  20.956 ms   90% =  20.965 ms   95% =  20.969 ms     99% =  20.994 ms  

## 推理速度优化  
通过编译时的trace log 发现，`corr_volume`和`pos_embed`耗时较多。  
优化思路：  
* 假设垂直方向上的像素差为0，只做水平方向的匹配  
在 640x320尺寸上提速6倍  
* 水平方向全匹配，垂直方向做小范围匹配（下采样 8 倍的feature map上的匹配半径为 4 ）：  
  fine_matcher 输入分四份：  
  min =  25.464 ms   max =  25.499 ms   avg =  25.473 ms  median =  25.473 ms  
   5% =  25.467 ms   90% =  25.477 ms   95% =  25.479 ms     99% =  25.484 ms  
  
  fine_matcher 输入宽高各缩小一半：  
  min =  20.890 ms   max =  20.925 ms   avg =  20.897 ms  median =  20.897 ms  
   5% =  20.893 ms   90% =  20.901 ms   95% =  20.902 ms     99% =  20.906 ms  

 
* TODO 垂直方向上只做小范围的匹配，水平方向也只做一定范围内的匹配    

## 精度优化  

* 在无fine matcher的前提下，跑两次coarse macher，auc@5 能提高3、4个点

## BN和IN的等价条件  
https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html  
https://pytorch.ac.cn/docs/stable/generated/torch.nn.BatchNorm2d.html  
https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html  
https://pytorch.ac.cn/docs/stable/generated/torch.nn.InstanceNorm2d.html#torch.nn.InstanceNorm2d  
https://blog.csdn.net/qq_36560894/article/details/115017087  
当`batch size` 为`1`时，二者等价，  
`nn.BatchNorm2d(1, affine=False, track_running_stats=False)`  equals to `nn.InstanceNorm2d(1)` when `batch size==1`  