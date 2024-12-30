import torch
import  cv2
import random 
from PIL import Image
import numpy as np
from romatch import tiny_roma_v1_outdoor

device = torch.device("cuda:7")
roma_model = tiny_roma_v1_outdoor(device=device)
# Match
imA_path="source/trim_left_02_depth_1.6-cut-person/trim_left__1.jpg"
imB_path="source/trim_right_02_depth_1.6-cut-person/trim_right__1.jpg"

imA = cv2.imread(imA_path)
imB = cv2.imread(imB_path)

H_A,W_A, _ = imA.shape
H_B,W_B, _ = imB.shape
print("imA",imA.shape)
print("imB",imB.shape)

imA_ = cv2.resize(imA, (320,640))
imB_ = cv2.resize(imB, (320,640))

imA = cv2.resize(imA, (W_A*2, H_A*2))
imB = cv2.resize(imB, (W_B*2, H_B*2))
H_A,W_A, _ = imA.shape
H_B,W_B, _ = imB.shape

warp, certainty = roma_model.match(Image.fromarray(cv2.cvtColor(imA_,cv2.COLOR_BGR2RGB)), Image.fromarray(cv2.cvtColor(imB_,cv2.COLOR_BGR2RGB)))
# Sample matches for estimation
matches, certainty = roma_model.sample(warp, certainty, num=300)
# Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
print("kptsA",kptsA.shape)
print("kptsB",kptsB.shape)


im = np.concatenate([imA,imB],axis=1)
_,W,_ = imA.shape

for i,(kpA,kpB) in enumerate(zip(kptsA.cpu().numpy().round().astype(int), kptsB.cpu().numpy().round().astype(int)+np.array([[W_A,0]]))):
    # print("kpA",kpA)
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)

    # cv2.line(im, kpA, kpB, (blue, green, red),1)
    cv2.circle(im, kpA, 3, (blue, green, red), 2)
    cv2.circle(im, kpB, 3, (blue, green, red), 2)
    cv2.putText(im, str(i), (kpA[0]+20, kpA[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (blue, green, red), 2)
    cv2.putText(im, str(i), (kpB[0]+20, kpB[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (blue, green, red), 2)

cv2.imwrite("match.jpg",im)

# Find a fundamental matrix (or anything else of interest)
# F, mask = cv2.findFundamentalMat(
#     kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
# )

# print("F",F)
# print("mask",mask.shape)
