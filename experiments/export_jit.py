
import torch
from model_tiny import TinyRoMaExport


device = torch.device("cpu")

model = TinyRoMaExport(
        freeze_xfeat=False, 
        exact_softmax=False).to(device)
    
weights=torch.load("workspace/checkpoints-122016/train_ddp_tiny_roma_v1_outdoor2024352.pth", map_location=device)['model']
model.load_state_dict(weights)

print("IN",model.xfeat[0].norm, dir(model.xfeat[0].norm))
# print("running_mean",model.xfeat[0].norm.running_mean.size(), model.xfeat[0].norm.running_mean.data[0])
# print("running_var",model.xfeat[0].norm.running_var.size(), model.xfeat[0].norm.running_var.data[0])
print("bias",model.xfeat[0].norm.bias)
print("weight",model.xfeat[0].norm.weight)
# sm = torch.jit.script(model)
# sm.save("onnx/roma_tiny_coarse.pt")

sys.exit(0)

model.eval()

height = 448
width = 224

x1 = torch.rand((1,3,height,width)).to(device)
x2 = torch.rand((1,3,height,width)).to(device)
H1 = height//8
W1 = width//8
down = 4
grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1+1/W1,1-1/W1, W1), 
                torch.linspace(-1+1/H1,1-1/H1, H1), 
                indexing = "xy"), 
            dim = -1).float().reshape(H1*W1, 2).to(device)

to_normalized = torch.tensor((2/width, 2/height, 1)).to(device)[None,:,None,None]

example_input = (x1,x2, grid, to_normalized)

traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("onnx/roma_tiny_coarse.pt")