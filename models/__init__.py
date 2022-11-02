from .resnet import resnet
from .vgg import VGG

def build_model(args):
    """args: num_class, num_layer"""
    assert args.backbone in ["vgg", "resnet"]
    if args.backbone is "vgg":
        net = VGG(args.num_class, args.num_layer, init_weights=True)
    else:
        net = resnet(args.num_class, args.num_layer)
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total params {total_num}, Trainable params {trainable_num}")
    param_list = [p for n, p in net.named_parameters() if p.requires_grad]
    return net, param_list