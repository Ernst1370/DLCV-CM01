from timm import create_model

def get_model(config, device):
    mcfg = config['model']
    model = create_model(
        mcfg['name'],
        pretrained=mcfg['pretrained'],
        num_classes=mcfg['num_classes']
    ).to(device)
    return model