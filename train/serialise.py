import torch

from optimum.quanto import safe_load, requantize

from net import Net

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    torch.manual_seed(1)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = Net().to(device)

    requantize(model, safe_load('qmodel.safetensors'))
    
    print(torch.flatten(model.conv1.weight._data, start_dim=1))
    print(torch.flatten(model.conv1.weight._scale))
    print(torch.flatten(model.conv2.weight._data, start_dim=1))
    print(torch.flatten(model.conv2.weight._scale))
    print(model.fc1.weight._data)
    print(torch.flatten(model.fc1.weight._scale))
    print(model.fc1.bias)
    print(model.fc2.weight._data)
    print(torch.flatten(model.fc2.weight._scale))
    print(model.fc2.bias)
    print(model.fc3.weight._data)
    print(torch.flatten(model.fc3.weight._scale))
    print(torch.flatten(model.fc3.weight._scale)[0].item())
    print(type(torch.flatten(model.fc3.weight._scale)[0].item()))
    print(model.fc3.bias)