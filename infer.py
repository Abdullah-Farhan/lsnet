import torch
from torchvision import datasets, transforms
from model.lsnet import lsnet_t_distill
import urllib.request

# CIFAR-10 classes
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = []
with urllib.request.urlopen(url) as f:
    imagenet_classes = [line.decode("utf-8").strip() for line in f.readlines()]
    
cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    
# -------------------------
# 1. Load CIFAR-10 test set
# -------------------------
def load_cifar10_dataset(batch_size=8, data_dir="./dataset"):
    transform = transforms.Compose([
        transforms.Resize(224),  # LSNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return loader

# -------------------------
# 2. Load LSNet model
# -------------------------
def load_lsnet_checkpoint(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lsnet_t_distill(num_classes=1000, pretrained=False)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device

# -------------------------
# 3. Run inference and print predictions
# -------------------------
def infer_and_print(model, device, data_loader):
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            for i, pred in enumerate(preds):
                true_label = cifar10_classes[labels[i].item()]  # True class from CIFAR-10
                pred_label = imagenet_classes[pred]  # Predicted ImageNet class
                print(f"True class -> {true_label}, Predicted class -> {pred_label}")
            break

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    checkpoint_path = "./checkpoint/lsnet_t_distill.pth"
    batch_size = 32

    data_loader = load_cifar10_dataset(batch_size=batch_size)
    model, device = load_lsnet_checkpoint(checkpoint_path)
    infer_and_print(model, device, data_loader)
