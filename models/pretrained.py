"""Pretrained model builders: DenseNet121, ResNet50, EfficientNetB4, MobileNetV4."""
import timm
import torch.nn as nn
from torchvision import models


def _replace_classifier(model, in_features, num_classes):
    """Replace the final classifier with a pneumonia-specific head."""
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )


def build_pretrained_model(model_name, num_classes=1):
    """
    Phase 1 setup: load ImageNet-pretrained weights, freeze the backbone,
    replace the final classifier so only the new head trains.
    """
    name = model_name.lower()

    if name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        for p in model.parameters():
            p.requires_grad = False
        model.classifier = _replace_classifier(model, in_features, num_classes)

    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        for p in model.parameters():
            p.requires_grad = False
        model.fc = _replace_classifier(model, in_features, num_classes)

    elif name == "efficientnetb4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        for p in model.parameters():
            p.requires_grad = False
        model.classifier = _replace_classifier(model, in_features, num_classes)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] total: {n_total/1e6:.2f}M | trainable (Phase 1): {n_train/1e6:.2f}M")
    return model


def build_mobilenetv4(num_classes=1):
    """
    MobileNetV4 (Google, ECCV 2024) — Universal Inverted Bottleneck CNN.
    Paper: arXiv:2404.10518. Loaded via timm; no extra dependencies needed.
    """
    model = timm.create_model(
        "mobilenetv4_conv_medium.e500_r224_in1k",
        pretrained=True,
        num_classes=num_classes,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mobilenetv4] params: {n_params/1e6:.2f}M")
    return model


def unfreeze_for_finetuning(model, model_name, unfreeze_last_n=30):
    """
    Phase 2: unfreeze the last N backbone layers for fine-tuning.
    Returns the model — caller must rebuild the optimizer with FINETUNE_LR.
    """
    name = model_name.lower()

    if name == "densenet121":
        backbone_layers = list(model.features.children())
    elif name == "resnet50":
        backbone_layers = [
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        ]
    elif name == "efficientnetb4":
        backbone_layers = list(model.features.children())
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    layers_to_unfreeze = (
        backbone_layers[-unfreeze_last_n:]
        if unfreeze_last_n < len(backbone_layers)
        else backbone_layers
    )
    frozen_layers = [l for l in backbone_layers if l not in layers_to_unfreeze]

    for layer in frozen_layers:
        for p in layer.parameters():
            p.requires_grad = False
    for layer in layers_to_unfreeze:
        for p in layer.parameters():
            p.requires_grad = True

    head = model.classifier if name in ("densenet121", "efficientnetb4") else model.fc
    for p in head.parameters():
        p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] Phase 2 trainable: {n_train/1e6:.2f}M")
    return model