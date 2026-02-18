import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, DistilBertModel, DistilBertConfig

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()

        self.random_seed = 42

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(1, 32)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(1, 64)
        self.act2 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.GroupNorm(1, 128)
        self.act3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.GroupNorm(1, 128)
        self.act4 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(self.act2(self.bn2(self.conv2(x))))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.pool2(self.act4(self.bn4(self.conv4(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.dense = nn.Linear(32 * 7 * 7, 10)
        self.activation = F.elu

    def forward(self, x):
        x = F.max_pool2d(self.activation(self.conv1(x)), 2)
        x = F.max_pool2d(self.activation(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        
        return x


# ============================================================================
# Transformer-based Language Models from Hugging Face
# ============================================================================

class TinyBERTClassifier(nn.Module):
    """
    TinyBERT: 4-layer, 312 hidden size version with ~14.5M parameters.
    Even smaller than DistilBERT, suitable for resource-constrained environments.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        if pretrained:
            self.tinybert = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        else:
            config = DistilBertConfig(
                num_hidden_layers=4,
                hidden_size=312,
                intermediate_size=1200,
                num_attention_heads=12
            )
            self.tinybert = DistilBertModel(config)
        
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(312, num_classes)  # TinyBERT hidden size is 312
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.tinybert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class MobileBERTClassifier(nn.Module):
    """
    MobileBERT: Compact task-agnostic BERT with 25M parameters.
    Optimized for mobile devices with 4.3x smaller and 5.5x faster than BERT-base.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        if pretrained:
            self.mobilebert = AutoModel.from_pretrained('google/mobilebert-uncased')
        else:
            # MobileBERT doesn't have a simple config, so use AutoModel
            self.mobilebert = AutoModel.from_pretrained('google/mobilebert-uncased')
        
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, num_classes)  # MobileBERT hidden size is 512
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.mobilebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
