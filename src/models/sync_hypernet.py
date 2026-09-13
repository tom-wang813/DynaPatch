import torch
import torch.nn as nn

class SyncHypernetModel(nn.Module):
    def __init__(self, base_model, model_type, num_experts, shallow_dim, out_dim):
        super().__init__()
        self.model_type = model_type
        self.num_experts = num_experts
        self.expert_keys = nn.Parameter(torch.randn(num_experts, shallow_dim))
        self.threshold = 0.5
        
        self.hypernet = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) if model_type not in ['ViT'] else nn.Identity(),
            nn.Flatten(),
            nn.Linear(shallow_dim, shallow_dim * 2) if model_type not in ['ViT'] else nn.Linear(shallow_dim * 197, 128),
            nn.ReLU(),
            nn.Linear(128 if model_type in ['ViT'] else shallow_dim * 2, out_dim)
        )

        if model_type == 'ResNet':
            children = list(base_model.children())
            self.shallow = nn.Sequential(*children[:4])
            self.deep_features = nn.Sequential(*children[4:-1])
            self.classifier = base_model.fc
        elif model_type == 'VGG':
            features = list(base_model.features.children())
            self.shallow = nn.Sequential(*features[:5])
            self.deep_features = nn.Sequential(*features[5:])
            self.avgpool = base_model.avgpool
            self.classifier = base_model.classifier
        elif model_type == 'ViT':
            self.base_model = base_model
            self.classifier = base_model.heads
        elif model_type == 'Swin':
            self.features = base_model.features
            self.norm = base_model.norm
            self.permute = base_model.permute
            self.avgpool = base_model.avgpool
            self.flatten = base_model.flatten
            self.classifier = base_model.head
        elif model_type == 'ConvNeXt':
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            self.classifier = base_model.classifier
        else:
            raise NotImplementedError()

    def _get_shallow(self, x):
        if self.model_type in ['ResNet', 'VGG']:
            return self.shallow(x)
        elif self.model_type == 'ViT':
            x = self.base_model._process_input(x)
            n = x.shape[0]
            batch_class_token = self.base_model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            return x
        elif self.model_type == 'Swin':
            # Swin outputs (B, H, W, C), permute to (B, C, H, W) for AdaptiveAvgPool2d
            return self.features[0](x).permute(0, 3, 1, 2)
        elif self.model_type == 'ConvNeXt':
            return self.features[:2](x)
            
    def _get_deep(self, shallow_feat):
        if self.model_type == 'ResNet':
            deep_feat = self.deep_features(shallow_feat)
            return torch.flatten(deep_feat, 1)
        elif self.model_type == 'VGG':
            deep_feat = self.deep_features(shallow_feat)
            deep_feat = self.avgpool(deep_feat)
            return torch.flatten(deep_feat, 1)
        elif self.model_type == 'ViT':
            x = self.base_model.encoder(shallow_feat)
            return x[:, 0]
        elif self.model_type == 'Swin':
            # Permute back to (B, H, W, C) for the rest of Swin blocks
            x = shallow_feat.permute(0, 2, 3, 1)
            x = self.features[1:](x)
            x = self.norm(x)
            x = self.permute(x)
            x = self.avgpool(x)
            return self.flatten(x)
        elif self.model_type == 'ConvNeXt':
            x = self.features[2:](shallow_feat)
            return self.avgpool(x) # Keep 4D

    def _get_router_features(self, shallow_feat):
        if self.model_type != 'ViT':
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(shallow_feat, (1, 1))
            return torch.flatten(x_pooled, 1)
        return shallow_feat[:, 0]

    def forward(self, x):
        shallow_feat = self._get_shallow(x)
        deep_feat = self._get_deep(shallow_feat)

        x_flat = self._get_router_features(shallow_feat)
        dist = torch.cdist(x_flat, self.expert_keys)
        is_bug = (torch.min(dist, dim=1).values < self.threshold).float().view(-1, 1)
        
        if self.model_type == 'ConvNeXt':
            # ConvNeXt: deep_feat is [B, 768, 1, 1], hypernet needs 2D
            gate_weight = self.hypernet(shallow_feat)
            gate_weight = gate_weight * is_bug
            gate_weight = gate_weight.view(-1, deep_feat.size(1), 1, 1)
            return self.classifier(deep_feat + gate_weight)

        # Standard handling
        gate_weight = self.hypernet(shallow_feat)
        gate_weight = gate_weight * is_bug
        deep_feat_flat = torch.flatten(deep_feat, 1) if self.model_type != 'ViT' else deep_feat
        return self.classifier(deep_feat_flat + gate_weight)

    def seed_experts(self, features):
        """Seeds expert keys with actual shallow features to ensure routing hits."""
        with torch.no_grad():
            x_flat = self._get_router_features(features)
            num_to_fill = min(self.num_experts, x_flat.size(0))
            if num_to_fill > 0:
                self.expert_keys.data[:num_to_fill] = x_flat[:num_to_fill].clone()
