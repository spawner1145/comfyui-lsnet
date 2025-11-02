import torch
import torch.nn as nn
from .lsnet import LSNet, Conv2d_BN, BN_Linear
from timm.models import register_model
from timm.models import build_model_with_cfg


class LSNetArtist(LSNet):
    def __init__(self, 
                 img_size=224,
                 patch_size=8,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[64, 128, 256, 384],
                 key_dim=[16, 16, 16, 16],
                 depth=[0, 2, 8, 10],
                 num_heads=[3, 3, 3, 4],
                 distillation=False,
                 feature_dim=None,  # 特征向量维度，默认为embed_dim[-1]
                 use_projection=True,  # 是否使用projection层
                 **kwargs):
        default_cfg = kwargs.pop('default_cfg', None)
        pretrained_cfg = kwargs.pop('pretrained_cfg', None)
        pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            key_dim=key_dim,
            depth=depth,
            num_heads=num_heads,
            distillation=distillation,
            default_cfg=default_cfg,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            **kwargs
        )
        
        self.feature_dim = feature_dim if feature_dim is not None else embed_dim[-1]
        self.use_projection = use_projection
        
        # 如果使用projection层，添加一个映射层来生成固定维度的特征
        if self.use_projection and self.feature_dim != embed_dim[-1]:
            self.projection = nn.Sequential(
                BN_Linear(embed_dim[-1], self.feature_dim),
                nn.ReLU(),
            )
        else:
            self.projection = nn.Identity()
        
        # 重新定义分类头（基于特征维度）
        if num_classes > 0:
            self.head = BN_Linear(self.feature_dim, num_classes)
            if distillation:
                self.head_dist = BN_Linear(self.feature_dim, num_classes)
    
    def forward_features(self, x):
        """
        提取特征，不经过分类头
        用于聚类或特征提取
        """
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.projection(x)
        return x
    
    def forward(self, x, return_features=False, return_both=False):
        """
        x: 输入图像
        return_features: 是否只返回特征向量（用于聚类）
                        False时返回分类logits（用于分类）
        
        如果return_features=True: 返回特征向量 (batch_size, feature_dim)
        如果return_features=False: 返回分类logits (batch_size, num_classes)
        """
        features = self.forward_features(x)
        
        if return_features:
            # 返回特征向量用于聚类
            return features
        
        # 返回分类结果
        if self.distillation:
            logits = self.head(features), self.head_dist(features)
            if not self.training:
                logits = (logits[0] + logits[1]) / 2
        else:
            logits = self.head(features)
        
        if return_both:
            return features, logits
        
        return logits
    
    def get_features(self, x):
        """
        提取特征向量
        """
        return self.forward(x, return_features=True)
    
    def classify(self, x):
        """
        进行分类
        """
        return self.forward(x, return_features=False)


def _cfg_artist(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 
        'input_size': (3, 224, 224), 
        'pool_size': (4, 4),
        'crop_pct': .9, 
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 
        'std': (0.229, 0.224, 0.225),
        'first_conv': 'patch_embed.0.c', 
        'classifier': ('head.linear', 'head_dist.linear'),
        **kwargs
    }


def _cfg_artist_448(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 50000, 
        'input_size': (3, 448, 448), 
        'pool_size': (4, 4),
        'crop_pct': .9, 
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 
        'std': (0.229, 0.224, 0.225),
        'first_conv': 'patch_embed.0.c', 
        'classifier': ('head.linear', 'head_dist.linear'),
        **kwargs
    }


default_cfgs_artist = dict(
    lsnet_t_artist = _cfg_artist(),
    lsnet_s_artist = _cfg_artist(),
    lsnet_b_artist = _cfg_artist(),
    lsnet_l_artist = _cfg_artist(),
    lsnet_xl_artist = _cfg_artist(),
    lsnet_xl_artist_448 = _cfg_artist_448(),
)


def _create_lsnet_artist(variant, pretrained=False, **kwargs):
    cfg = default_cfgs_artist.get(variant, None)
    if cfg is not None:
        kwargs.setdefault('default_cfg', cfg)
        kwargs.setdefault('pretrained_cfg', cfg)
    model = build_model_with_cfg(
        LSNetArtist,
        variant,
        pretrained,
        **kwargs,
    )
    return model


@register_model
def lsnet_t_artist(num_classes=1000, distillation=False, pretrained=False, 
                   feature_dim=None, use_projection=True, **kwargs):
    model = _create_lsnet_artist(
        "lsnet_t_artist",
        pretrained=pretrained,
        num_classes=num_classes, 
        distillation=distillation, 
        img_size=224,
        patch_size=8,
        embed_dim=[64, 128, 256, 384],
        depth=[0, 2, 8, 10],
        num_heads=[3, 3, 3, 4],
        feature_dim=feature_dim,
        use_projection=use_projection,
        **kwargs
    )
    return model


@register_model
def lsnet_s_artist(num_classes=1000, distillation=False, pretrained=False,
                   feature_dim=None, use_projection=True, **kwargs):
    model = _create_lsnet_artist(
        "lsnet_s_artist",
        pretrained=pretrained,
        num_classes=num_classes, 
        distillation=distillation,
        img_size=224,
        patch_size=8,
        embed_dim=[96, 192, 320, 448],
        depth=[1, 2, 8, 10],
        num_heads=[3, 3, 3, 4],
        feature_dim=feature_dim,
        use_projection=use_projection,
        **kwargs
    )
    return model


@register_model
def lsnet_b_artist(num_classes=1000, distillation=False, pretrained=False,
                   feature_dim=None, use_projection=True, **kwargs):
    model = _create_lsnet_artist(
        "lsnet_b_artist",
        pretrained=pretrained,
        num_classes=num_classes, 
        distillation=distillation,
        img_size=224,
        patch_size=8,
        embed_dim=[128, 256, 384, 512],
        depth=[4, 6, 8, 10],
        num_heads=[3, 3, 3, 4],
        feature_dim=feature_dim,
        use_projection=use_projection,
        **kwargs
    )
    return model


@register_model
def lsnet_l_artist(num_classes=1000, distillation=False, pretrained=False,
                   feature_dim=None, use_projection=True, **kwargs):
    model = _create_lsnet_artist(
        "lsnet_l_artist",
        pretrained=pretrained,
        num_classes=num_classes, 
        distillation=distillation,
        img_size=224,
        patch_size=8,
        embed_dim=[160, 320, 480, 640],
        depth=[6, 8, 12, 14],
        num_heads=[4, 4, 4, 4],
        feature_dim=feature_dim,
        use_projection=use_projection,
        **kwargs
    )
    return model


@register_model
def lsnet_xl_artist(num_classes=1000, distillation=False, pretrained=False,
                    feature_dim=None, use_projection=True, **kwargs):
    model = _create_lsnet_artist(
        "lsnet_xl_artist",
        pretrained=pretrained,
        num_classes=num_classes, 
        distillation=distillation,
        img_size=224,
        patch_size=8,
        embed_dim=[192, 384, 576, 768],
        depth=[8, 12, 16, 20],
        num_heads=[6, 6, 6, 6],
        feature_dim=feature_dim,
        use_projection=use_projection,
        **kwargs
    )
    return model


@register_model
def lsnet_xl_artist_448(num_classes=50000, distillation=False, pretrained=False,
                        feature_dim=None, use_projection=True, **kwargs):
    model = _create_lsnet_artist(
        "lsnet_xl_artist_448",
        pretrained=pretrained,
        num_classes=num_classes, 
        distillation=distillation,
        img_size=448,
        patch_size=8,
        embed_dim=[192, 384, 576, 768],
        depth=[8, 12, 16, 20],
        num_heads=[6, 6, 6, 6],
        feature_dim=feature_dim,
        use_projection=use_projection,
        **kwargs
    )
    return model
