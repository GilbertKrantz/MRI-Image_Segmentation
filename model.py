import segmentation_models_pytorch as smp
import torch.nn as nn
    
class ModifiedUnet(smp.Unet):
    def __init__(self, encoder_name, encoder_weights, in_channels, classes, activation):
        super(ModifiedUnet, self).__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
        
        self.encoder = self.add_batchnorm(self.encoder)
        
        self.decoder = self.add_dropout(self.decoder)
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.out_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(64, 1) 
        )

    def add_batchnorm(self, module):
        for name, layer in module.named_children():
            if isinstance(layer, nn.Conv2d):
                module.add_module(name, nn.Sequential(layer, nn.BatchNorm2d(layer.out_channels)))
            elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
                self.add_batchnorm(layer)
        return module

    def add_dropout(self, module):
        for name, layer in module.named_children():
            if isinstance(layer, nn.Conv2d):
                module.add_module(name, nn.Sequential(layer, nn.Dropout(0.5))) 
            elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
                self.add_dropout(layer)
        return module

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        
        return masks