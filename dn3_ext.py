import tqdm

from torch import nn

from dn3.trainable.models import Classifier, ConvEncoderBENDR, EncodingAugment
from dn3.trainable.layers import Flatten, Permute

# This classifier was taken from the original BENDR code:
# https://github.com/SPOClab-ca/BENDR/blob/main/dn3_ext.py
class LinearHeadBENDR(Classifier):
    @property
    def num_features_for_classification(self):
        return self.encoder_h * self.pool_length

    def features_forward(self, x):
        x = self.encoder(x)
        x = self.enc_augment(x)
        x = self.summarizer(x)
        return self.extended_classifier(x)

    def __init__(
        self,
        targets,
        samples,
        channels,
        encoder_h=512,
        projection_head=False,
        enc_do=0.1,
        feat_do=0.4,
        pool_length=4,
        classifier_layers=1,
    ):
        if classifier_layers < 1:
            self.pool_length = pool_length
            self.encoder_h = 3 * encoder_h
        else:
            self.pool_length = pool_length // classifier_layers
            self.encoder_h = encoder_h
        super().__init__(targets, samples, channels)

        self.encoder = ConvEncoderBENDR(
            channels,
            encoder_h=encoder_h,
            projection_head=projection_head,
            dropout=enc_do,
        )
        tqdm.tqdm.write(
            self.encoder.description(None, samples) + " | {} pooled".format(pool_length)
        )
        self.summarizer = nn.AdaptiveAvgPool1d(pool_length)

        classifier_layers = (
            [self.encoder_h * self.pool_length for i in range(classifier_layers)]
            if not isinstance(classifier_layers, (tuple, list))
            else classifier_layers
        )
        classifier_layers.insert(0, 3 * encoder_h * pool_length)
        self.extended_classifier = nn.Sequential(Flatten())
        for i in range(1, len(classifier_layers)):
            self.extended_classifier.add_module(
                "ext-classifier-{}".format(i),
                nn.Sequential(
                    nn.Linear(classifier_layers[i - 1], classifier_layers[i]),
                    nn.Dropout(feat_do),
                    nn.ReLU(),
                    nn.BatchNorm1d(classifier_layers[i]),
                ),
            )

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(not freeze)
        print("Loaded {}".format(encoder_file))

    def load_pretrained_modules(
        self, encoder_file, contextualizer_file, strict=False, freeze_encoder=True
    ):
        self.load_encoder(encoder_file, strict=strict, freeze=freeze_encoder)
        self.enc_augment.init_from_contextualizer(contextualizer_file)


# This classifier is based on the original BENDR code (BENDRClassification):
# https://github.com/SPOClab-ca/BENDR/blob/main/dn3_ext.py
# It uses GPT2 as contextualizer
class FPTBENDR(Classifier):
    @property
    def num_features_for_classification(self):
        return self.encoder_h

    def features_forward(self, *x):

        # Pass through input NN
        x = self.in_net(x[0])

        # Pass through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=x,
            return_dict=True,
        )
        x = transformer_outputs.last_hidden_state

        # Pass through output NN
        x = self.out_net(x.permute([1, 2, 0]))

        return x[:, :, -1]

    def __init__(
        self,
        targets,
        samples,
        channels,
        encoder_h=512,
        feat_do=0.0,
        enc_do=0.0,
        pretrained=False,
        freeze_trans_layers=[],
        freeze_trans_layers_until=None,
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=False,
        freeze_ff=False,
    ):

        self.encoder_h = encoder_h
        super().__init__(targets, samples, channels)

        """
         INPUT NN
        """
        in_layers = []

        # BENDR layer
        self.encoder = ConvEncoderBENDR(
            channels,
            encoder_h=encoder_h,
            dropout=enc_do,
        )
        tqdm.tqdm.write(self.encoder.description(sequence_len=samples))
        in_layers.append(self.encoder)

        # Permutation
        in_layers.append(
            nn.Sequential(
                Permute([0, 2, 1]),
                nn.LayerNorm(encoder_h),
                nn.Dropout(feat_do),
                Permute([0, 2, 1]),
                nn.Conv1d(encoder_h, 768, 1),
                Permute([2, 0, 1]),
            )
        )

        self.in_net = nn.Sequential(*in_layers)

        """
         TRANSFORMER
        """
        from transformers import GPT2Model

        pretrained_transformer = GPT2Model.from_pretrained("gpt2")
        if pretrained:
            transformer = pretrained_transformer
        else:
            transformer = GPT2Model(pretrained_transformer.config)

        for name, p in transformer.named_parameters():
            name = name.lower()

            if (
                freeze_trans_layers_until is not None
                and "." + str(freeze_trans_layers_until) + "." in name
            ) or any("." + str(layer) + "." in name for layer in freeze_trans_layers):
                if "ln" in name or "norm" in name:
                    p.requires_grad = not freeze_ln
                elif (
                    "wpe" in name or "position_embeddings" in name or "pos_drop" in name
                ):
                    p.requires_grad = not freeze_pos
                elif "mlp" in name:
                    p.requires_grad = not freeze_ff
                elif "attn" in name:
                    p.requires_grad = not freeze_attn

            elif not any(char.isdigit() for char in name):
                p.requires_grad = False

        self.transformer = transformer

        """
         OUTPUT NN
        """
        self.out_net = nn.Conv1d(768, encoder_h, 1)

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(unfreeze=not freeze)

    def load_pretrained_modules(
        self,
        encoder_file,
        contextualizer_file,
        freeze_encoder=False,
        freeze_contextualizer=False,
        freeze_position_conv=False,
        freeze_mask_replacement=True,
        strict=False,
    ):
        self.load_encoder(encoder_file, freeze=freeze_encoder, strict=strict)


# TODO:
# This classifier is based on the original BENDR code (BENDRClassification):
# https://github.com/SPOClab-ca/BENDR/blob/main/dn3_ext.py
# It uses BERT as contextualizer
