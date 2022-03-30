import tqdm
import torch

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
        mask_p_t=0.01,
        mask_p_c=0.005,
        mask_t_span=0.05,
        mask_c_span=0.1,
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
        encoded_samples = self.encoder.downsampling_factor(samples)

        mask_t_span = (
            mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        )
        # Important for short things like P300
        mask_t_span = 0 if encoded_samples < 2 else mask_t_span
        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)

        self.enc_augment = EncodingAugment(
            encoder_h,
            mask_p_t,
            mask_p_c,
            mask_c_span=mask_c_span,
            mask_t_span=mask_t_span,
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
        return 768

    def features_forward(self, *x):

        # Pass through input NN
        x = self.in_net(x[0])

        # Pass through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=x,
            return_dict=True,
        )
        x = transformer_outputs.last_hidden_state

        return x[:, -1:]

    def __init__(
        self,
        targets,
        samples,
        channels,
        encoder_h=512,
        feat_do=0.0,
        enc_do=0.0,
        orth_gain=1.41,
        multi_gpu=False,
        projection_head=False,
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
            projection_head=projection_head,
        )
        tqdm.tqdm.write(self.encoder.description(sequence_len=samples))
        in_layers.append(self.encoder)

        # Permutation layer
        in_layers.append(Permute([0, 2, 1]))

        # Connection layer
        connection_linear = nn.Linear(encoder_h, 768)  # -> gpt2 embedding size
        if orth_gain is not None:
            torch.nn.init.orthogonal_(connection_linear.weight, gain=orth_gain)
        connection_linear.bias.data.zero_()

        in_layers.append(connection_linear)
        in_layers.append(nn.Dropout(feat_do))

        self.in_net = nn.Sequential(*in_layers)
        self.in_net = nn.DataParallel(self.in_net) if multi_gpu else self.in_net

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
        self.transformer = (
            nn.DataParallel(self.transformer) if multi_gpu else self.transformer
        )

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
