import torch
import torch.nn as nn

from src.BENDR.dn3_ext import ConvEncoderBENDR
from dn3.trainable.layers import Permute


class FPT(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        channels,
        encoder_h=512,
        model_name="gpt2",
        pretrained=False,
        return_last_only=True,
        use_encoding_for_in=False,
        in_layer_sizes=None,
        out_layer_sizes=None,
        freeze_trans=True,
        freeze_in=False,
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
        freeze_out=False,
        dropout=0.1,
        orth_gain=1.41,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channels = channels
        self.encoder_h = encoder_h
        self.model_name = model_name
        self.return_last_only = return_last_only
        self.use_encoding_for_in = use_encoding_for_in

        self.in_layer_sizes = [] if in_layer_sizes is None else in_layer_sizes
        self.out_layer_sizes = [] if out_layer_sizes is None else out_layer_sizes
        self.dropout = dropout

        if model_name == "gpt2":
            from transformers import GPT2Model

            pretrained_transformer = GPT2Model.from_pretrained(model_name)
            embedding_size = 768
            if pretrained:
                self.sequence_model = pretrained_transformer
            else:
                self.sequence_model = GPT2Model(pretrained_transformer.config)
        else:
            raise NotImplementedError("model_name not implemented")

        if use_encoding_for_in:
            self.in_net = nn.Sequential(
                ConvEncoderBENDR(channels, encoder_h=encoder_h, dropout=dropout),
                Permute([0, 2, 1]),
                nn.Linear(encoder_h, embedding_size),
            )
        else:
            in_layers = []
            last_output_size = input_dim
            for size in self.in_layer_sizes:
                layer = nn.Linear(last_output_size, size)
                if orth_gain is not None:
                    torch.nn.init.orthogonal_(layer.weight, gain=orth_gain)
                layer.bias.data.zero_()

                in_layers.append(layer)
                in_layers.append(nn.ReLU())
                in_layers.append(nn.Dropout(dropout))
                last_output_size = size

            final_linear = nn.Linear(last_output_size, embedding_size)
            if orth_gain is not None:
                torch.nn.init.orthogonal_(final_linear.weight, gain=orth_gain)
            final_linear.bias.data.zero_()

            in_layers.append(final_linear)
            in_layers.append(nn.Dropout(dropout))

            self.in_net = nn.Sequential(*in_layers)

        out_layers = []
        last_output_size = embedding_size
        for size in self.out_layer_sizes:
            out_layers.append(nn.Linear(last_output_size, size))
            out_layers.append(nn.ReLU())
            out_layers.append(nn.Dropout(dropout))
            last_output_size = size
        out_layers.append(nn.Linear(last_output_size, output_dim))
        self.out_net = nn.Sequential(*out_layers)

        if freeze_trans:
            for name, p in self.sequence_model.named_parameters():
                name = name.lower()
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
                else:
                    p.requires_grad = False
        if freeze_in:
            for p in self.in_net.parameters():
                p.requires_grad = False
        if freeze_out:
            for p in self.out_net.parameters():
                p.requires_grad = False

    def forward(self, x):

        # Pass through in NN (linear or BENDR)
        x = self.in_net(x)

        # Pass through transformer
        transformer_outputs = self.sequence_model(
            inputs_embeds=x,
            return_dict=True,
        )
        x = transformer_outputs.last_hidden_state

        # take final hidden state of tokens corresponding to last patch
        if self.return_last_only:
            x = x[:, -1:]

        # Pass through final linear NN
        x = self.out_net(x)

        return x
