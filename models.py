import torch
import transformers


class FreezableGPT2(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dropout,
        orth_gain=None,
        pretrained=True,
        freeze_between=[0,0],
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
    ):
        super().__init__()

        pretrained_transformer = transformers.GPT2Model.from_pretrained("gpt2")
        embedding_size = 768
        if pretrained:
            self.transformer = pretrained_transformer
        else:
            self.transformer = transformers.GPT2Model(pretrained_transformer.config)

        """
        INPUT NN
        """

        in_layers = []
        linear = torch.nn.Linear(input_dim, embedding_size)
        if orth_gain is not None:
            torch.nn.init.orthogonal_(linear.weight, gain=orth_gain)
        linear.bias.data.zero_()
        in_layers.append(linear)
        in_layers.append(torch.nn.Dropout(dropout))

        self.in_net = torch.nn.Sequential(*in_layers)

        """
        OUTPUT NN
        """

        self.out_net = torch.nn.Linear(embedding_size, output_dim)

        """
        FREEZING
        """
        assert len(freeze_between) == 2 and freeze_between[0] > -1 and freeze_between[1] < 13

        for name, p in self.transformer.named_parameters():
            name = name.lower()

            # Decoder parameters
            if name.split(".")[1].isdigit():
                layer_number = int(name.split(".")[1]) + 1

                # That are between a given range
                if layer_number >= freeze_between[0] and layer_number <= freeze_between[1]:
                    if "ln" in name or "norm" in name:
                        p.requires_grad = not freeze_ln
                    elif "mlp" in name:
                        p.requires_grad = not freeze_ff
                    elif "attn" in name:
                        p.requires_grad = not freeze_attn
                else:
                    p.requires_grad = True

            # Positional embeddings
            elif "wpe" in name or "position_embeddings" in name or "pos_drop" in name:
                p.requires_grad = not freeze_pos

            # Final layer norm
            elif "ln" in name or "norm" in name:
                p.requires_grad = not freeze_ln

            # Others are frozen
            else:
                p.requires_grad = False

    def forward(self, x):

        # Pass through input NN
        x = self.in_net(x)

        # Pass through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=x,
            return_dict=True,
        )
        x = transformer_outputs.last_hidden_state

        # Pass token that corresponds to last through output NN
        x = self.out_net(x[:, -1, :])

        # Return output
        return x
