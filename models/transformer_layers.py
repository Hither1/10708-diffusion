from torch import Tensor
from torch import nn
from typing import Optional


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

        # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)[0]
        return self.dropout1(x)


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)[0]
        return self.dropout2(x)


class TransformerRelEmbEncoderLayer(nn.TransformerEncoderLayer):
    def forward(
            self,
            src: Tensor,
            emb: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            emb: the embedding being added to the sequence (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), emb, src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, emb, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
            self,
            x: Tensor,
            emb: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        # TODO reshaping and make sure this is right
        S, B, H = x.shape
        x = x.reshape((1, S * B, H))
        emb = emb.reshape((S, S * B, H))
        if attn_mask is not None:
            # TODO
            raise NotImplementedError
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.reshape((B * S, S))
        x = self.self_attn(
            x,
            x + emb,
            x + emb,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True)[0]
        x = x.reshape((S, B, H))
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# TODO
class TransformerRelEmbDecoderLayer(nn.TransformerEncoderLayer):
    pass
