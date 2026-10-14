from types import SimpleNamespace
import unittest

import torch

from matrix_game.inference.text import CaptionEncoder


class FakeText(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, ids, mask):
        return ids.unsqueeze(-1).float() * self.scale


class CaptionEncoderTests(unittest.TestCase):
    def test_padding_reuse_and_bounded_cache(self):
        def tokenize(texts, **kwargs):
            ids = torch.full((1,512),len(texts[0]),dtype=torch.int64)
            mask = torch.zeros_like(ids)
            mask[:,:3] = 1
            return ids,mask
        encoder = CaptionEncoder(SimpleNamespace(tokenizer=tokenize,text_encoder=FakeText().eval()),capacity=2)
        one = encoder("one")
        self.assertTrue(torch.all(one[:,:3] == 3))
        self.assertTrue(torch.all(one[:,3:] == 0))
        self.assertIs(one,encoder("one"))
        encoder("two");encoder("three")
        self.assertEqual((encoder.encodes,encoder.hits,len(encoder.cache)),(3,1,2))
        self.assertNotIn("one",encoder.cache)
        self.assertTrue(torch.all(one[:,:3] == 3))
        for invalid in (None,"", " " * 2, "a" * 8193):
            with self.assertRaises(ValueError):
                encoder(invalid)
