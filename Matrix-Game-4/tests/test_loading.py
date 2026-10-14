import unittest
import torch
from matrix_game.inference.loading import clean_state_dict_keys


class LoadingTests(unittest.TestCase):
    def test_prefixes(self):
        value=torch.ones(2)
        result=clean_state_dict_keys({'_fsdp_wrapped_module.model.weight':value})
        self.assertIs(result['weight'],value)

    def test_reject_ambiguous_or_non_tensor_state(self):
        for value in ({}, [], {'a':'metadata'}, {'_orig_mod.a':torch.ones(1),'a':torch.zeros(1)}):
            with self.assertRaises(ValueError):clean_state_dict_keys(value)
