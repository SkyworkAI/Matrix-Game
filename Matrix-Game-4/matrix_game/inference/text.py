"""Resident T5 with fixed-length CUDA graph replay and bounded caption reuse."""
from collections import OrderedDict

import torch


class CaptionEncoder:
    def __init__(self, text, *, graph=False, capacity=8):
        self.text, self.graph_enabled = text, graph
        self.capacity, self.cache = capacity, OrderedDict()
        self.entry = None
        self.encodes = self.hits = 0

    @torch.no_grad()
    def __call__(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8192:
            raise ValueError("Expected a nonempty bounded caption")
        if prompt in self.cache:
            self.hits += 1
            self.cache.move_to_end(prompt)
            return self.cache[prompt]
        ids, mask = self.text.tokenizer([prompt], return_mask=True, add_special_tokens=True)
        model = self.text.text_encoder
        device = next(model.parameters()).device
        ids, mask = ids.to(device), mask.to(device)
        if ids.shape != (1, 512) or mask.shape != ids.shape:
            raise ValueError("Caption encoding must retain native 512-token padding")
        if self.graph_enabled:
            if device.type != "cuda" or model.training:
                raise ValueError("T5 graphs require CUDA eval mode")
            if self.entry is None:
                si, sm = ids.clone(), mask.clone()
                side = torch.cuda.Stream(device=device)
                side.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(side):
                    for _ in range(3):
                        model(si, sm)
                side.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=side):
                    output = model(si, sm)
                torch.cuda.current_stream(device).wait_stream(side)
                self.entry = graph, si, sm, output
            graph, si, sm, output = self.entry
            si.copy_(ids)
            sm.copy_(mask)
            graph.replay()
        else:
            output = model(ids, mask)
        # Independent storage is required when the next graph replay updates text.
        encoded = output.masked_fill(~mask.bool().unsqueeze(-1), 0)
        self.encodes += 1
        self.cache[prompt] = encoded
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return encoded
