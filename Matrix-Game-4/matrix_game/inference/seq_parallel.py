"""Single-device compatibility for the upstream backbone's optional SP hooks.

Distributed training and sequence-parallel inference are outside this release.
"""


def sp_size():
    return 1


def init_sequence_parallel(size):
    if int(size) != 1:
        raise ValueError("This release supports single-GPU inference only")
