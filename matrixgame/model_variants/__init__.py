from .matrixgame_dit_src import MGVideoDiffusionTransformerI2V



def get_dit(model_name, config_path, weight_dtype):
    if model_name == 'matrixgame':
        return MGVideoDiffusionTransformerI2V.from_config(config_path).to(weight_dtype), MGVideoDiffusionTransformerI2V
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')


   