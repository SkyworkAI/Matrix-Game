from .matrixgame_vae import get_mg_vae_wrapper
def get_vae(vae_name, model_path, weight_dtype):
    if vae_name == 'matrixgame':
        return get_mg_vae_wrapper(model_path, weight_dtype)
   
