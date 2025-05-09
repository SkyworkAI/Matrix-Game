from .matrixgame_i2v import MatrixGameEncoderWrapperI2V
def get_text_enc(enc_name, model_path, weight_dtype, i2v_type):
    if enc_name == 'matrixgame':
        return MatrixGameEncoderWrapperI2V(model_path, weight_dtype, task = 'i2v', i2v_type = i2v_type)
    else:
        raise NotImplementedError(f'{enc_name} is not implemented.')