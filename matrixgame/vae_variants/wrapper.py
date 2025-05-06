class VAEWrapper():
    def __init__(self, vae):
        self.vae = vae

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.vae, name)
    
    def encode(self, x):
        raise NotImplementedError
    
    def decode(self, latents):
        return NotImplementedError