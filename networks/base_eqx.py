import dataclasses

import equinox as eqx
import optax

class TrainState(eqx.Module):
    model: eqx.Module
    optim: optax.GradientTransformation
    optim_state: optax.OptState
    
    @classmethod
    def create(cls, model, optim, **kwargs):
        optim_state = optim.init(eqx.filter(model,  eqx.is_array))
        return cls(model=model, optim=optim, optim_state=optim_state, **kwargs)
    
    #@eqx.filter_jit
    def apply_updates(self, grads):
        updates, new_optim_state = self.optim.update(grads, self.optim_state)
        new_model = eqx.apply_updates(self.model, updates)
        return dataclasses.replace(
            self,
            model=new_model,
            optim_state=new_optim_state
        )
        
    def evaluate(self, input, sample_key):
        return self.model(input).sample(seed=sample_key)

class TrainTargetState(TrainState):
    target_model: eqx.Module
    tau: float = 0.005
    
    @classmethod
    def create(cls, *, model, target_model, optim, **kwargs):
        optim_state = optim.init(eqx.filter(model, eqx.is_array))
        return cls(model=model, optim=optim, optim_state=optim_state, target_model=target_model,
                   **kwargs)

    @eqx.filter_jit
    def soft_update(self):
        model_params = eqx.filter(self.model, eqx.is_array)
        target_model_params, target_model_static = eqx.partition(self.target_model, eqx.is_array)
        new_target_params = optax.incremental_update(model_params, target_model_params, self.tau)
        
        return dataclasses.replace(
            self,
            model=self.model,
            target_model=eqx.combine(new_target_params, target_model_static)
        )
        