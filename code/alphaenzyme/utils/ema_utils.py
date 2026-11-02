from collections import OrderedDict
import copy
import torch
import alphaenzyme.core.core as core
from alphaenzyme.core.core import Registry as R
from alphaenzyme.core.core import _Configurable
from alphaenzyme.utils.logging_utils import LoggerWrapper
logger = LoggerWrapper

from alphaenzyme.utils.tensor_utils import tensor_tree_map


@R.register('utils.EMAWrapperV1')
class EMAWrapperV1(core.Configurable):
    """A wrapper class for exponential moving average of model weights."""
    # The EMA Wrapper from Protenix

    def __init__(self, configs):
        """
        model: a pytorch model to apply EMA
        decay: a scaler to indicate the decay rate
        mutable_param_keywords: keywords of parameters to apply EMA decay, other params will stay untouched
        """
        super(EMAWrapperV1, self).__init__()
        self.configs = configs
        self.decay = self.configs.get("decay", 0.99)
        mutable_param_keywords = self.configs.get("mutable_param_keywords", None)
        if mutable_param_keywords is None:
            self.mutable_param_keywords = []
        else:
            self.mutable_param_keywords = [s.strip() for s in mutable_param_keywords if s.strip()]

        self.shadow = {}
        self.backup = {}

    # def set_model(self, model: torch.nn.Module):
    #     self.model = model


    def register(self):
        # clone the parameters of self.model into self.shadow, i.e. the shadow model maintained in the EMAWrapper.
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()


    def update(self):
        # update the params of s
        for name, param in self.model.named_parameters():
            if self.mutable_param_keywords and not any(
                [keyword in name for keyword in self.mutable_param_keywords]
            ):
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[
                name
            ]
            self.shadow[name] = new_average.clone()


    def apply_shadow(self):
        # copy param.data to backup, and copy shadow to param.data
        for name, param in self.model.named_parameters():
            assert name in self.shadow
            self.backup[name] = param.data
            param.data = self.shadow[name]


    def restore(self):
        # copy backup to param.data, and clear the backup.
        for name, param in self.model.named_parameters():
            assert name in self.backup
            param.data = self.backup[name]
        self.backup = {}

    @property
    def state_dict(self):
        return OrderedDict(
            {"params": self.shadow.items(), "config": self.configs}
        )



@R.register('utils.EMAWrapperV2')
class EMAWrapperV2(core.Configurable):
    """A wrapper class for exponential moving average of model weights."""
    # The EMA Wrapper from BM-scoring

    def __init__(self, configs):
        super(EMAWrapperV2, self).__init__()
        self.configs = configs
        self.decay = self.configs.get("decay", 0.99)

    def set_model(self, model: torch.nn.Module):
        # the parameters of the model are cloned to self.params
        # the model is not maintained in the EMAWrapperV2
        clone_param = lambda t: t.clone().detach()
        self.params = tensor_tree_map(clone_param, model.state_dict())
        self.device = next(model.parameters()).device

    def to(self, device):
        # move the params in self.params to the device
        self.params = tensor_tree_map(lambda t: t.to(device), self.params)
        self.device = device

    def _update_state_dict(self, update, state_dict):
        # update the parameters in self.params by update. update is the new params from that model, and state_dict is the params in ema wrapper.
        # update: New state dict of the same model
        # state_dict: self.params
        with torch.no_grad():
            for k, v in update.items():
                stored = state_dict[k]
                if not isinstance(v, torch.Tensor):
                    self._update_state_dict(v, stored)     # recursively find the tensors
                else:
                    # update the params
                    diff = stored - v
                    diff = (
                        diff.float()
                    )
                    diff *= (
                        1 - self.decay
                    )
                    stored = (
                        stored.float()
                    )
                    stored -= diff

    def update(self, model):
        # update the params in EMAWrapperV2 by the new params of model
        # the entrance for updating params stored in the EMAWrapperV2.
        self._update_state_dict(model.state_dict(), self.params)

    def load_state_dict(self, state_dict):
        # Load the params of EMAWrapperV2 saved in the ckpt.
        self.params = state_dict['params']
        self.configs = state_dict['config']

        decay = self.configs.get("decay", 0.99)
        if decay != self.decay:
            logger.warning(f"The decay in the configs is not the same as the ckpt. Please check!!! Using the decay in ckpt as default: {decay}")
            self.decay = decay

    def apply_to_model(self, model):
        assert hasattr(self, "cached_weights"), "You must backup the model params first!!!"

        missing_keys, unexpected_keys = model.load_state_dict(self.params, strict=False)
        assert len(missing_keys) == 0
        assert len(unexpected_keys) == 0


    def backup(self, model):
        assert not hasattr(self, "cached_weights"), "Cached weights are storing in the EMAWrapper. You should call .restore(model) or .clear_cache() after calling .backup(model)."

        clone_param = lambda t: t.clone().detach()
        self.cached_weights = tensor_tree_map(clone_param, model.state_dict())

    def restore(self, model):
        # restore the cached_weights back to the model
        # THIS METHOD WILL MODIFY THE PARAMS OF THE MODEL!!! CAREFULLY USE!!!
        assert hasattr(self, "cached_weights"), "There is no cached weights storing in the EMAWrapper. You should call .backup(model) first."

        missing_keys, unexpected_keys = model.load_state_dict(self.cached_weights, strict=False)
        assert len(missing_keys) == 0
        assert len(unexpected_keys) == 0
        self.clear_cache()

    def clear_cache(self):
        del self.cached_weights

    def state_dict(self):
        return OrderedDict(
            {"params": self.params, "config": self.configs}
        )

    def _check_class(self):
        for k, v in self.params.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Parameter {k} is not a torch.Tensor")

    def __repr__(self):
        self._check_class()
        param_str = ""
        for k, v in self.params.items():
            param_str += f"{k}: {v.size()}\n"

        return f"EMAWrapperV2(configs={self.configs}, decay={self.decay}, params=\n{param_str})"