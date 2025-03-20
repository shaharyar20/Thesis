import torch
import inspect
from dataclasses import dataclass
import modulus
from modulus.models.meta import ModelMetaData
from modulus.registry import ModelRegistry

from torchlbm.core.advance import AdvanceModule
from torchlbm.node_data import NodeData


def from_torchlbm(torch_model_class: torch.nn.Module, meta: ModelMetaData = None) -> modulus.Module:
    """Construct a Modulus module from a PyTorch module

    Parameters
    ----------
    torch_model_class : torch.nn.Module
        PyTorch module class
    meta : ModelMetaData, optional
        Meta data for the model, by default None

    Returns
    -------
    Module
    """

    # Define an internal class as before
    class ModulusModel(modulus.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(meta=meta)
            self.inner_model = torch_model_class(*args, **kwargs)

        def forward(self, x: NodeData):
            return self.inner_model(x)

    # Get the argument names and default values of the PyTorch model's init method
    init_argspec = inspect.getfullargspec(torch_model_class.__init__)
    model_argnames = init_argspec.args[1:]  # Exclude 'self'
    model_defaults = init_argspec.defaults or []
    defaults_dict = dict(zip(model_argnames[-len(model_defaults) :], model_defaults))

    # Define the signature of new init
    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    params += [
        inspect.Parameter(
            argname,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=defaults_dict.get(argname, inspect.Parameter.empty),
        )
        for argname in model_argnames
    ]
    init_signature = inspect.Signature(params)

    # Replace ModulusModel.__init__ signature with new init signature
    ModulusModel.__init__.__signature__ = init_signature

    # Generate a unique name for the created class
    new_class_name = f"{torch_model_class.__name__}ModulusModel"
    ModulusModel.__name__ = new_class_name

    # Add this class to the dict of models classes
    registry = ModelRegistry()
    registry.register(ModulusModel, new_class_name)

    return ModulusModel


@dataclass
class AdvanceMetaData(modulus.ModelMetaData):
    name: str = "Advance"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


ModulusAdvanceModel = from_torchlbm(AdvanceModule, meta=AdvanceMetaData())
