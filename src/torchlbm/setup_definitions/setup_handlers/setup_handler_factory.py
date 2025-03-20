# own modules
from torchlbm.exceptions import SetupError
from .setup_handler_type import SetupHandlerType
from .setup_handler import SetupHandler
from .json_setup_handler import JSONSetupHandler
from .xml_setup_handler import XMLSetupHandler


def instantiate_setup_handler(setup_handler_type: SetupHandlerType) -> SetupHandler:
    """Gives the appropriate setup handler for a certain setup handler type.

    Args:
        setup_handler_type (SetupHandlerType): The corresponding setup handler type to be used.

    Returns:
        SetupHandler: The instantiated setup handler.
    """
    if not isinstance(setup_handler_type, SetupHandlerType):
        raise SetupError("Cannot instantiate setup handler. Type must be of 'SetupHandlerType'")
    if setup_handler_type == SetupHandlerType.json:
        return JSONSetupHandler()
    if setup_handler_type == SetupHandlerType.yaml:
        return JSONSetupHandler()
    else:
        return XMLSetupHandler()
