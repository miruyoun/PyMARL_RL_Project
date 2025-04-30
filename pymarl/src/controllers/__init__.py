REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC

from .mappo_controller import MAPPOController
REGISTRY["mappo_mac"] = MAPPOController