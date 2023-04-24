from typing import List

from torch import Tensor, nn


def unpack_a_list(list_of_stuff: List, device) -> List:
    return [stuff.to(device) if type(stuff) == Tensor else unpack_a_list(stuff, device) for stuff in list_of_stuff]


class NamedModuleList(nn.ModuleList):
    def __init__(self, network_list: nn.ModuleList, name: str):
        super().__init__(network_list)
        self.__name = name

    def get_name(self):
        return self.__name
