import typing 

class Module:

    def __init__(self) -> None:
        pass

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return self.forward(args, kwargs)

    def forward(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise NotImplementedError("Inherting from `axon.Module` requires the forward method to be implemented.")
