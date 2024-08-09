from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List

from diffusers import UNet2DConditionModel

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage


@dataclass
class CallbackMetadata:
    callback_id: str
    order: int


@dataclass
class CallbackFunctionWithMetadata:
    metadata: CallbackMetadata
    function: Callable[[DenoiseContext], None]


def callback(callback: Callable[..., None], order: int = 0):
    def _decorator(function):
        function._ext_metadata = CallbackMetadata(
            callback_id=callback.__qualname__,
            order=order,
        )
        return function

    return _decorator


@dataclass
class OverrideMetadata:
    override_id: str


@dataclass
class OverrideFunctionWithMetadata:
    metadata: OverrideMetadata
    function: Callable[..., Any]


def override(override: Callable[..., None]):
    def _decorator(function):
        function._ext_metadata = OverrideMetadata(
            override_id=override.__qualname__,
        )
        return function

    return _decorator


class ExtensionBase:
    def __init__(self):
        self._callbacks: Dict[str, List[CallbackFunctionWithMetadata]] = {}
        self._overrides: Dict[str, OverrideFunctionWithMetadata] = {}

        # Register all of the callback methods for this instance.
        for func_name in dir(self):
            func = getattr(self, func_name)
            metadata = getattr(func, "_ext_metadata", None)
            if metadata is not None:
                if isinstance(metadata, CallbackMetadata):
                    if metadata.callback_id not in self._callbacks:
                        self._callbacks[metadata.callback_id] = []
                    self._callbacks[metadata.callback_id].append(CallbackFunctionWithMetadata(metadata, func))
                elif isinstance(metadata, OverrideMetadata):
                    if metadata.override_id in self._overrides:
                        raise RuntimeError(
                            f"Override {metadata.override_id} defined multiple times in {type(self).__qualname__}"
                        )
                    self._overrides[metadata.override_id] = OverrideFunctionWithMetadata(metadata, func)

    def get_callbacks(self):
        return self._callbacks

    def get_overrides(self):
        return self._overrides

    @contextmanager
    def patch_extension(self, ctx: DenoiseContext):
        yield None

    @contextmanager
    def patch_unet(self, unet: UNet2DConditionModel, original_weights: OriginalWeightsStorage):
        """A context manager for applying patches to the UNet model. The context manager's lifetime spans the entire
        diffusion process. Weight unpatching is handled upstream, and is achieved by saving unchanged weights by
        `original_weights.save` function. Note that this enables some performance optimization by avoiding redundant
        operations. All other patches (e.g. changes to tensor shapes, function monkey-patches, etc.) should be unpatched
        by this context manager.

        Args:
            unet (UNet2DConditionModel): The UNet model on execution device to patch.
            original_weights (OriginalWeightsStorage): A storage with copy of the model's original weights in CPU, for
                unpatching purposes. Extension should save tensor which being modified in this storage, also extensions
                can access original weights values.
        """
        yield
