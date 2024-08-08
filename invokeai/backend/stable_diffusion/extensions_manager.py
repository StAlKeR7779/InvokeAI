from __future__ import annotations

from functools import partial
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch
from diffusers import UNet2DConditionModel

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.stable_diffusion.diffusion.custom_atttention import AttentionContext
    from invokeai.backend.stable_diffusion.extensions.base import CallbackFunctionWithMetadata, ExtensionBase


class CallbackApi:
    def __init__(self, handler: Callable[..., None]):
        self._handler = handler

    def setup(self, ctx: DenoiseContext):
        pass

    def pre_denoise_loop(self, ctx: DenoiseContext):
        pass

    def post_denoise_loop(self, ctx: DenoiseContext):
        pass

    def pre_step(self, ctx: DenoiseContext):
        pass

    def post_step(self, ctx: DenoiseContext):
        pass

    def pre_unet_forward(self, ctx: DenoiseContext):
        pass

    def post_unet_forward(self, ctx: DenoiseContext):
        pass

    def post_combine_noise_preds(self, ctx: DenoiseContext):
        pass

    def pre_prepare_attention_mask(self, ctx: AttentionContext):
        pass

    def post_prepare_attention_mask(self, ctx: AttentionContext):
        pass

    def pre_run_attention(self, ctx: AttentionContext):
        pass

    def post_run_attention(self, ctx: AttentionContext):
        pass

    def __getattribute__(self, name: str):
        if name.startswith("_"):
            return super().__getattribute__(name)

        func = type(self).__dict__.get(name, None)
        if func is not None:
            return partial(self._handler, func.__qualname__)

        raise AttributeError(f"Callback \"{name}\" is not defined!")


class ExtensionsManager:
    def __init__(self, is_canceled: Optional[Callable[[], bool]] = None):
        self._is_canceled = is_canceled

        self.callback = CallbackApi(self._run_callback)

        # A list of extensions in the order that they were added to the ExtensionsManager.
        self._extensions: List[ExtensionBase] = []
        self._ordered_callbacks: Dict[ExtensionCallbackType, List[CallbackFunctionWithMetadata]] = {}

    def add_extension(self, extension: ExtensionBase):
        self._extensions.append(extension)
        self._regenerate_ordered_callbacks()

    def _regenerate_ordered_callbacks(self):
        """Regenerates self._ordered_callbacks. Intended to be called each time a new extension is added."""
        self._ordered_callbacks = {}

        # Fill the ordered callbacks dictionary.
        for extension in self._extensions:
            for callback_id, callbacks in extension.get_callbacks().items():
                if callback_id not in self._ordered_callbacks:
                    self._ordered_callbacks[callback_id] = []
                self._ordered_callbacks[callback_id].extend(callbacks)

        # Sort each callback list.
        for callback_id, callbacks in self._ordered_callbacks.items():
            # Note that sorted() is stable, so if two callbacks have the same order, the order that they extensions were
            # added will be preserved.
            self._ordered_callbacks[callback_id] = sorted(callbacks, key=lambda x: x.metadata.order)

    def _run_callback(self, callback_id: str, *args, **kwargs):
        if self._is_canceled and self._is_canceled():
            raise CanceledException

        callbacks = self._ordered_callbacks.get(callback_id, [])
        for cb in callbacks:
            cb.function(*args, **kwargs)

    @contextmanager
    def patch_extensions(self, ctx: DenoiseContext):
        if self._is_canceled and self._is_canceled():
            raise CanceledException

        with ExitStack() as exit_stack:
            for ext in self._extensions:
                exit_stack.enter_context(ext.patch_extension(ctx))

            yield None

    @contextmanager
    def patch_unet(self, unet: UNet2DConditionModel, cached_weights: Optional[Dict[str, torch.Tensor]] = None):
        if self._is_canceled and self._is_canceled():
            raise CanceledException

        original_weights = OriginalWeightsStorage(cached_weights)
        try:
            with ExitStack() as exit_stack:
                for ext in self._extensions:
                    exit_stack.enter_context(ext.patch_unet(unet, original_weights))

                yield None

        finally:
            with torch.no_grad():
                for param_key, weight in original_weights.get_changed_weights():
                    unet.get_parameter(param_key).copy_(weight)
