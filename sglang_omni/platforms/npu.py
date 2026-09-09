from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.interface import OmniPlatform

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig


class NPUOmniPlatform(OmniPlatform):
    _enum: PlatformEnum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("npu", local_rank)

    def set_device(self, device: "torch.device") -> None:
        torch.npu.set_device(device)

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Keep every card visible, preserving a group-wide visibility list.

        Ranks bind their card through ``gpu_id`` (``torch.npu.set_device``) and
        HCCL needs every rank's peers discoverable, so this never narrows
        ``ASCEND_RT_VISIBLE_DEVICES`` per rank -- it only validates that an
        inherited list can still host the whole TP group.
        """
        if spec.tp_size <= 1:
            return {}
        if spec.gpu_id is None:
            raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")

        updates = {"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false"}
        source_env = env if env is not None else os.environ
        visible_devices = source_env.get("ASCEND_RT_VISIBLE_DEVICES")
        if visible_devices is None:
            return updates

        # An empty ASCEND_RT_VISIBLE_DEVICES disables every NPU rather than
        # exposing all of them (same semantics as vllm-ascend): fail fast
        # instead of letting ranks die later at device binding.
        visible_devices = visible_devices.strip()
        if not visible_devices:
            raise ValueError(
                "ASCEND_RT_VISIBLE_DEVICES is set to an empty string, which "
                "disables every NPU: unset it or list the cards to expose."
            )

        visible = [item.strip() for item in visible_devices.split(",") if item.strip()]
        if len(visible) < spec.tp_size:
            raise ValueError(
                f"tp stage {spec.stage_name!r} needs tp_size={spec.tp_size} cards, "
                f"but ASCEND_RT_VISIBLE_DEVICES={visible_devices!r} exposes "
                f"{len(visible)}. Widen the variable to cover the whole TP group: "
                "every rank must see its peers for HCCL discovery, and narrowing "
                "it per rank instead would relocate the stage onto different "
                "physical cards."
            )
        if spec.gpu_id >= len(visible):
            raise ValueError(
                f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, but "
                f"ASCEND_RT_VISIBLE_DEVICES={visible_devices!r} exposes only "
                f"{len(visible)} cards ({', '.join(visible)}). gpu_id indexes "
                "into the variable, not the host."
            )
        return updates

    def enable_code2wav_graph(self):
        return False

    def supports_torchaudio_resample(self) -> bool:
        """Disabled as it run on CPU and faced errors during inference for now"""
        return False
