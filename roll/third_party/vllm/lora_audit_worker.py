"""Audit-only vLLM worker for comparing file and in-memory LoRA loading."""

from collections import OrderedDict

import torch

from roll.third_party.vllm.vllm_utils import TensorLoRARequest
from roll.third_party.vllm.worker import WorkerV1


class LoRAAuditWorker(WorkerV1):
    def custom_load_lora_state(
        self,
        peft_config,
        tensors,
        lora_int_id=None,
        lora_name="training_lora",
    ):
        self.tensor_lora_manager.lora_params = OrderedDict(tensors)
        if lora_int_id is not None:
            request = TensorLoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path="/root/.cache/roll/training_lora_v1",
                peft_config=peft_config,
                lora_tensors=self.tensor_lora_manager.lora_params,
            )
            self.tensor_lora_manager.lora_params = OrderedDict()
            self.model_runner.remove_lora(lora_int_id)
            installed = self.model_runner.add_lora(request)
            # set_lora() copies the CPU adapter into the active GPU slot.
            # collective_rpc may otherwise return before that asynchronous
            # transfer is visible to the immediately following generation.
            torch.cuda.synchronize()
            return installed
        return self.custom_add_lora(peft_config)

    def custom_remove_lora(self, lora_id: int):
        """Remove an audit adapter so the next path reuses the same slot."""
        return self.model_runner.remove_lora(lora_id)

    def custom_snapshot_lora(self, lora_id: int, label: str):
        """Save the registered adapter tensors for an exact path comparison."""
        worker_manager = self.model_runner.lora_manager
        adapter = worker_manager._adapter_manager.get_adapter(lora_id)
        if adapter is None:
            raise RuntimeError(f"LoRA adapter {lora_id} is not registered")

        tensors = OrderedDict()
        for module_name, layer in sorted(adapter.loras.items()):
            for parameter_name in ("lora_a", "lora_b"):
                value = getattr(layer, parameter_name)
                values = value if isinstance(value, (list, tuple)) else [value]
                for index, tensor in enumerate(values):
                    if tensor is None:
                        continue
                    key = f"{module_name}.{parameter_name}.{index}"
                    tensors[key] = tensor.detach().float().cpu().clone()

        if not hasattr(self, "_lora_audit_snapshots"):
            self._lora_audit_snapshots = {}
        self._lora_audit_snapshots[label] = tensors
        return {
            "label": label,
            "tensor_count": len(tensors),
            "numel": sum(tensor.numel() for tensor in tensors.values()),
            "l2_norm_sum": sum(
                float(torch.linalg.vector_norm(tensor))
                for tensor in tensors.values()
            ),
        }

    def custom_snapshot_active_lora(self, lora_id: int, label: str):
        """Capture the actual GPU slot buffers consumed by Punica."""
        adapter_manager = self.model_runner.lora_manager._adapter_manager
        try:
            slot = adapter_manager.lora_index_to_id.index(lora_id)
        except ValueError as error:
            raise RuntimeError(f"LoRA adapter {lora_id} is not active") from error

        tensors = OrderedDict()

        def capture_value(prefix, value):
            if torch.is_tensor(value):
                tensors[prefix] = value.detach().float().cpu().clone()
                return
            if isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    if child is not None:
                        capture_value(f"{prefix}.{index}", child)

        for module_name, module in sorted(adapter_manager.modules.items()):
            capture_value(
                f"{module_name}.lora_a_stacked",
                module.lora_a_stacked[slot],
            )
            capture_value(
                f"{module_name}.lora_b_stacked",
                module.lora_b_stacked[slot],
            )

        if not hasattr(self, "_lora_audit_snapshots"):
            self._lora_audit_snapshots = {}
        self._lora_audit_snapshots[label] = tensors
        return {
            "label": label,
            "slot": slot,
            "tensor_count": len(tensors),
            "numel": sum(tensor.numel() for tensor in tensors.values()),
            "l2_norm_sum": sum(
                float(torch.linalg.vector_norm(tensor))
                for tensor in tensors.values()
            ),
        }

    def custom_compare_lora_snapshots(self, left_label: str, right_label: str):
        """Compare two snapshots captured after vLLM packing/activation."""
        snapshots = getattr(self, "_lora_audit_snapshots", {})
        left = snapshots[left_label]
        right = snapshots[right_label]
        left_keys = set(left)
        right_keys = set(right)
        common_keys = sorted(left_keys & right_keys)
        shape_mismatches = []
        differing = []
        max_abs_diff = 0.0
        mean_abs_diff_sum = 0.0
        compared = 0
        for key in common_keys:
            left_tensor = left[key]
            right_tensor = right[key]
            if left_tensor.shape != right_tensor.shape:
                shape_mismatches.append(
                    {
                        "key": key,
                        "left": list(left_tensor.shape),
                        "right": list(right_tensor.shape),
                    }
                )
                continue
            delta = (left_tensor - right_tensor).abs()
            tensor_max = float(delta.max())
            tensor_mean = float(delta.mean())
            compared += 1
            max_abs_diff = max(max_abs_diff, tensor_max)
            mean_abs_diff_sum += tensor_mean
            if tensor_max != 0.0 and len(differing) < 12:
                differing.append(
                    {
                        "key": key,
                        "max_abs_diff": tensor_max,
                        "mean_abs_diff": tensor_mean,
                        "left_norm": float(
                            torch.linalg.vector_norm(left_tensor)
                        ),
                        "right_norm": float(
                            torch.linalg.vector_norm(right_tensor)
                        ),
                    }
                )

        return {
            "left_label": left_label,
            "right_label": right_label,
            "left_tensor_count": len(left),
            "right_tensor_count": len(right),
            "missing_from_right": sorted(left_keys - right_keys)[:12],
            "missing_from_left": sorted(right_keys - left_keys)[:12],
            "shape_mismatches": shape_mismatches[:12],
            "compared_tensor_count": compared,
            "exactly_identical": (
                left_keys == right_keys
                and not shape_mismatches
                and max_abs_diff == 0.0
            ),
            "max_abs_diff": max_abs_diff,
            "mean_of_tensor_mean_abs_diff": (
                mean_abs_diff_sum / compared if compared else None
            ),
            "first_differing_tensors": differing,
        }
