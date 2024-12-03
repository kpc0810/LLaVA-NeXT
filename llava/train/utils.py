import torch
import weakref
import numpy as np
from transformers.training_args import TrainingArguments
from numpy import typing as npt
from collections import defaultdict
from typing import NewType, Callable, Dict, List, Optional, Union

Number = NewType('Number', (int, float))

class TrainLoggingState:
    """Refer to https://github.com/zipzou/hf-multitask-trainer.git
    """
    def __init__(self, args: TrainingArguments) -> None:
        self.metrics: Dict[str, List[Union[Number, torch.Tensor,npt.NDArray]]] = defaultdict(list)
        self.args = weakref.ref(args)
    def add_metrics(self, **metrics: Union[Number, torch.Tensor, npt.NDArray]):
        for k, v in metrics.items():
            self.metrics[k].append(v)
    def get_metrics(
        self,
        step_scale: float = 1.0,
        gather_func: Optional[Callable[[Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor]] = None,
        round_digits: Optional[int] = None
    ) -> Dict[str, Number]:
        metrics: Dict[str, List[Number]] = defaultdict(list)
        for k, values in self.metrics.items():
            for value in values:
                if isinstance(value, torch.Tensor):
                    if gather_func is not None:
                        value = gather_func(value).mean().item()
                    else:
                        value = value.mean().cpu().item()
                    val = value
                    val = val / self.args().gradient_accumulation_steps
                elif isinstance(value, (int, float)):
                    val = value
                    val = val / self.args().gradient_accumulation_steps
                elif isinstance(value, np.ndarray):
                    val = value.mean().item()
                    val = val / self.args().gradient_accumulation_steps
                else:
                    val = value
                metrics[k].append(val)
        step_metrics = {
            k: sum(v) / (len(v) / self.args().gradient_accumulation_steps)
            for k, v in metrics.items()
        }
        if round_digits is not None:
            step_metrics = {k: round(v, round_digits)for k, v in step_metrics.items()}
        return step_metrics
    def pop_metrics(
        self,
        gather_func: Optional[Callable[[Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor]] = None,
        round_digits: Optional[int] = None
    ):
        ret = self.get_metrics(gather_func, round_digits)
        self.clear()
        return ret
    def clear(self):
        self.metrics.clear()