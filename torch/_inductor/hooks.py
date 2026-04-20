# mypy: allow-untyped-defs
import contextlib
from collections.abc import Callable, Iterator
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import torch

# Executed in the order they're registered
INTERMEDIATE_HOOKS: list[Callable[[str, "torch.Tensor"], None]] = []


@contextlib.contextmanager
def intermediate_hook(fn):
    INTERMEDIATE_HOOKS.append(fn)
    try:
        yield
    finally:
        INTERMEDIATE_HOOKS.pop()


def run_intermediate_hooks(name, val):
    global INTERMEDIATE_HOOKS
    hooks = INTERMEDIATE_HOOKS
    INTERMEDIATE_HOOKS = []
    try:
        for hook in hooks:
            hook(name, val)
    finally:
        INTERMEDIATE_HOOKS = hooks


NODE_HOOKS: list[Callable[[str, str, list[Any]], None]] = []


@contextlib.contextmanager
def node_hook(
    fn: Callable[[str, str, list[Any]], None],
) -> Iterator[None]:
    """Register a hook called before/after each compiled node execution.

    The callback signature is ``fn(phase, kernel_name, tensors)`` where
    *phase* is ``"pre"`` or ``"post"``.
    """
    NODE_HOOKS.append(fn)
    try:
        yield
    finally:
        NODE_HOOKS.pop()


def run_pre_node_hook(node_name: str, inputs: list[Any]) -> None:
    """Called from generated code before each node execution."""
    global NODE_HOOKS
    hooks = NODE_HOOKS
    NODE_HOOKS = []
    try:
        for hook in hooks:
            hook("pre", node_name, inputs)
    finally:
        NODE_HOOKS = hooks


def run_post_node_hook(node_name: str, outputs: list[Any]) -> None:
    """Called from generated code after each node execution."""
    global NODE_HOOKS
    hooks = NODE_HOOKS
    NODE_HOOKS = []
    try:
        for hook in hooks:
            hook("post", node_name, outputs)
    finally:
        NODE_HOOKS = hooks
