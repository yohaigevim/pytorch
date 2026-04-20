# Owner(s): ["module: inductor"]
import torch
import torch._inductor.config as config
from torch._inductor.hooks import node_hook
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU


class TestNodeHooks(TestCase):
    @config.patch(generate_node_hooks=True)
    def test_hooks_called_with_pointwise(self):
        """Hooks fire for pointwise (SIMD) kernels with correct phases."""
        calls: list[tuple[str, str]] = []

        def hook(phase, kernel_name, tensors):
            calls.append((phase, kernel_name))

        def fn(x, y):
            return x + y

        opt = torch.compile(fn)
        x = torch.randn(8)
        y = torch.randn(8)

        with node_hook(hook):
            opt(x, y)

        pre_calls = [c for c in calls if c[0] == "pre"]
        post_calls = [c for c in calls if c[0] == "post"]
        self.assertGreater(len(pre_calls), 0, "Expected at least one pre hook call")
        self.assertGreater(len(post_calls), 0, "Expected at least one post hook call")

    @config.patch(generate_node_hooks=True)
    def test_hooks_called_with_extern_kernel(self):
        """Hooks fire for extern kernels (e.g. mm)."""
        calls: list[tuple[str, str, list]] = []

        def hook(phase, kernel_name, tensors):
            calls.append((phase, kernel_name, tensors))

        def fn(x, y):
            return torch.mm(x, y)

        opt = torch.compile(fn)
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with node_hook(hook):
            opt(x, y)

        pre_calls = [c for c in calls if c[0] == "pre"]
        post_calls = [c for c in calls if c[0] == "post"]
        self.assertGreater(len(pre_calls), 0, "Expected pre hook for mm")
        self.assertGreater(len(post_calls), 0, "Expected post hook for mm")

    @config.patch(generate_node_hooks=True)
    def test_pre_hook_receives_tensors(self):
        """Pre hook receives actual tensor objects for extern kernels."""
        received_tensors: list[list] = []

        def hook(phase, kernel_name, tensors):
            if phase == "pre":
                received_tensors.append(tensors)

        def fn(x, y):
            return torch.mm(x, y)

        opt = torch.compile(fn)
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with node_hook(hook):
            opt(x, y)

        self.assertGreater(len(received_tensors), 0)
        for tensor_list in received_tensors:
            for t in tensor_list:
                self.assertIsInstance(t, torch.Tensor)

    @config.patch(generate_node_hooks=True)
    def test_post_hook_receives_tensors(self):
        """Post hook receives actual tensor objects."""
        received_tensors: list[list] = []

        def hook(phase, kernel_name, tensors):
            if phase == "post":
                received_tensors.append(tensors)

        def fn(x, y):
            return x + y

        opt = torch.compile(fn)
        x = torch.randn(8)
        y = torch.randn(8)

        with node_hook(hook):
            opt(x, y)

        self.assertGreater(len(received_tensors), 0)
        for tensor_list in received_tensors:
            for t in tensor_list:
                self.assertIsInstance(t, torch.Tensor)

    def test_hooks_not_emitted_when_disabled(self):
        """No hook calls when generate_node_hooks is False (default)."""
        calls: list[tuple] = []

        def hook(phase, kernel_name, tensors):
            calls.append((phase, kernel_name))

        def fn(x, y):
            return x + y

        opt = torch.compile(fn)
        x = torch.randn(8)
        y = torch.randn(8)

        with node_hook(hook):
            opt(x, y)

        self.assertEqual(len(calls), 0, "Hooks should not fire when config is off")

    @config.patch(generate_node_hooks=True)
    def test_hook_context_manager_cleanup(self):
        """node_hook properly removes hooks on exit."""
        from torch._inductor.hooks import NODE_HOOKS

        def hook(phase, kernel_name, tensors):
            pass

        self.assertEqual(len(NODE_HOOKS), 0)
        with node_hook(hook):
            self.assertEqual(len(NODE_HOOKS), 1)
        self.assertEqual(len(NODE_HOOKS), 0)

    @config.patch(generate_node_hooks=True)
    def test_multiple_hooks(self):
        """Multiple registered hooks all get called."""
        calls_a: list[str] = []
        calls_b: list[str] = []

        def hook_a(phase, kernel_name, tensors):
            calls_a.append(phase)

        def hook_b(phase, kernel_name, tensors):
            calls_b.append(phase)

        def fn(x, y):
            return x + y

        opt = torch.compile(fn)
        x = torch.randn(8)
        y = torch.randn(8)

        with node_hook(hook_a), node_hook(hook_b):
            opt(x, y)

        self.assertGreater(len(calls_a), 0)
        self.assertGreater(len(calls_b), 0)
        self.assertEqual(len(calls_a), len(calls_b))


if __name__ == "__main__":
    if HAS_CPU:
        run_tests()
