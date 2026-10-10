"""Check precision selection and checkpoint recasting without ML dependencies."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
model_tree = ast.parse((root / "SFT/HumanML3D/models/mllm_single_lora.py").read_text(encoding="utf-8"))
model_class = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == "MotionLLM")
constructor = next(n for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
dtype_assign = next(n for n in constructor.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr == "llm_dtype" for t in n.targets))
activate = next(n for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name == "activate_full_model")
torch = SimpleNamespace(bfloat16="bfloat16", float16="float16")


class LLM:
    def __init__(self):
        self.dtype = None
        self.merged = False
        self.param = SimpleNamespace(requires_grad=False)

    def merge_and_unload(self):
        self.merged = True
        return self

    def to(self, *, dtype):
        self.dtype = dtype
        return self

    def named_parameters(self):
        return [("weight", self.param)]


# Exercise the actual merge method for the legacy default and T4 configuration.
for args in (SimpleNamespace(), SimpleNamespace(llm_dtype="float16")):
    for has_adapter in (True, False):
        self = SimpleNamespace(llm=LLM(), w_lora=has_adapter)
        namespace = {"torch": torch, "args": args, "self": self}
        module = ast.Module(body=[copy.deepcopy(dtype_assign), copy.deepcopy(activate)], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "precision-selection", "exec"), namespace)
        namespace["activate_full_model"](self)
        expected = getattr(args, "llm_dtype", "bfloat16")
        assert self.llm.dtype == expected
        assert self.llm.merged == has_adapter
        assert self.w_lora is False


class VQ:
    dtype = "float32"

    def float(self):
        self.dtype = "float32"
        return self


class FakeMotionLLM:
    def __init__(self, args):
        self.llm_dtype = args.llm_dtype
        self.llm = LLM()
        self.net = VQ()

    def activate_full_model(self):
        self.llm.to(dtype=self.llm_dtype)

    def load_model(self, path, verbose):
        # Simulate upstream param.data assignments restoring BF16 from disk.
        self.llm.dtype = "bfloat16"
        return {}, {"weight"}, set()

    def to(self, device):
        assert self.llm.dtype == self.llm_dtype, "Checkpoint dtype was not corrected before GPU transfer"
        assert self.net.dtype == "float32", "VQ precision changed"
        return self

    def eval(self):
        return self


tree = ast.parse((root / "SFT/HumanML3D/run_prompt_baseline.py").read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
             and n.value.func.id == "MotionLLM")
end = next(i for i, n in enumerate(main.body[start:], start) if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "processor" for t in n.targets))
block = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(main.body[start:end]), type_ignores=[]))
for dtype in ("float16", "bfloat16"):
    namespace = {"MotionLLM": FakeMotionLLM, "args": SimpleNamespace(
        llm_dtype=dtype, device="cuda:0", eval_task="unified_mogen_cot_v3",
        generation_mode="think_w_analysis_multi_round_gen"), "checkpoint": Path("fake.pth")}
    exec(compile(block, "checkpoint-precision", "exec"), namespace)
    assert namespace["model"].llm.dtype == dtype
print("Precision checks passed: BF16 default, FP16 merge, checkpoint recast, FP32 VQ")
