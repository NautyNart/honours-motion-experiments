"""Lazy adapter to the repository's frozen IRG model and its original task templates."""
from contextlib import contextmanager
import os
from pathlib import Path
import random
import re
import sys
import time
import numpy as np
from .common import REPO, file_hash, digest, recover_xyz

@contextmanager
def upstream_directory():
    old = Path.cwd()
    path = REPO / 'SFT/HumanML3D'
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)

def arguments():
    from options.option_llm import get_args_parser
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        return get_args_parser()
    finally:
        sys.argv = argv

def parse_motion(text, nb_code=512, first=False):
    blocks = re.findall(r'<Motion>(.*?)</Motion>', text, flags=re.S)
    if not blocks:
        raise ValueError('No complete motion block; no dummy-token fallback permitted')
    block = blocks[0] if first else blocks[-1]
    if re.sub(r'<Motion_\d+>', '', block).strip():
        raise ValueError('Malformed motion block')
    tokens = [int(v) for v in re.findall(r'<Motion_(\d+)>', block)]
    if not tokens or min(tokens) < 0 or max(tokens) >= nb_code:
        raise ValueError('Invalid motion code indices')
    return np.asarray(tokens, dtype=np.int64)

class IRGBackend:
    def __init__(self, config):
        try:
            self._initialize(config)
        except Exception as exc:
            if hasattr(self, 'gpu'):
                exc.gpu_session = dict(command='failed_model_load', identity={'gpu': self.gpu},
                    elapsed_seconds=time.perf_counter() - self.started,
                    allocated_process_gpu_hours=(time.perf_counter() - self.started) / 3600,
                    used_gpu_count=1, error=str(exc))
            raise

    def _initialize(self, config):
        self.started = time.perf_counter()
        self.config = config['model']
        ckpt = Path(self.config['checkpoint'])
        backbone = Path(self.config['backbone'])
        ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
        backbone = backbone if backbone.is_absolute() else REPO / backbone
        assets = [REPO / 'SFT/HumanML3D/ckpt/vqvae.pth',
                  REPO / 'SFT/HumanML3D/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy',
                  REPO / 'SFT/HumanML3D/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy']
        for path in [ckpt, backbone, *assets]:
            if not path.exists():
                raise FileNotFoundError(f'Missing model asset: {path}; see pilot/README.md')
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('IRG editing requires CUDA; CPU measurements do not')
        self.device = 'cuda:0'
        self.gpu = torch.cuda.get_device_name(0)
        dtype = self.config['dtype']
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        if dtype == 'auto':
            dtype = 'bfloat16' if native_bf16 else 'float16'
        if dtype == 'bfloat16' and not native_bf16:
            raise ValueError('Selected GPU lacks native BF16; select auto or float16')
        if self.config['candidate_budget'] != 1:
            raise ValueError('This pilot implements exactly one editor candidate per round')
        torch.cuda.reset_peak_memory_stats()
        with upstream_directory():
            from models.mllm_single_lora import MotionLLM
            from dataset.data_processor import TextProcessor
            args = arguments()
            args.device, args.llm_backbone, args.llm_dtype = self.device, str(backbone), dtype
            args.eval_task, args.prompt_w_response = 'unified_mogen_cot_v3', True
            args.generation_mode = 'think_w_analysis_multi_round_gen'
            model = MotionLLM(args)
            model.activate_full_model()
            _, loaded, missing = model.load_model(str(ckpt), verbose=True)
            if missing:
                raise ValueError(f'Checkpoint left parameters unloaded: {sorted(missing)[:8]}')
            model.llm.to(dtype=model.llm_dtype)
            model.net.float()
            self.model = model.to(self.device).eval()
            self.model.requires_grad_(False)
            self.processor = TextProcessor(args)
            self.args = args
        self.identity = dict(checkpoint_sha256=file_hash(ckpt),
                             codec_sha256=file_hash(assets[0]),
                             mean_sha256=file_hash(assets[1]), std_sha256=file_hash(assets[2]),
                             backbone_config_sha256=file_hash(backbone / 'config.json'),
                             checkpoint_name=ckpt.name, backbone_name=backbone.name,
                             torch=torch.__version__, numpy=np.__version__, gpu=self.gpu,
                             llm_dtype=dtype, vq_dtype='float32', loaded_parameters=len(loaded))
        self.codec_id = digest({k: v for k, v in self.identity.items() if k.startswith(('codec_', 'mean_', 'std_'))})
        self.load_seconds = time.perf_counter() - self.started
        self.max_allocated_gib = torch.cuda.max_memory_allocated() / 2**30

    def decode(self, tokens):
        torch = self.torch
        with torch.inference_mode():
            indices = torch.as_tensor(tokens, device=self.device, dtype=torch.long).reshape(1, -1)
            if indices.numel() == 0 or not ((indices >= 0) & (indices < self.args.nb_code)).all():
                raise ValueError('Motion tokens out of range')
            normalized = self.model.net.forward_decoder(indices)
            features = self.model.motion_denorm(normalized.float()).float().cpu().numpy()[0]
        self.max_allocated_gib = max(self.max_allocated_gib, torch.cuda.max_memory_allocated() / 2**30)
        recover_xyz(features)  # validate before writing an output
        return features

    def encode(self, features):
        torch = self.torch
        # VQ downsamples by 4 with the upstream defaults. Explicitly record truncation.
        unit = self.args.stride_t ** self.args.down_t
        length = len(features) // unit * unit
        if length < unit:
            raise ValueError('Clip is too short for the tokenizer')
        mean, std = self.model.mean.astype(np.float32), self.model.std.astype(np.float32)
        if np.any(std <= 0):
            raise ValueError('Invalid pretrained normalization statistics')
        normalized = (features[:length].astype(np.float32) - mean) / std
        with torch.inference_mode():
            x = torch.as_tensor(normalized, device=self.device).unsqueeze(0)
            tokens = self.model.net.encode(x).cpu().numpy().reshape(-1)
        return tokens, self.decode(tokens), dict(input_frames=len(features), encoded_frames=length,
                                               dropped_tail_frames=len(features) - length, codec_id=self.codec_id)

    def call(self, task, caption, tokens, instruction, seed, budget=None):
        torch = self.torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        inputs = {'texts': [{'<Goal_Caption>': caption}]}
        if instruction is not None:
            inputs['texts'][0]['<Refinement_Instruction>'] = instruction
        if tokens is not None:
            motion = ''.join(f'<Motion_{int(t)}>' for t in tokens)
            inputs['motions'] = [{'<Generated_Motion>': motion}]
        ids, mask = self.processor.build_test_context(task, inputs, self.model.tokenizer)
        kwargs = dict(max_new_tokens=budget or self.config['max_new_tokens'],
                      do_sample=self.config['do_sample'], use_cache=True)
        if kwargs['do_sample']:
            kwargs.update(temperature=self.config['temperature'], top_p=self.config['top_p'])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.llm.generate(ids.to(self.device), attention_mask=mask.to(self.device), **kwargs)
        torch.cuda.synchronize()
        text = self.model.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
        record = dict(task=task, input_text=self.model.tokenizer.decode(ids[0]), generated_text=text,
                      seconds=time.perf_counter() - start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      seed=seed, decoding=kwargs, input_tokens=int(ids.shape[1]), output_tokens=int(output.shape[1] - ids.shape[1]))
        self.max_allocated_gib = max(self.max_allocated_gib, record['peak_allocated_gib'])
        # Upstream refinement contexts end with <Motion>; generation continues inside it.
        if task in ('instructed_motion_refinement', 't2m') and self.processor.gen_forcing:
            record['motion_open_tag_in_context'] = True
            text = '<Motion>' + text
        return text, record

    def semantic(self, caption, tokens, seed):
        text, record = self.call('refine_instruction', caption, tokens, None, seed)
        clean = text.strip().lower().rstrip('.')
        decision = 'no_defect' if clean == 'no refinements needed' else 'defect' if clean else 'uncertain'
        return dict(decision=decision, defects=[], instruction=text if decision == 'defect' else '',
                    scope='caption_alignment_only', raw_text=text), record

    def edit(self, caption, tokens, feedback, seed):
        text, record = self.call('instructed_motion_refinement', caption, tokens, feedback['instruction'], seed)
        try:
            indices = parse_motion(text, self.args.nb_code)
        except ValueError as exc:
            exc.generation_trace = record
            raise
        return indices, self.decode(indices), record

    def generate(self, caption, seed):
        text, record = self.call('unified_mogen_cot_v3', caption, None, None, seed, budget=1300)
        try:
            indices = parse_motion(text, self.args.nb_code, first=True)
        except ValueError as exc:
            exc.generation_trace = record
            raise
        return indices, self.decode(indices), record

    def session_record(self):
        self.torch.cuda.synchronize()
        seconds = time.perf_counter() - self.started
        return dict(identity=self.identity, load_seconds=self.load_seconds, elapsed_seconds=seconds,
                    used_gpu_count=1, allocated_process_gpu_hours=seconds / 3600,
                    peak_allocated_gib=self.max_allocated_gib,
                    note='One process uses cuda:0. Includes loading/CPU work while model resident; excludes idle notebook reservation outside this process.')
