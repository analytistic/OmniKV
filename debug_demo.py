"""
EchoKV / ArachneKV 统一调试入口

两阶段：先复述 (recitation)，再问答 (query)
支持通过 --config 或 config 中的 model_cls 切换策略。
"""

import sys, torch
from transformers import AutoTokenizer, set_seed, TextStreamer
from modeling.omnikv.omnikv_config import LlamaCompressorConfig
from tiny_tools.read_json import read_config

sys.modules['baselines'] = type(sys)('baselines')
sys.modules['baselines.infllm'] = type(sys)('baselines.infllm')
sys.modules['baselines.raw_h2o'] = type(sys)('baselines.raw_h2o')


VERANDIA_PASSAGE = (
    "The small island nation of Verandia, located in the southern Pacific Ocean, "
    "declared independence from colonial rule on March 3, 1987. Its first president, "
    "Elena Korvath, served two consecutive terms from 1987 to 1999. Under her "
    "leadership, Verandia transitioned from a plantation economy dependent on "
    "sugar cane exports to a diversified economy centered on eco-tourism, "
    "sustainable fisheries, and a growing technology sector.\n\n"
    "In 2003, Verandia established the Korvath Marine Reserve, named after the "
    "former president, which spans 12,000 square kilometers around the island's "
    "coral reef system. The reserve became a UNESCO World Heritage Site in 2008 "
    "and attracts roughly 40,000 visitors per year. Revenue from the reserve's "
    "entrance fees funds the majority of the island's public school system.\n\n"
    "Verandia's current population stands at approximately 58,000 people. The "
    "official languages are English and Verandian Creole. The capital, Port Alani, "
    "is home to the National University of Verandia, which opened in 1995 and "
    "is known for its marine biology program. The university collaborates with "
    "research institutions worldwide and operates a deep-sea research vessel, "
    "the RV Coral Pioneer, launched in 2019."
)

RECITATION_PROMPT = VERANDIA_PASSAGE + (
    "\nPlease repeat the context above verbatim, word for word, "
    "without any additions or omissions."
)

QA_PROMPT_Q1 = VERANDIA_PASSAGE + (
    "\nBased on the passage above, what year did Verandia declare independence? "
    "Answer briefly."
)

QA_PROMPT_Q2 = VERANDIA_PASSAGE + (
    "\nBased on the passage above, who was Verandia's first president? "
    "Answer briefly."
)

GROUND_TRUTH = {
    "Q1": "1987 (March 3, 1987)",
    "Q2": "Elena Korvath",
}


def get_model_class(model_cls: str):
    """根据 model_cls 返回 (LM_class, cache_class)"""
    from modeling.arachnekv import ArachneLM
    from modeling.arachnekv.cache import ArachneCache
    from modeling.echokv import EchoLM
    from modeling.echokv.cache import EchoCache

    mapping = {
        "arachne": (ArachneLM, ArachneCache),
        "echo": (EchoLM, EchoCache),
    }
    for key, (lm_cls, cache_cls) in mapping.items():
        if key in model_cls.lower():
            return lm_cls, cache_cls
    # 默认 Arachne
    return ArachneLM, ArachneCache


def print_cache_report(model, phase_name: str):
    """打印压缩报告（支持 ArachneCache 和 EchoCache）"""
    from modeling.arachnekv.cache import ArachneCache
    from modeling.echokv.cache import EchoCache

    cache = getattr(model, '_last_cache', None)
    if cache is None:
        return

    hd = model.config.hidden_size // model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    num_layers = cache.num_hidden_layers
    num_sparse = sum(1 for l in range(num_layers) if cache.layer_state[l][0])
    num_full = num_layers - num_sparse

    # ── seq 长度 ──
    if isinstance(cache, ArachneCache):
        seq = None
        for l in range(num_layers):
            if cache.layer_state[l][0] and l in cache.k_residual:
                seq = cache.k_residual[l].shape[1]
                break
        if seq is None:
            return
    else:
        # EchoCache: 从第一个 full 层的 KV 取 seq
        for l in range(num_layers):
            if not cache.layer_state[l][0] and l < len(cache.key_cache):
                seq = cache.key_cache[l].shape[-2]
                break
        else:
            seq = cache._seen_tokens

    # ── Compute CR ──
    comp_orig_total = num_layers * seq * nkv * 2

    if isinstance(cache, ArachneCache):
        num_sel = cache.selected_idx.numel() if cache.selected_idx is not None else seq
    else:
        num_sel = cache.selected_idx.numel() if (hasattr(cache, 'selected_idx') and cache.selected_idx is not None) else seq

    comp_comp = (num_full * seq + num_sparse * num_sel) * nkv * 2
    cr = comp_orig_total / comp_comp if comp_comp > 0 else 1.0

    # ── Storage CR ──
    storage_orig = num_layers * seq * nkv * 2 * hd

    if isinstance(cache, ArachneCache):
        nck = seq // model.config.k_rq_cenyroid
        ncv = seq // model.config.v_rq_cenyroid
        sparse_k = nck * model.config.k_rq_levels + seq * model.config.k_residual_keep_ratio
        sparse_v = ncv * model.config.v_rq_levels + seq * model.config.v_residual_keep_ratio
        storage_comp = (num_full * seq + num_sparse * sparse_k) * nkv * hd \
                     + (num_full * seq + num_sparse * sparse_v) * nkv * hd
    else:
        # EchoKV: 按 echo_mode + store_every 算存储
        echo_mode = getattr(cache, 'echo_mode', 'kv')
        store_every = getattr(cache, 'store_every', 0)

        if store_every and hasattr(cache, 'kv_source'):
            num_echo_store = sum(1 for l in range(num_layers)
                                 if cache.layer_state[l][0]
                                 and cache.kv_source.get(l, l) == l)
            num_echo_other = num_sparse - num_echo_store
        else:
            num_echo_store = 0
            num_echo_other = num_sparse

        storage_full = (num_full + num_echo_store) * seq * nkv * 2 * hd  # 存 K+V

        if num_echo_other == 0:
            storage_comp = storage_full
        elif echo_mode is None:
            storage_comp = storage_full + num_echo_other * seq * nkv * 2 * hd
        elif echo_mode == "kv":
            storage_comp = storage_full
        else:
            # "k" 或 "v": 存一份
            storage_comp = storage_full + num_echo_other * seq * nkv * hd

    sr = storage_orig / storage_comp if storage_comp > 0 else 1.0

    # ── 打印 ──
    if isinstance(cache, ArachneCache):
        variant = "ArachneKV"
    else:
        em = getattr(cache, 'echo_mode', 'kv')
        variant = "EchoKV(None)" if em is None or em == "" else f"EchoKV({em})"
        se = getattr(cache, 'store_every', 0)
        if se:
            variant += f"_s{se}"
    print(f"\n{'='*60}")
    print(f"【{phase_name}】{variant} 压缩报告:")
    print(f"  - pref={seq}, layers={num_layers} (full={num_full}, sparse={num_sparse})")
    print(f"  - 原始计算: {comp_orig_total/1e6:.2f}M, 压缩计算: {comp_comp/1e6:.2f}M, CR={cr:.2f}x ({100/cr:.1f}%)")
    print(f"  - 原始存储: {storage_orig/1e6:.2f}M, 压缩存储: {storage_comp/1e6:.2f}M, SR={sr:.2f}x ({100/sr:.1f}%)")

    if isinstance(cache, ArachneCache):
        nck = seq // model.config.k_rq_cenyroid
        ncv = seq // model.config.v_rq_cenyroid
        print(f"  - 稀疏层: K中心={nck*model.config.k_rq_levels}, V中心={ncv*model.config.v_rq_levels}, "
              f"残差={int(seq*model.config.k_residual_keep_ratio)}")
    print(f"{'='*60}")


def run_generation(model, tkn, prompt: str, max_new_tokens: int, phase_name: str) -> None:
    """统一生成流程"""
    input_ids = tkn.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors='pt',
    )
    print(f"\n【{phase_name}】输入长度: {input_ids.shape[1]} tokens")

    with torch.no_grad():
        streamer = TextStreamer(tkn, skip_prompt=False)
        model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=50,
            streamer=streamer,
        )

    print_cache_report(model, phase_name)


def main():
    set_seed(42)

    cfg_path = "configs/demo_arachne.json"
    config = read_config(cfg_path)
    model_name = config['model_name']
    model_cls = config.get('model_cls', 'arachne')

    LMClass, _ = get_model_class(model_cls)

    hf_cfg = LlamaCompressorConfig.from_pretrained(model_name, local_files_only=True)
    hf_cfg.set_config_of_compressor(**config)

    model = LMClass.from_pretrained(
        model_name, config=hf_cfg, torch_dtype=torch.float32, local_files_only=True,
    )
    model.eval()
    print(f"模型加载完成: {model_name} ({model_cls})")
    print(f"参数: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    tkn = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tkn.pad_token is None:
        tkn.add_special_tokens({'pad_token': '[PAD]'})

    run_generation(model, tkn, RECITATION_PROMPT, max_new_tokens=1000, phase_name="Phase 1: 复述")
    run_generation(model, tkn, QA_PROMPT_Q1, max_new_tokens=64, phase_name="Phase 2: Q1 独立年份")
    print(f"  >>> 正确答案: {GROUND_TRUTH['Q1']}")
    run_generation(model, tkn, QA_PROMPT_Q2, max_new_tokens=64, phase_name="Phase 3: Q2 首任总统")
    print(f"  >>> 正确答案: {GROUND_TRUTH['Q2']}")


if __name__ == '__main__':
    main()
