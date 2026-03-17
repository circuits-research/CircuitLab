import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sqlite3
import torch

from sae_lens.load_model import load_model
from clt_forge.config import AutoInterpConfig
from clt_forge.clt import CLT
from clt_forge.training.activations_store import ActivationsStore
from clt_forge.training.optim import JumpReLU
from clt_forge.utils import DTYPE_MAP
from clt_forge import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# DDL shared by _save_to_sqlite and merge_job_databases
_FEATURES_TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS features (
        layer                 INTEGER NOT NULL,
        feature_id            INTEGER NOT NULL,
        average_activation    REAL    NOT NULL,
        top_examples          TEXT    NOT NULL,
        top_examples_tks      TEXT    NOT NULL,
        top_activating_tokens TEXT    NOT NULL,
        description           TEXT    NOT NULL,
        explanation           TEXT    NOT NULL,
        raw_explanation       TEXT    NOT NULL,
        PRIMARY KEY (layer, feature_id)
    )
"""

class AutoInterp:
    """
    Single-pass streaming AutoInterp.

    Usage:
        cfg = AutoInterpConfig(...)
        interp = AutoInterp(cfg)
        interp.run(job_id=0, total_jobs=32, top_k=100, save_dir=Path(...))
    """

    def __init__(self, cfg: AutoInterpConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.ctx = cfg.context_size

        # patch_official_model_names()
        # patch_convert_hf_model_config()

        self.model = load_model(
            cfg.model_class_name,
            cfg.model_name,
            device=self.device,
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
        )

        # CLT stays on CPU; only the feature subset is moved to GPU per job.
        self.clt = CLT.load_from_pretrained(cfg.clt_path, "cpu")

        self.activations_store = ActivationsStore(
            self.model,
            cfg,
            estimated_norm_scaling_factor_in=self.clt.estimated_norm_scaling_factor_in,
            estimated_norm_scaling_factor_out=self.clt.estimated_norm_scaling_factor_out,
        )
        # ActivationsStore defaults to return_tokens=False (yields 2-tuples).
        # We need the token ids alongside activations, so enable it here.
        self.activations_store.return_tokens = True

    def run(
        self,
        *,
        job_id: int,
        total_jobs: int,
        save_dir: Optional[Path] = None,
        generate_explanations: bool = False,
    ) -> None:
        """
        Run the full pipeline for one job (= one feature slice).

        Args:
            job_id:               Index of this job in [0, total_jobs).
            total_jobs:           Total number of parallel jobs.
            top_k:                Number of top-activating sequences per feature.
            save_dir:             Root output directory.
            generate_explanations: If True, run vLLM to generate explanations.
        """
        if save_dir is None:
            if self.cfg.latent_cache_path is None:
                raise ValueError(
                    "Either pass save_dir or set cfg.latent_cache_path."
                )
            save_dir = Path(self.cfg.latent_cache_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_list = self._compute_index_list(job_id, total_jobs)
        logger.info(
            f"[Job {job_id}/{total_jobs}] features {index_list[0]}–{index_list[-1]}"
            f" ({len(index_list)} total)"
        )

        run_dtype = DTYPE_MAP[self.cfg.dtype]
        self._prepare_encoder_subset(index_list=index_list, dtype=run_dtype)

        state = self._stream_and_build_topk(top_k=self.cfg.topk, run_dtype=run_dtype)

        logger.info(f"[Job {job_id}] Building feature dictionaries…")
        feature_dicts_by_layer = self._state_to_feature_dicts(
            state=state, index_list=index_list, top_k=self.cfg.topk
        )

        self._save_features(
            feature_dicts_by_layer=feature_dicts_by_layer,
            job_id=job_id,
            save_dir=save_dir,
        )

        if generate_explanations:
            logger.info(f"[Job {job_id}] Generating LLM explanations…")
            self._generate_and_add_explanations(
                feature_dicts_by_layer=feature_dicts_by_layer,
                job_id=job_id,
                save_dir=save_dir,
            )

        logger.info(f"[Job {job_id}] Done.")

    def _compute_index_list(self, job_id: int, total_jobs: int) -> List[int]:
        d_latent = self.clt.d_latent
        per_job = d_latent // total_jobs
        start = job_id * per_job
        end = start + per_job if job_id < total_jobs - 1 else d_latent
        return list(range(start, end))

    def _prepare_encoder_subset(
        self, *, index_list: List[int], dtype: torch.dtype
    ) -> None:
        idx = torch.as_tensor(index_list, dtype=torch.long)
        self._index_list = list(index_list)
        self._W_enc_sub = self.clt.W_enc[:, :, idx].to(
            self.device, dtype=dtype, non_blocking=True
        )  # [L, d_in, F]
        self._b_enc_sub = self.clt.b_enc[:, idx].to(
            self.device, dtype=dtype, non_blocking=True
        )  # [L, F]
        self._threshold_sub = torch.exp(self.clt.log_threshold[:, idx]).to(
            self.device, dtype=dtype, non_blocking=True
        )  # [L, F]
        self._bandwidth = self.clt.bandwidth

    @torch.no_grad()
    def _encode_subset(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_in] → [B, L, F]  (all on GPU)"""
        hidden_pre = (
            torch.einsum("bld,ldf->blf", x, self._W_enc_sub) + self._b_enc_sub
        )
        return JumpReLU.apply(hidden_pre, self._threshold_sub, self._bandwidth)

    def _stream_and_build_topk(
        self, *, top_k: int, run_dtype: torch.dtype
    ) -> Dict[str, Any]:
        """One pass over the data: encode and accumulate per-feature top-K."""
        state: Optional[Dict[str, Any]] = None
        n_tokens = 0
        iterator = iter(self.activations_store)
        log_every = max(1, self.cfg.total_autointerp_tokens // 20)

        while n_tokens < self.cfg.total_autointerp_tokens:
            tokens_cpu, acts_in, _ = next(iterator)
            x = acts_in.to(self.device, dtype=run_dtype)

            with torch.no_grad():
                feat_act = self._encode_subset(x)  # [B, L, F]

            tokens_seq, acts_seq = self._to_sequences(tokens_cpu, feat_act)
            n_tokens += int(feat_act.shape[0])

            if tokens_seq.shape[0] > 0:
                if state is None:
                    state = self._init_topk_state(
                        n_layers=int(acts_seq.shape[2]),
                        F=len(self._index_list),
                        top_k=top_k,
                        ctx=self.ctx,
                        dtype=run_dtype,
                        device=self.device,
                    )
                self._update_state(
                    state=state,
                    tokens_seq=tokens_seq,
                    acts_seq=acts_seq,
                    top_k=top_k,
                )

            del x, feat_act, acts_seq
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if n_tokens % log_every < self.cfg.train_batch_size_tokens:
                logger.info(
                    f"  {n_tokens:,} / {self.cfg.total_autointerp_tokens:,} tokens"
                )

        if state is None:
            raise RuntimeError("No data was processed — check ActivationsStore setup.")
        return state

    def _to_sequences(
        self, tokens_cpu: torch.Tensor, feat_act_gpu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape flat token/activation arrays into context-sized sequences.

        tokens_cpu:   [B]          (CPU)
        feat_act_gpu: [B, L, F]   (GPU)

        returns:
            tokens_seq: [B_seq, ctx]        (CPU)
            acts_seq:   [B_seq, ctx, L, F]  (GPU)
        """
        B = int(feat_act_gpu.shape[0])
        excess = B % self.ctx
        if excess:
            feat_act_gpu = feat_act_gpu[:-excess]
            tokens_cpu = tokens_cpu[:-excess]
            B -= excess

        if B == 0:
            L, F = feat_act_gpu.shape[1], feat_act_gpu.shape[2]
            return (
                tokens_cpu.new_zeros((0, self.ctx)),
                feat_act_gpu.new_zeros((0, self.ctx, L, F)),
            )

        B_seq = B // self.ctx
        tokens_seq = tokens_cpu.view(B_seq, self.ctx)
        acts_seq = feat_act_gpu.view(
            B_seq, self.ctx, feat_act_gpu.shape[1], feat_act_gpu.shape[2]
        )
        return tokens_seq, acts_seq

    @staticmethod
    def _init_topk_state(
        *,
        n_layers: int,
        F: int,
        top_k: int,
        ctx: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, Any]:
        neg_inf = torch.full((top_k, F), float("-inf"), device=device, dtype=dtype)
        tok_zero = torch.zeros((top_k, F, ctx), device=device, dtype=torch.int32)
        act_zero = torch.zeros((top_k, F, ctx), device=device, dtype=dtype)
        return {
            "top_vals":   [neg_inf.clone() for _ in range(n_layers)],
            "top_tokens": [tok_zero.clone() for _ in range(n_layers)],
            "top_acts":   [act_zero.clone() for _ in range(n_layers)],
            "sum_pos":    [torch.zeros(F, device=device, dtype=dtype) for _ in range(n_layers)],
            "count_pos":  [torch.zeros(F, device=device, dtype=torch.long) for _ in range(n_layers)],
        }

    @torch.no_grad()
    def _update_state(
        self,
        *,
        state: Dict[str, Any],
        tokens_seq: torch.Tensor,  # [B_seq, ctx]      CPU
        acts_seq: torch.Tensor,    # [B_seq, ctx, L, F] GPU
        top_k: int,
    ) -> None:
        B_seq = int(tokens_seq.shape[0])
        if B_seq == 0:
            return

        tokens_gpu = tokens_seq.to(self.device, dtype=torch.int32)
        L = int(acts_seq.shape[2])
        F = int(acts_seq.shape[3])
        ctx = int(acts_seq.shape[1])

        # Broadcast token sequence across all F features: [B_seq, F, ctx]
        batch_tokens = tokens_gpu[:, None, :].expand(B_seq, F, ctx).contiguous()

        for layer in range(L):
            # [B_seq, F, ctx]
            batch_acts = acts_seq[:, :, layer, :].permute(0, 2, 1).contiguous()
            # [B_seq, F] — max activation per sequence per feature
            batch_vals = batch_acts.max(dim=2).values

            # Running stats (for average activation computation)
            pos = batch_vals > 0
            state["sum_pos"][layer].add_((batch_vals * pos).sum(dim=0))
            state["count_pos"][layer].add_(pos.sum(dim=0))

            # Merge batch with current top-K, keep top-K
            merged_vals   = torch.cat([state["top_vals"][layer],   batch_vals  ], dim=0)  # [K+B, F]
            merged_tokens = torch.cat([state["top_tokens"][layer], batch_tokens], dim=0)  # [K+B, F, ctx]
            merged_acts   = torch.cat([state["top_acts"][layer],   batch_acts  ], dim=0)  # [K+B, F, ctx]

            new_vals, sel = torch.topk(merged_vals, k=top_k, dim=0)   # [K, F]
            gather_idx = sel[:, :, None].expand(top_k, F, ctx)        # [K, F, ctx]

            state["top_vals"][layer]   = new_vals
            state["top_tokens"][layer] = torch.gather(merged_tokens, 0, gather_idx)
            state["top_acts"][layer]   = torch.gather(merged_acts,   0, gather_idx)

        del tokens_gpu, batch_tokens

    def _state_to_feature_dicts(
        self,
        *,
        state: Dict[str, Any],
        index_list: List[int],
        top_k: int,
    ) -> List[Dict[str, Dict[str, Any]]]:
        """
        Returns a list of length n_layers.
        Each element is {str(feat_id): feature_dict} for that layer.
        String keys are used for JSON compatibility.
        """
        tokenizer = self.model.tokenizer
        n_layers = len(state["top_vals"])
        result: List[Dict[str, Dict[str, Any]]] = [{} for _ in range(n_layers)]

        for layer in range(n_layers):
            top_vals   = state["top_vals"][layer].cpu()    # [K, F]
            top_tokens = state["top_tokens"][layer].cpu()  # [K, F, ctx]
            top_acts   = state["top_acts"][layer].cpu()    # [K, F, ctx]
            sum_pos    = state["sum_pos"][layer].cpu()     # [F]
            count_pos  = state["count_pos"][layer].cpu()   # [F]

            for j, feat_id in enumerate(index_list):
                # Collect non-trivial top sequences (descending order from topk)
                sequences: List[Dict[str, Any]] = []
                for k in range(top_k):
                    v = float(top_vals[k, j])
                    if v == float("-inf") or v <= 0:
                        break  # remaining slots are unfilled or zero
                    sequences.append(
                        {
                            "tokens":      top_tokens[k, j],
                            "activations": top_acts[k, j],
                            "max_val":     v,
                        }
                    )

                top_examples: List[str] = []
                sequences_serializable: List[Dict[str, Any]] = []
                for s in sequences:
                    tks  = s["tokens"][1:]       # drop BOS token
                    acts = s["activations"][1:]  # drop BOS
                    top_examples.append(highlight_activations(tks, acts, tokenizer))
                    sequences_serializable.append(
                        {
                            "tokens":      s["tokens"].tolist(),
                            "activations": s["activations"].tolist(),
                            "max_val":     float(s["max_val"]),
                        }
                    )

                c = int(count_pos[j])
                avg_activation = float(sum_pos[j]) / c if c > 0 else 0.0

                result[layer][str(feat_id)] = {
                    "layer":                 int(layer),
                    "feature_index":         int(feat_id),
                    "average_activation":    avg_activation,
                    "top_examples":          top_examples,
                    "top_examples_tks":      sequences_serializable,
                    "top_activating_tokens": _get_top_activating_tokens(
                        sequences=sequences,
                        tokenizer=tokenizer,
                        top_k=self.cfg.n_top_activating_tokens_shown,
                    ),
                    "description":           "Unknown",
                    "explanation":           "No explanation generated",
                    "raw_explanation":       "",
                }

        return result
    
    def _save_features(
        self,
        *,
        feature_dicts_by_layer: List[Dict[str, Dict[str, Any]]],
        job_id: int,
        save_dir: Path,
    ) -> Path:
        backend = getattr(self.cfg, "storage_backend", "parquet")

        if backend == "sqlite":
            return self._save_to_sqlite(
                feature_dicts_by_layer=feature_dicts_by_layer,
                job_id=job_id,
                save_dir=save_dir,
            )
        if backend == "lmdb":
            return self._save_to_lmdb(
                feature_dicts_by_layer=feature_dicts_by_layer,
                job_id=job_id,
                save_dir=save_dir,
            )
        if backend == "parquet":
            return self._save_to_parquet(
                feature_dicts_by_layer=feature_dicts_by_layer,
                job_id=job_id,
                save_dir=save_dir,
            )

        raise ValueError(f"Unknown storage_backend: {backend}")

    def _save_to_parquet(
        self,
        *,
        feature_dicts_by_layer: List[Dict[str, Dict[str, Any]]],
        job_id: int,
        save_dir: Path,
    ) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
        import json

        out_dir = save_dir / "parquet"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"job_{job_id}.parquet"

        rows = []
        for layer_dicts in feature_dicts_by_layer:
            for d in layer_dicts.values():
                rows.append(
                    {
                        "layer": int(d["layer"]),
                        "feature_id": int(d["feature_index"]),
                        "average_activation": float(d["average_activation"]),
                        # store nested stuff as json text (easy + portable)
                        "top_examples": json.dumps(d["top_examples"]),
                        "top_examples_tks": json.dumps(d["top_examples_tks"]),
                        "top_activating_tokens": json.dumps(d["top_activating_tokens"]),
                        "description": d["description"],
                        "explanation": d["explanation"],
                        "raw_explanation": d["raw_explanation"],
                    }
                )

        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, compression="zstd")

        logger.info(f"[Job {job_id}] Saved {len(rows)} features → {path}")
        return path

    def _save_to_lmdb(
        self,
        *,
        feature_dicts_by_layer: List[Dict[str, Dict[str, Any]]],
        job_id: int,
        save_dir: Path,
    ) -> Path:
        import lmdb
        import pickle

        db_path = save_dir / f"job_{job_id}.lmdb"

        # Map size must be large enough (e.g. 10GB here — adjust if needed)
        env = lmdb.open(
            str(db_path),
            map_size=10 * 1024**3,
            subdir=False,
            readonly=False,
            meminit=False,
            map_async=True,
        )

        with env.begin(write=True) as txn:
            count = 0
            for layer_dicts in feature_dicts_by_layer:
                for d in layer_dicts.values():
                    key = f"{d['layer']}:{d['feature_index']}".encode()
                    value = pickle.dumps(d)
                    txn.put(key, value)
                    count += 1

        env.sync()
        env.close()

        logger.info(f"[Job {job_id}] Saved {count} features → {db_path}")
        return db_path

    def _save_to_sqlite(
        self,
        *,
        feature_dicts_by_layer: List[Dict[str, Dict[str, Any]]],
        job_id: int,
        save_dir: Path,
    ) -> Path:
        """
        Persist all features for this job in a single SQLite database.

        Output: save_dir/job_{job_id}.db  — one file per job regardless of
        how many layers or features it contains.

        Query specific features later with load_features(db_path, layer, [ids]).
        Merge all jobs with merge_job_databases([...], output_db_path).
        """

        db_path = save_dir / f"job_{job_id}.db"
        con = sqlite3.connect(db_path)
        con.execute(_FEATURES_TABLE_DDL)

        rows = [
            (
                d["layer"],
                d["feature_index"],
                d["average_activation"],
                json.dumps(d["top_examples"]),
                json.dumps(d["top_examples_tks"]),
                json.dumps(d["top_activating_tokens"]),
                d["description"],
                d["explanation"],
                d["raw_explanation"],
            )
            for layer_dicts in feature_dicts_by_layer
            for d in layer_dicts.values()
        ]

        con.executemany(
            "INSERT OR REPLACE INTO features VALUES (?,?,?,?,?,?,?,?,?)", rows
        )
        con.commit()
        con.close()

        logger.info(f"[Job {job_id}] Saved {len(rows)} features → {db_path}")
        return db_path

    def _generate_and_add_explanations(
        self,
        *,
        feature_dicts_by_layer: List[Dict[str, Dict[str, Any]]],
        job_id: int,
        save_dir: Path,
    ) -> None:
        """
        Build prompts in-memory, run vLLM, parse responses, patch the
        in-memory dicts, and re-save using the configured storage backend.
        """
        from clt_forge.autointerp.client import run_client # only import VLLM if needed
        from clt_forge.autointerp.prompt import generate_prompt

        prompt_texts: List[str] = []
        feat_layer_keys: List[Tuple[int, str]] = []

        for layer, layer_dicts in enumerate(feature_dicts_by_layer):
            for feat_key, feat_dict in layer_dicts.items():
                if not feat_dict["top_examples"]:
                    continue  # dead feature — skip
                prompt_texts.append(
                    generate_prompt(feat_dict["top_examples"], layer, int(feat_key))
                )
                feat_layer_keys.append((layer, feat_key))

        if not prompt_texts:
            logger.info(f"[Job {job_id}] No live features — skipping vLLM.")
            return

        explanations = run_client(
            prompts=prompt_texts,
            vllm_model=self.cfg.vllm_model,
            vllm_max_tokens=self.cfg.vllm_max_tokens,
        )

        for raw, (layer, feat_key) in zip(explanations, feat_layer_keys):
            desc, expl = _parse_explanation(raw)
            d = feature_dicts_by_layer[layer][feat_key]
            d["raw_explanation"] = raw
            d["description"]     = desc
            d["explanation"]     = expl

        # Re-save to SQLite with explanations filled in
        self._save_features(
            feature_dicts_by_layer=feature_dicts_by_layer,
            job_id=job_id,
            save_dir=save_dir,
        )

def highlight_activations(
    tokens: torch.Tensor,
    activations: torch.Tensor,
    tokenizer,
    threshold_ratio: float = 0.6,
) -> str:
    """
    Build a text string with <<highlighted>> spans around the most active tokens.
    Handles Chinese text (no word-boundary extension) vs. Latin scripts.
    """
    assert len(tokens) == len(activations), "Token/activation length mismatch."
    if activations.numel() == 0:
        return ""

    max_act = float(activations.max())
    if max_act <= 0:
        return tokenizer.decode(tokens.tolist())

    threshold = max_act * threshold_ratio
    str_tokens = tokenizer.convert_ids_to_tokens(tokens.tolist())
    highlight_mask = (activations > threshold).tolist()

    def _contains_chinese(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    sample = tokenizer.convert_tokens_to_string(str_tokens[: min(10, len(str_tokens))])
    is_chinese = _contains_chinese(sample)

    extended = list(highlight_mask)
    if not is_chinese:
        WORD_STARTS = {"▁", " ", "Ġ"}
        SPECIAL = {"<|endoftext|>", "</s>", "<s>", "[CLS]", "[SEP]"}
        for i in range(len(str_tokens)):
            if not highlight_mask[i]:
                continue
            # Extend backwards to word start
            j = i - 1
            while (
                j >= 0
                and str_tokens[j][:1] not in WORD_STARTS
                and str_tokens[j] not in SPECIAL
            ):
                extended[j] = True
                j -= 1
            # Extend forwards to word end
            j = i + 1
            while (
                j < len(str_tokens)
                and str_tokens[j][:1] not in WORD_STARTS
                and str_tokens[j] not in SPECIAL
            ):
                extended[j] = True
                j += 1

    # Build marked token list
    marked: List[str] = []
    in_hl = False
    for tok, hi in zip(str_tokens, extended):
        if hi and not in_hl:
            marked.append("<<")
            in_hl = True
        elif not hi and in_hl:
            marked.append(">>")
            in_hl = False
        marked.append(tok)
    if in_hl:
        marked.append(">>")

    # Merge token subwords back into strings
    segments: List[str] = []
    buf: List[str] = []
    for tok in marked:
        if tok in {"<<", ">>"}:
            if buf:
                segments.append(tokenizer.convert_tokens_to_string(buf))
                buf = []
            segments.append(tok)
        else:
            buf.append(tok)
    if buf:
        segments.append(tokenizer.convert_tokens_to_string(buf))

    # Assemble final string
    result = ""
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg == "<<":
            if result and not result.endswith(" "):
                result += " "
            result += "<<"
            i += 1
            if i < len(segments):
                result += segments[i].lstrip()
                i += 1
            if i < len(segments) and segments[i] == ">>":
                result += ">>"
                i += 1
        else:
            if result and not result.endswith(" "):
                result += " "
            result += seg
            i += 1

    return result

def _get_top_activating_tokens(
    *,
    sequences: List[Dict[str, Any]],
    tokenizer,
    top_k: int = 4,
    threshold_ratio: float = 0.6,
) -> List[Dict[str, Any]]:
    """Return the most frequently highly-active tokens across top sequences."""
    if not sequences:
        return []

    all_tks  = torch.cat([s["tokens"][1:]      for s in sequences])
    all_acts = torch.cat([s["activations"][1:] for s in sequences])

    if all_acts.numel() == 0:
        return []

    thresh = float(all_acts.max()) * threshold_ratio
    mask = all_acts > thresh

    stats: Dict[int, Dict[str, float]] = {}
    for tid, act in zip(all_tks[mask].tolist(), all_acts[mask].tolist()):
        if tid not in stats:
            stats[tid] = {"count": 0.0, "total": 0.0}
        stats[tid]["count"] += 1.0
        stats[tid]["total"] += float(act)

    ranking = [
        {
            "token":              tokenizer.decode([int(tid)]),
            "token_id":           int(tid),
            "frequency":          int(v["count"]),
            "average_activation": v["total"] / v["count"],
        }
        for tid, v in stats.items()
    ]
    ranking.sort(key=lambda x: x["frequency"], reverse=True)
    return ranking[:top_k]


def _parse_explanation(raw: str) -> Tuple[str, str]:
    """Parse the [DESCRIPTION]: / [EXPLANATION]: response format."""
    description = "Unknown"
    explanation = raw
    for line in raw.splitlines():
        if line.startswith("[DESCRIPTION]:"):
            description = line[len("[DESCRIPTION]:"):].strip()
        elif line.startswith("[EXPLANATION]:"):
            explanation = line[len("[EXPLANATION]:"):].strip()
    return description, explanation

### Public utilities for loading and merging results

def load_features_parquet(
    root: Path,
    layer: int,
    feature_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    import pyarrow.dataset as ds
    import json

    root = Path(root)

    # Accept:
    # 1) root is a parquet file
    # 2) root is a directory containing job_*.parquet
    # 3) root is a directory that contains a "parquet/" subdir (legacy layout)
    if root.is_file():
        dataset_path = root
    else:
        parquet_subdir = root / "parquet"
        dataset_path = parquet_subdir if parquet_subdir.is_dir() else root

    dataset = ds.dataset(str(dataset_path), format="parquet")

    filt = (ds.field("layer") == layer)
    if feature_ids is not None:
        filt = filt & ds.field("feature_id").isin(feature_ids)

    table = dataset.to_table(filter=filt)
    rows = table.to_pylist()

    for r in rows:
        r["top_examples"] = json.loads(r["top_examples"])
        r["top_examples_tks"] = json.loads(r["top_examples_tks"])
        r["top_activating_tokens"] = json.loads(r["top_activating_tokens"])

    return rows

def load_features_lmdb(
    db_path: Path,
    layer: int,
    feature_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    import lmdb
    import pickle

    env = lmdb.open(str(db_path), readonly=True, lock=False, subdir=False)

    results = []

    with env.begin() as txn:
        cursor = txn.cursor()

        if feature_ids is None:
            prefix = f"{layer}:".encode()
            for key, value in cursor:
                if key.startswith(prefix):
                    results.append(pickle.loads(value))
        else:
            for fid in feature_ids:
                key = f"{layer}:{fid}".encode()
                value = txn.get(key)
                if value is not None:
                    results.append(pickle.loads(value))

    env.close()
    return results

def load_features_sqlite(
    db_path: Path,
    layer: int,
    feature_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Load feature dicts from a job SQLite database.

    Args:
        db_path:     Path to a job_{j}.db file (or merged features.db).
        layer:       Which CLT layer to query.
        feature_ids: Specific feature IDs to load.  None → all for this layer.

    Returns:
        List of feature dicts with JSON fields already deserialized.

    Example:
        features = load_features(Path("save_dir/job_0.db"), layer=3)
    """
    import sqlite3

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    if feature_ids is None:
        cursor = con.execute(
            "SELECT * FROM features WHERE layer = ?", (layer,)
        )
    else:
        placeholders = ",".join("?" * len(feature_ids))
        cursor = con.execute(
            f"SELECT * FROM features WHERE layer = ? AND feature_id IN ({placeholders})",
            (layer, *feature_ids),
        )

    rows = cursor.fetchall()
    con.close()

    result = []
    for row in rows:
        d = dict(row)
        d["top_examples"]          = json.loads(d["top_examples"])
        d["top_examples_tks"]      = json.loads(d["top_examples_tks"])
        d["top_activating_tokens"] = json.loads(d["top_activating_tokens"])
        result.append(d)
    return result

def merge_job_lmdbs(
    job_db_paths: List[Path],
    output_db_path: Path,
) -> None:
    import lmdb

    # Large map_size to avoid resize issues
    env_out = lmdb.open(
        str(output_db_path),
        map_size=50 * 1024**3,  # adjust if needed
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    total = 0

    with env_out.begin(write=True) as txn_out:
        for db_path in job_db_paths:
            env_in = lmdb.open(
                str(db_path),
                readonly=True,
                lock=False,
                subdir=False,
            )

            with env_in.begin() as txn_in:
                cursor = txn_in.cursor()
                for key, value in cursor:
                    txn_out.put(key, value)
                    total += 1

            env_in.close()

    env_out.sync()
    env_out.close()

    logger.info(f"Merged {len(job_db_paths)} LMDBs → {output_db_path}")
    logger.info(f"Total features copied: {total}")

def merge_job_databases(job_db_paths: List[Path], output_db_path: Path) -> None:
    """
    Merge all per-job SQLite databases into one file.

    Typical use after all jobs have finished:
        merge_job_databases(
            sorted(save_dir.glob("job_*.db")),
            save_dir / "features.db",
        )

    The resulting features.db supports the same load_features() queries
    across the full feature space.
    """
    import sqlite3

    con = sqlite3.connect(output_db_path)
    con.execute(_FEATURES_TABLE_DDL)
    con.commit()

    for db_path in job_db_paths:
        con.execute("ATTACH DATABASE ? AS src", (str(db_path),))
        con.execute("INSERT OR REPLACE INTO features SELECT * FROM src.features")
        con.execute("DETACH DATABASE src")
        con.commit()

    con.close()
    logger.info(f"Merged {len(job_db_paths)} databases → {output_db_path}")
