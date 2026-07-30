# """
# Modular leave-one-out evaluation for feedback-driven active learning retrieval.

# This script evaluates baseline FAISS retrieval against feedback-lift reranking
# with strict per-query isolation:
#   - all feedback rows involving the query ticket are removed before lift fitting
#   - the query ticket is removed from retrieval candidates before generation
#   - baseline and AL use the same classifier outputs, generator, and metrics

# It is designed for:
#   - comprehensive_feedback_250x250.db
#   - comprehensive_feedback_v3.db

# Examples:
#   python run_modular_looe_active_learning.py --feedback-db comprehensive_feedback_250x250.db --limit 10
#   python run_modular_looe_active_learning.py --feedback-db comprehensive_feedback_v3.db --lift tanh --gating fuzzy --llm-model openai/gpt-4o-mini
# """
# from __future__ import annotations

# import argparse
# import asyncio
# import hashlib
# import json
# import math
# import os
# import platform
# import sqlite3
# import sys
# import time
# from dataclasses import asdict, dataclass
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Callable, Optional

# import numpy as np
# import pandas as pd

# for _stream in (sys.stdout, sys.stderr):
#     try:
#         _stream.reconfigure(encoding="utf-8", errors="replace")
#     except Exception:
#         pass

# try:
#     from dotenv import load_dotenv

#     load_dotenv(override=True)
# except Exception:
#     pass

# rt = None
# _async_generate_response_with_openai = None
# async_judge_items = None


# def _json_safe(obj: Any) -> Any:
#     if isinstance(obj, dict):
#         return {str(k): _json_safe(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [_json_safe(v) for v in obj]
#     if isinstance(obj, tuple):
#         return [_json_safe(v) for v in obj]
#     if isinstance(obj, (np.integer,)):
#         return int(obj)
#     if isinstance(obj, (np.floating,)):
#         value = float(obj)
#         return value if math.isfinite(value) else None
#     if isinstance(obj, float):
#         return obj if math.isfinite(obj) else None
#     if pd.isna(obj) if not isinstance(obj, (list, dict, tuple)) else False:
#         return None
#     return obj


# def _sha256_file(path: str | Path) -> Optional[str]:
#     try:
#         h = hashlib.sha256()
#         with open(path, "rb") as f:
#             for chunk in iter(lambda: f.read(1024 * 1024), b""):
#                 h.update(chunk)
#         return h.hexdigest()
#     except Exception:
#         return None


# def _file_fingerprint(path: str | Path) -> dict:
#     p = Path(path)
#     if not p.exists():
#         return {"path": str(path), "exists": False}
#     return {
#         "path": str(path),
#         "exists": True,
#         "size_bytes": p.stat().st_size,
#         "mtime": p.stat().st_mtime,
#         "sha256": _sha256_file(p),
#     }


# def _mean(values: list[Optional[float]]) -> Optional[float]:
#     clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
#     return float(np.mean(clean)) if clean else None


# def _std(values: list[Optional[float]]) -> Optional[float]:
#     clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
#     return float(np.std(clean)) if clean else None


# def _norm_id(value: Any) -> str:
#     return str(value).strip()


# def _clean_team(team: Any) -> str:
#     return str(team or "").replace(",", "").replace("|", "").strip()


# def _ticket_text(row: pd.Series) -> str:
#     return f"{row.get('Title_anon', '')} {row.get('Description_anon', '')}".strip()


# def _quality_flags(generated: str, expected_team: str, predicted_team: str) -> dict:
#     text = str(generated or "")
#     return {
#         "length": len(text),
#         "has_greeting": any(g in text.lower() for g in ["dear", "hello", "thank you", "hi"]),
#         "has_structure": text.count("\n-") >= 2 or text.count(":") >= 3,
#         "team_match": str(expected_team) == str(predicted_team),
#     }


# @dataclass(frozen=True)
# class LiftConfig:
#     name: str
#     alpha: float = 1.0
#     beta: float = 1.0
#     multiplier: float = 0.60
#     cap: float = 0.20
#     sensitivity: float = 5.0
#     lcb_k: float = 1.0
#     positive_only: bool = False


# @dataclass(frozen=True)
# class RoutingConfig:
#     name: str
#     w_global: float = 0.10
#     w_class: float = 0.55
#     w_team: float = 0.35    
#     relevance_threshold: float = 0.70
#     require_semantic: bool = False


# @dataclass(frozen=True)
# class GatingConfig:
#     name: str
#     faiss_ceiling: float = 0.656
#     fuzzy_low: float = 0.50
#     fuzzy_high: float = 0.75
#     miracle_tau: float = 0.005


# @dataclass(frozen=True)
# class EvalConfig:
#     tickets_db: str
#     feedback_db: str
#     results_dir: str
#     top_k: int
#     search_k: int
#     sentence_model: str
#     llm_model: str
#     judge_model: Optional[str]
#     lift: LiftConfig
#     routing: RoutingConfig
#     gating: GatingConfig
#     calculate_bert: bool
#     judge_generated: bool
#     retrieval_only: bool
#     limit: Optional[int]
#     query_offset: int


# class LiftRegistry:
#     @staticmethod
#     def laplace(pos: float, neg: float, scores: pd.Series, sims: pd.Series, cfg: LiftConfig) -> float:
#         weighted_neg = 0.0 if cfg.positive_only else neg
#         p = (pos + cfg.alpha) / (pos + weighted_neg + cfg.alpha + cfg.beta)
#         n = pos if cfg.positive_only else pos + neg
#         scale = min(1.0, n / 2.0)
#         return (p - 0.5) * scale * cfg.multiplier

#     @staticmethod
#     def tanh(pos: float, neg: float, scores: pd.Series, sims: pd.Series, cfg: LiftConfig) -> float:
#         sim_values = sims.fillna(0.5).replace(0, 0.5).astype(float)
#         evidence = ((scores.astype(float) - 0.5) * sim_values).sum()
#         return math.tanh(float(evidence) / cfg.sensitivity) * cfg.cap

#     @staticmethod
#     def bayesian_lcb(pos: float, neg: float, scores: pd.Series, sims: pd.Series, cfg: LiftConfig) -> float:
#         vals = scores.astype(float)
#         if vals.empty:
#             return 0.0
#         mu = float(vals.mean()) - 0.5
#         sigma = float(vals.std(ddof=0)) if len(vals) > 1 else 0.25
#         uncertainty = cfg.lcb_k * sigma / math.sqrt(max(1, len(vals)))
#         return (mu - uncertainty) * cfg.multiplier

#     @classmethod
#     def get(cls, name: str) -> Callable[[float, float, pd.Series, pd.Series, LiftConfig], float]:
#         mapping = {
#             "laplace": cls.laplace,
#             "tanh": cls.tanh,
#             "bayesian_lcb": cls.bayesian_lcb,
#         }
#         if name not in mapping:
#             raise ValueError(f"Unknown lift formula '{name}'. Choose one of {sorted(mapping)}")
#         return mapping[name]


# class ModularLOOActiveLearning:
#     def __init__(self, cfg: EvalConfig):
#         global rt, _async_generate_response_with_openai, async_judge_items
#         if rt is None:
#             import resolution_task as _rt
#             from resolution_task_async import (
#                 _async_generate_response_with_openai as _async_generate,
#                 async_judge_items as _async_judge_items,
#             )

#             rt = _rt
#             _async_generate_response_with_openai = _async_generate
#             async_judge_items = _async_judge_items

#         self.cfg = cfg
#         self.results_dir = Path(cfg.results_dir)
#         self.results_dir.mkdir(parents=True, exist_ok=True)

#         if cfg.llm_model:
#             os.environ["LLM_MODEL"] = cfg.llm_model
#             rt.LLM_MODEL = cfg.llm_model

#             rt.GEN_MODEL_TEMPLATE = cfg.llm_model
#             rt.GEN_MODEL_PERSONAL = cfg.llm_model
#             rt.GEN_MODEL_SHORT = cfg.llm_model
#             rt.GEN_MODEL_STATUS = cfg.llm_model
#         if cfg.judge_model:
#             os.environ["JUDGE_MODEL"] = cfg.judge_model
#         os.environ.setdefault("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

#         self.kb = self._load_kb(cfg.tickets_db)
#         self.feedback = self._load_feedback(cfg.feedback_db)

#         # Extract unique ticket IDs referenced in the feedback database
#         feedback_ids = set(self.feedback["query_ticket_id"].unique()).union(
#             set(self.feedback["feedback_ticket_id"].unique())
#         )
#         self.unique_feedback_ids = {str(tid).strip() for tid in feedback_ids if tid}

#         self.rag = rt.RAGSystem(self.kb, sentence_model_name=cfg.sentence_model, kb_path=cfg.tickets_db)
#         self.rag.build_index()
#         self.id_to_positions = self._build_id_index()
#         self.query_ids = self._query_ids()

#         self.rouge = None
#         try:
#             from rouge_score import rouge_scorer

#             self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
#         except Exception:
#             pass

#     def _load_kb(self, tickets_db: str) -> pd.DataFrame:
#         con = sqlite3.connect(tickets_db)
#         try:
#             df = pd.read_sql_query("SELECT * FROM tickets", con)
#         finally:
#             con.close()
#         df = df.rename(
#             columns={
#                 "ticket_id": "Ref",
#                 "title": "Title_anon",
#                 "description": "Description_anon",
#                 "service_subcategory": "label_auto",
#                 "team": "Team",
#             }
#         )
#         for col in ["Ref", "sequential_id", "Title_anon", "Description_anon", "first_reply", "label_auto", "Team"]:
#             if col not in df.columns:
#                 df[col] = ""
#             df[col] = df[col].fillna("").astype(str)
#         return df.reset_index(drop=True)

#     def _load_feedback(self, feedback_db: str) -> pd.DataFrame:
#         con = sqlite3.connect(feedback_db)
#         try:
#             df = pd.read_sql_query("SELECT * FROM feedback", con)
#         finally:
#             con.close()
#         required = [
#             "query_ticket_id",
#             "query_class",
#             "query_team",
#             "feedback_ticket_id",
#             "feedback_class",
#             "feedback_team",
#             "score",
#         ]
#         for col in required:
#             if col not in df.columns:
#                 df[col] = ""
#         if "similarity" not in df.columns:
#             df["similarity"] = 0.5
#         df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.5)
#         df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.5)
#         for col in ["query_ticket_id", "feedback_ticket_id", "query_class", "query_team", "feedback_class", "feedback_team"]:
#             df[col] = df[col].fillna("").astype(str)
#         return df

#     def _build_id_index(self) -> dict[str, set[int]]:
#         mapping: dict[str, set[int]] = {}
#         for pos, row in self.kb.iterrows():
#             for value in [row.get("Ref"), row.get("sequential_id")]:
#                 key = _norm_id(value)
#                 if key:
#                     mapping.setdefault(key, set()).add(int(pos))
#         return mapping

#     def _aliases_for_id(self, ticket_id: str) -> set[str]:
#         aliases = {_norm_id(ticket_id)}
#         for pos in self.id_to_positions.get(_norm_id(ticket_id), set()):
#             row = self.kb.iloc[pos]
#             aliases.update({_norm_id(row.get("Ref")), _norm_id(row.get("sequential_id"))})
#         return {a for a in aliases if a}

#     def _query_ids(self) -> list[str]:
#         ids = sorted({_norm_id(v) for v in self.feedback["query_ticket_id"].unique() if _norm_id(v)})
#         if self.cfg.query_offset:
#             ids = ids[self.cfg.query_offset :]
#         if self.cfg.limit is not None:
#             ids = ids[: self.cfg.limit]
#         return ids

#     def _find_ticket_row(self, ticket_id: str) -> Optional[tuple[int, pd.Series]]:
#         aliases = self._aliases_for_id(ticket_id)
#         positions: set[int] = set()
#         for alias in aliases:
#             positions.update(self.id_to_positions.get(alias, set()))
#         if not positions:
#             return None
#         pos = min(positions)
#         return pos, self.kb.iloc[pos]

#     def _isolate_feedback(self, aliases: set[str]) -> tuple[pd.DataFrame, int]:
#         q = self.feedback["query_ticket_id"].astype(str).isin(aliases)
#         f = self.feedback["feedback_ticket_id"].astype(str).isin(aliases)
#         removed = int((q | f).sum())
#         return self.feedback.loc[~(q | f)].copy(), removed

#     def _fit_lifts(
#         self,
#         isolated_feedback: pd.DataFrame,
#         query_class: str,
#         query_team: str,
#         query_embedding: np.ndarray,
#     ) -> dict[str, dict]:
#         lift_fn = LiftRegistry.get(self.cfg.lift.name)
#         routing = self.cfg.routing.name
#         q_class = str(query_class or "").lower()
#         q_team = _clean_team(query_team)

#         def calc(df: pd.DataFrame) -> tuple[float, dict]:
#             if df.empty:
#                 return 0.0, {"pos": 0.0, "neg": 0.0, "count": 0, "mean_score": None}
#             pos = float((df["score"] >= 0.8).sum())
#             neg = float((df["score"] <= 0.4).sum())
#             lift = lift_fn(pos, neg, df["score"], df["similarity"], self.cfg.lift)
#             return float(lift), {
#                 "pos": pos,
#                 "neg": neg,
#                 "count": int(len(df)),
#                 "mean_score": float(df["score"].mean()),
#             }

#         out: dict[str, dict] = {}
#         target_ids = sorted({_norm_id(v) for v in isolated_feedback["feedback_ticket_id"].unique() if _norm_id(v)})
#         for rid in target_ids:
#             candidate_positions = self.id_to_positions.get(rid, set())
#             candidate_pos = min(candidate_positions) if candidate_positions else None
#             # semantic_similarity = None
#             # semantic_pass = True
#             # if routing in {"semantic", "hybrid"} or self.cfg.routing.require_semantic:
#             #     semantic_pass = False
#             #     if candidate_pos is not None:
#             #         semantic_similarity = float(np.dot(query_embedding[0], self.rag.embeddings[candidate_pos]))
#             #         semantic_pass = semantic_similarity >= self.cfg.routing.relevance_threshold
#             # if not semantic_pass:
#             #     continue
#             # ── AFTER ───────────────────────────────────────────────────────────────────
#             semantic_similarity = None
#             semantic_weight = 1.0          # default: full lift weight
#             if routing in {"semantic", "hybrid"} or self.cfg.routing.require_semantic:
#                 if candidate_pos is not None:
#                     semantic_similarity = float(np.dot(query_embedding[0], self.rag.embeddings[candidate_pos]))
#                     # Soft weighting: lift is scaled by how semantically relevant
#                     # the candidate is. Below a hard floor (0.25) we still skip —
#                     # that prevents truly unrelated tickets from getting any signal.
#                     if semantic_similarity < 0.25:
#                         continue
#                     semantic_weight = min(1.0, max(0.0,
#                         (semantic_similarity - 0.25) / (self.cfg.routing.relevance_threshold - 0.25)
#                     ))

#             target = isolated_feedback[isolated_feedback["feedback_ticket_id"].astype(str) == rid]
#             class_df = target[target["query_class"].astype(str).str.lower() == q_class]
#             team_df = target[target["query_team"].map(_clean_team) == q_team]

#             scopes: list[tuple[str, float, pd.DataFrame]] = []
#             if routing == "global":
#                 scopes.append(("global", 1.0, target))
#             elif routing == "categorical":
#                 scopes.append(("class", self.cfg.routing.w_class, class_df))
#                 scopes.append(("team", self.cfg.routing.w_team, team_df))
#             elif routing == "semantic":
#                 scopes.append(("global_semantic", 1.0, target))
#             elif routing == "hybrid":
#                 scopes.append(("global_semantic", self.cfg.routing.w_global, target))
#                 scopes.append(("class_semantic", self.cfg.routing.w_class, class_df))
#                 scopes.append(("team_semantic", self.cfg.routing.w_team, team_df))
#             else:
#                 raise ValueError(f"Unknown routing model '{routing}'")

#             weighted_lift = 0.0
#             used_weight = 0.0
#             scope_details = {}
#             for scope_name, weight, df_scope in scopes:
#                 scope_lift, stats = calc(df_scope)
#                 scope_details[scope_name] = {"weight": weight, "lift": scope_lift, **stats}
#                 # if stats["count"] > 0 and weight > 0:
#                 #     weighted_lift += weight * scope_lift
#                 #     used_weight += weight
#                 # ── AFTER ────────────────────────────────────────────────────────────────────
#                 if stats["count"] > 0 and weight > 0:
#                     weighted_lift += weight * scope_lift * semantic_weight   # ← apply sem weight
#                     used_weight += weight

#             if used_weight <= 0:
#                 continue
#             lift = weighted_lift / used_weight
#             lift = max(-self.cfg.lift.cap, min(self.cfg.lift.cap, lift))
#             if self.cfg.lift.positive_only:
#                 lift = max(0.0, lift)
#             all_stats = calc(target)[1]
#             out[rid] = {
#                 "lift": float(lift),
#                 **all_stats,
#                 "routing_rows": int(all_stats["count"]),
#                 "semantic_similarity": semantic_similarity,
#                 "semantic_weight": semantic_weight,
#                 "scopes": scope_details,
#             }
#         return out

#     def _gate_weight(self, base_top1: float, base_gap: float, max_lift: float) -> tuple[float, str, dict]:
#         name = self.cfg.gating.name
#         details = {"base_top1": base_top1, "base_gap": base_gap, "max_lift": max_lift}
#         if name == "none":
#             return 1.0, "none", details
#         if name == "static":
#             active = base_top1 < self.cfg.gating.faiss_ceiling
#             return (1.0 if active else 0.0), ("static_active" if active else "static_blocked"), details
#         if name == "fuzzy":
#             low, high = self.cfg.gating.fuzzy_low, self.cfg.gating.fuzzy_high
#             if base_top1 <= low:
#                 return 1.0, "fuzzy_full", details
#             if base_top1 >= high:
#                 return 0.0, "fuzzy_blocked", details
#             w = 1.0 - (base_top1 - low) / (high - low)
#             return float(w), "fuzzy_partial", details
#         if name == "miracle":
#             composite = base_gap * (max_lift + 0.1)
#             details["composite"] = composite
#             active = composite >= self.cfg.gating.miracle_tau
#             return (1.0 if active else 0.0), ("miracle_active" if active else "miracle_blocked"), details
#         raise ValueError(f"Unknown gating strategy '{name}'")

#     def _retrieve(
#         self,
#         query_text: str,
#         query_aliases: set[str],
#         predicted_class: str,
#         predicted_team: str,
#         isolated_feedback: pd.DataFrame,
#     ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
#         query_embedding = self.rag.sentence_model.encode([query_text])
#         import faiss

#         faiss.normalize_L2(query_embedding)
#         # search_k = min(max(self.cfg.search_k, self.cfg.top_k), len(self.kb))
#         # distances, indices = self.rag.index.search(np.array(query_embedding).astype("float32"), search_k)

#         # Force a full-table query size so we can extract all possible 
#         # unique feedback candidates sorted by their underlying proximity
#         search_k = len(self.kb)
#         distances, indices = self.rag.index.search(np.array(query_embedding).astype("float32"), search_k)

#         lifts = self._fit_lifts(isolated_feedback, predicted_class, predicted_team, query_embedding)
#         candidates = []
#         excluded_positions: set[int] = set()
#         for alias in query_aliases:
#             excluded_positions.update(self.id_to_positions.get(alias, set()))

#         for rank, idx in enumerate(indices[0]):
#             idx = int(idx)
#             if idx in excluded_positions:
#                 continue
#             row = self.kb.iloc[idx]
#             rid = _norm_id(row.get("Ref")) or _norm_id(row.get("sequential_id")) or str(idx)
#             sid = _norm_id(row.get("sequential_id"))

#             # Pool Constraint: Only retain tickets found in the feedback database
#             if rid not in self.unique_feedback_ids and sid not in self.unique_feedback_ids:
#                 continue

#             lift_info = lifts.get(rid) or lifts.get(sid) or {"lift": 0.0, "pos": 0, "neg": 0, "count": 0, "mean_score": None}
#             sim = float(distances[0][rank])
#             candidates.append(
#                 {
#                     "retrieved_id": rid,
#                     "sequential_id": sid,
#                     "Title_anon": row.get("Title_anon", ""),
#                     "Description_anon": row.get("Description_anon", ""),
#                     "first_reply": row.get("first_reply", ""),
#                     "label_auto": row.get("label_auto", ""),
#                     "Team": row.get("Team", ""),
#                     "faiss_rank": int(rank + 1),
#                     "faiss_score": sim,
#                     "enhanced_score": sim,
#                     "feedback_lift_raw": float(lift_info.get("lift", 0.0)),
#                     "feedback_pos": lift_info.get("pos", 0),
#                     "feedback_neg": lift_info.get("neg", 0),
#                     "feedback_count": lift_info.get("count", 0),
#                     "feedback_mean_score": lift_info.get("mean_score"),
#                 }
#             )

#         cand = pd.DataFrame(candidates)
#         if cand.empty:
#             return cand, cand, {"reason": "no_candidates", "al_weight": 0.0}
        
#         # Both baseline and AL sort out their top_k from this identical pool
#         base_sorted = cand.sort_values("faiss_score", ascending=False).head(self.cfg.top_k).copy()
#         base_top1 = float(base_sorted.iloc[0]["faiss_score"]) if not base_sorted.empty else 0.0
#         base_gap = 0.0
#         if len(base_sorted) > 1:
#             base_gap = float(base_sorted.iloc[0]["faiss_score"] - base_sorted.iloc[1]["faiss_score"])
#         max_lift = float(cand["feedback_lift_raw"].abs().max()) if "feedback_lift_raw" in cand else 0.0
#         al_weight, gate_reason, gate_details = self._gate_weight(base_top1, base_gap, max_lift)

       

#         cand["al_weight"] = al_weight
#         cand["feedback_lift"] = cand["feedback_lift_raw"] * al_weight
#         cand["enhanced_score"] = cand["faiss_score"] + cand["feedback_lift"]
#         cand["confidence_gated"] = al_weight == 0.0
#         al_sorted = cand.sort_values(["enhanced_score", "feedback_lift", "faiss_score"], ascending=[False, False, False]).head(self.cfg.top_k).copy()

#         metadata = {
#             "gate_reason": gate_reason,
#             "al_weight": al_weight,
#             "gate_details": gate_details,
#             "num_candidates_scored": int(len(cand)),
#             "num_nonzero_lifts_in_candidates": int((cand["feedback_lift_raw"].abs() > 1e-12).sum()),
#             "baseline_ids": base_sorted["retrieved_id"].astype(str).tolist(),
#             "al_ids": al_sorted["retrieved_id"].astype(str).tolist(),
#             "overlap_at_k": int(len(set(base_sorted["retrieved_id"].astype(str)).intersection(set(al_sorted["retrieved_id"].astype(str))))),
#         }
#         return base_sorted, al_sorted, metadata

#     async def _generate(self, title: str, description: str, predicted_class: str, predicted_team: str, team_conf: float, retrieved: pd.DataFrame) -> str:
#         return await _async_generate_response_with_openai(
#             title,
#             description,
#             predicted_class,
#             predicted_team,
#             team_conf,
#             retrieved,
#             temporal_context=rt.detect_temporal_context(title, description),
#             rt_module=rt,
#         )

#     async def _judge_generated(self, title: str, description: str, expected: str, label: str, response: str) -> dict:
#         if not self.cfg.judge_generated:
#             return {"enabled": False, "scores": [], "avg_helpfulness": None}
#         item = [{"retrieved_id": label, "Title_anon": title, "Description_anon": description, "first_reply": response}]
#         try:
#             result = await async_judge_items(title, description, item, 1, expected_first_reply=expected)
#             scores = result.get("scores", []) if isinstance(result, dict) else []
#             avg = _mean([s.get("helpfulness") for s in scores if isinstance(s, dict)])
#             return {"enabled": True, "scores": scores, "avg_helpfulness": avg}
#         except Exception as e:
#             return {"enabled": True, "error": str(e), "scores": [], "avg_helpfulness": None}

#     def _response_metrics(self, generated: str, expected: str, expected_team: str, predicted_team: str, expected_emb: Any) -> dict:
#         from sentence_transformers import util as st_util

#         cosine = None
#         if generated and expected and expected_emb is not None:
#             gen_emb = self.rag.sentence_model.encode(generated, convert_to_tensor=True)
#             cosine = float(st_util.cos_sim(expected_emb, gen_emb).item())
#         rouge_l = None
#         if self.rouge is not None and generated and expected:
#             rouge_l = float(self.rouge.score(expected, generated)["rougeL"].fmeasure)
#         flags = _quality_flags(generated, expected_team, predicted_team)
#         return {
#             "cosine_similarity": cosine,
#             "rouge_l_f1": rouge_l,
#             "bertscore_f1": None,
#             **flags,
#         }

#     async def evaluate_one(self, query_id: str, ordinal: int, total: int) -> Optional[dict]:
#         found = self._find_ticket_row(query_id)
#         if found is None:
#             return {"query_ticket_id": query_id, "skipped": True, "skip_reason": "query_not_found_in_tickets_db"}
#         query_pos, row = found
#         aliases = self._aliases_for_id(query_id)
#         isolated_feedback, removed_rows = self._isolate_feedback(aliases)

#         title = str(row.get("Title_anon", ""))
#         description = str(row.get("Description_anon", ""))
#         query_text = _ticket_text(row)
#         expected_reply = str(row.get("first_reply", ""))
#         expected_team = str(row.get("Team", ""))
#         expected_class = str(row.get("label_auto", ""))

#         # using oricle class and team
#         word_class = expected_class #word_class, _ = rt.classify_ticket(query_text)
#         predicted_team, team_confidence = expected_team, 1.0 #rt.classify_team_with_distilbert(query_text)

#         base_retrieval, al_retrieval, retrieval_meta = self._retrieve(
#             query_text, aliases, word_class, predicted_team, isolated_feedback
#         )
#         if base_retrieval.empty or al_retrieval.empty:
#             return {"query_ticket_id": query_id, "skipped": True, "skip_reason": "no_retrieval_candidates"}

#         if self.cfg.retrieval_only:
#             base_response = ""
#             al_response = ""
#         else:
#             base_response, al_response = await asyncio.gather(
#                 self._generate(title, description, word_class, predicted_team, team_confidence, base_retrieval),
#                 self._generate(title, description, word_class, predicted_team, team_confidence, al_retrieval), 
#             )
#             # base_response = await self._generate(title, description, word_class, predicted_team, team_confidence, base_retrieval) ##################izklopil al začasno#################
            
#             # al_response = base_response

#         expected_emb = self.rag.sentence_model.encode(expected_reply, convert_to_tensor=True) if expected_reply else None
#         base_metrics = self._response_metrics(base_response, expected_reply, expected_team, predicted_team, expected_emb)
#         al_metrics = self._response_metrics(al_response, expected_reply, expected_team, predicted_team, expected_emb)
#         judge_base, judge_al = await asyncio.gather(
#             self._judge_generated(title, description, expected_reply, "generated_baseline", base_response),
#             self._judge_generated(title, description, expected_reply, "generated_al", al_response),
#         )
#         base_metrics["judge"] = judge_base
#         al_metrics["judge"] = judge_al

#         base_cos = base_metrics["cosine_similarity"]
#         al_cos = al_metrics["cosine_similarity"]
#         if base_cos is None or al_cos is None:
#             metric_msg = "metrics=pending"
#         else:
#             metric_msg = f"cos base={base_cos:.4f} al={al_cos:.4f} delta={al_cos - base_cos:+.4f}"
#         print(f"[{ordinal}/{total}] {query_id}: {metric_msg} gate={retrieval_meta.get('gate_reason')}")

#         return {
#             "query_ticket_id": query_id,
#             "query_aliases": sorted(aliases),
#             "query_kb_position": int(query_pos),
#             "query": {
#                 "Ref": str(row.get("Ref", "")),
#                 "sequential_id": str(row.get("sequential_id", "")),
#                 "title": title,
#                 "description": description,
#                 "expected_first_reply": expected_reply,
#                 "expected_team": expected_team,
#                 "expected_class": expected_class,
#             },
#             "isolation": {
#                 "removed_feedback_rows": removed_rows,
#                 "remaining_feedback_rows": int(len(isolated_feedback)),
#                 "excluded_retrieval_aliases": sorted(aliases),
#             },
#             "classification": {
#                 "predicted_class": word_class,
#                 "predicted_team": predicted_team,
#                 "team_confidence": float(team_confidence),
#             },
#             "retrieval_metadata": retrieval_meta,
#             "baseline": {
#                 "retrieval_model": "pure_faiss",
#                 "retrieval": base_retrieval.to_dict(orient="records"),
#                 "response": base_response,
#                 "metrics": base_metrics,
#             },
#             "active_learning": {
#                 "routing_model": asdict(self.cfg.routing),
#                 "gating_method": asdict(self.cfg.gating),
#                 "lift_formula": asdict(self.cfg.lift),
#                 "retrieval_model": "faiss_plus_feedback_lift",
#                 "retrieval": al_retrieval.to_dict(orient="records"),
#                 "response": al_response,
#                 "metrics": al_metrics,
#             },
#         }

#     def _add_bertscore(self, results: list[dict]) -> None:
#         if not self.cfg.calculate_bert or not results:
#             return
#         try:
#             from bert_score import score as bert_score_fn
#         except Exception as e:
#             print(f"BERTScore unavailable: {e}")
#             return

#         base_cand, base_ref, base_idx = [], [], []
#         al_cand, al_ref, al_idx = [], [], []
#         for i, r in enumerate(results):
#             if r.get("skipped"):
#                 continue
#             expected = r["query"]["expected_first_reply"]
#             b = r["baseline"]["response"]
#             a = r["active_learning"]["response"]
#             if expected and b:
#                 base_cand.append(b)
#                 base_ref.append(expected)
#                 base_idx.append(i)
#             if expected and a:
#                 al_cand.append(a)
#                 al_ref.append(expected)
#                 al_idx.append(i)

#         if base_cand:
#             _, _, f1 = bert_score_fn(base_cand, base_ref, lang="en", verbose=False)
#             for j, idx in enumerate(base_idx):
#                 results[idx]["baseline"]["metrics"]["bertscore_f1"] = float(f1[j])
#         if al_cand:
#             _, _, f1 = bert_score_fn(al_cand, al_ref, lang="en", verbose=False)
#             for j, idx in enumerate(al_idx):
#                 results[idx]["active_learning"]["metrics"]["bertscore_f1"] = float(f1[j])

#     def _summary(self, results: list[dict], started: float) -> dict:
#         valid = [r for r in results if not r.get("skipped")]
#         def vals(path: tuple[str, ...]) -> list[Optional[float]]:
#             out = []
#             for r in valid:
#                 cur: Any = r
#                 for p in path:
#                     cur = cur.get(p, {}) if isinstance(cur, dict) else {}
#                 out.append(cur if isinstance(cur, (int, float)) else None)
#             return out

#         base_cos = vals(("baseline", "metrics", "cosine_similarity"))
#         al_cos = vals(("active_learning", "metrics", "cosine_similarity"))
#         base_rouge = vals(("baseline", "metrics", "rouge_l_f1"))
#         al_rouge = vals(("active_learning", "metrics", "rouge_l_f1"))
#         base_bert = vals(("baseline", "metrics", "bertscore_f1"))
#         al_bert = vals(("active_learning", "metrics", "bertscore_f1"))

#         return {
#             "timestamp": datetime.now().isoformat(timespec="seconds"),
#             "duration_s": time.time() - started,
#             "total_queries_requested": len(self.query_ids),
#             "total_results": len(results),
#             "valid_results": len(valid),
#             "skipped": len(results) - len(valid),
#             "metrics": {
#                 "cosine_base_mean": _mean(base_cos),
#                 "cosine_al_mean": _mean(al_cos),
#                 "cosine_delta_mean": (_mean(al_cos) or 0.0) - (_mean(base_cos) or 0.0),
#                 "cosine_base_std": _std(base_cos),
#                 "cosine_al_std": _std(al_cos),
#                 "rouge_base_mean": _mean(base_rouge),
#                 "rouge_al_mean": _mean(al_rouge),
#                 "rouge_delta_mean": (_mean(al_rouge) or 0.0) - (_mean(base_rouge) or 0.0),
#                 "bert_base_mean": _mean(base_bert),
#                 "bert_al_mean": _mean(al_bert),
#                 "bert_delta_mean": (_mean(al_bert) or 0.0) - (_mean(base_bert) or 0.0),
#             },
#             "retrieval": {
#                 "mean_overlap_at_k": _mean([r.get("retrieval_metadata", {}).get("overlap_at_k") for r in valid]),
#                 "mean_nonzero_lift_candidates": _mean([r.get("retrieval_metadata", {}).get("num_nonzero_lifts_in_candidates") for r in valid]),
#                 "al_changed_rate": _mean([
#                     1.0 if r.get("retrieval_metadata", {}).get("baseline_ids") != r.get("retrieval_metadata", {}).get("al_ids") else 0.0
#                     for r in valid
#                 ]),
#                 "gate_counts": pd.Series([r.get("retrieval_metadata", {}).get("gate_reason", "unknown") for r in valid]).value_counts().to_dict()
#                 if valid
#                 else {},
#             },
#             "config": asdict(self.cfg),
#             "fingerprint": {
#                 "platform": platform.platform(),
#                 "python": platform.python_version(),
#                 "tickets_db": _file_fingerprint(self.cfg.tickets_db),
#                 "feedback_db": _file_fingerprint(self.cfg.feedback_db),
#                 "openrouter": {
#                     "base_url": os.getenv("OPENROUTER_BASE_URL"),
#                     "strict_consistency": os.getenv("OPENROUTER_STRICT_CONSISTENCY"),
#                     "provider": os.getenv("OPENROUTER_PROVIDER"),
#                     "cache": os.getenv("OPENROUTER_ENABLE_CACHE"),
#                 },
#             },
#         }

#     async def run(self) -> dict:
#         started = time.time()
#         results: list[dict] = []
#         total = len(self.query_ids)
#         for i, qid in enumerate(self.query_ids, 1):
#             try:
#                 result = await self.evaluate_one(qid, i, total)
#             except Exception as e:
#                 result = {"query_ticket_id": qid, "skipped": True, "skip_reason": "exception", "error": repr(e)}
#                 print(f"[{i}/{total}] {qid}: ERROR {e}")
#             if result is not None:
#                 results.append(result)

#             if i == 1 or i % 10 == 0:
#                 self._save_interim(results)

#         self._add_bertscore(results)
#         summary = self._summary(results, started)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         stem = f"modular_looe_{Path(self.cfg.feedback_db).stem}_{self.cfg.lift.name}_{self.cfg.routing.name}_{self.cfg.gating.name}_{timestamp}"
#         details_path = self.results_dir / f"{stem}_details.json"
#         summary_path = self.results_dir / f"{stem}_summary.json"
#         with open(details_path, "w", encoding="utf-8") as f:
#             json.dump(_json_safe(results), f, indent=2, ensure_ascii=False)
#         with open(summary_path, "w", encoding="utf-8") as f:
#             json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
#         summary["details_path"] = str(details_path)
#         summary["summary_path"] = str(summary_path)
#         print(f"Saved details: {details_path}")
#         print(f"Saved summary: {summary_path}")
#         return summary

#     def _save_interim(self, results: list[dict]) -> None:
#         path = self.results_dir / "modular_looe_interim.json"
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(_json_safe(results), f, indent=2, ensure_ascii=False)


# def build_config(args: argparse.Namespace) -> EvalConfig:
#     lift = LiftConfig(
#         name=args.lift,
#         alpha=args.alpha,
#         beta=args.beta,
#         multiplier=args.lift_multiplier,
#         cap=args.lift_cap,
#         sensitivity=args.tanh_sensitivity,
#         lcb_k=args.lcb_k,
#         positive_only=args.positive_only,
#     )
#     routing = RoutingConfig(
#         name=args.routing,
#         w_global=args.w_global,
#         w_class=args.w_class,
#         w_team=args.w_team,
#         relevance_threshold=args.relevance_threshold,
#         require_semantic=args.require_semantic,
#     )
#     gating = GatingConfig(
#         name=args.gating,
#         faiss_ceiling=args.faiss_ceiling,
#         fuzzy_low=args.fuzzy_low,
#         fuzzy_high=args.fuzzy_high,
#         miracle_tau=args.miracle_tau,
#     )
#     return EvalConfig(
#         tickets_db=args.tickets_db,
#         feedback_db=args.feedback_db,
#         results_dir=args.results_dir,
#         top_k=args.top_k,
#         search_k=args.search_k,
#         sentence_model=args.sentence_model,
#         llm_model=args.llm_model,
#         judge_model=args.judge_model,
#         lift=lift,
#         routing=routing,
#         gating=gating,
#         calculate_bert=args.bert,
#         judge_generated=args.judge_generated,
#         retrieval_only=args.retrieval_only,
#         limit=args.limit,
#         query_offset=args.query_offset,
#     )


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Strict modular LOO evaluation for feedback-driven active learning retrieval.")
#     parser.add_argument("--tickets-db", default="tickets.db")
#     parser.add_argument("--feedback-db", default="comprehensive_feedback_250x250.db")
#     parser.add_argument("--results-dir", default="test_results_modular_looe")
#     parser.add_argument("--limit", type=int, default=None)
#     parser.add_argument("--query-offset", type=int, default=0)
#     parser.add_argument("--top-k", type=int, default=5)
#     parser.add_argument("--search-k", type=int, default=100)
#     parser.add_argument("--sentence-model", default=os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2"))
#     parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo-0613"))
#     parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"))

#     parser.add_argument("--lift", choices=["laplace", "tanh", "bayesian_lcb"], default="tanh")
#     parser.add_argument("--alpha", type=float, default=1.0)
#     parser.add_argument("--beta", type=float, default=1.0)
#     parser.add_argument("--lift-multiplier", type=float, default=0.60)
#     parser.add_argument("--lift-cap", type=float, default=0.20)
#     parser.add_argument("--tanh-sensitivity", type=float, default=5.0)
#     parser.add_argument("--lcb-k", type=float, default=1.0)
#     parser.add_argument("--positive-only", action="store_true")

#     parser.add_argument("--routing", choices=["global", "categorical", "semantic", "hybrid"], default="hybrid")
#     parser.add_argument("--w-global", type=float, default=0.10)
#     parser.add_argument("--w-class", type=float, default=0.55)
#     parser.add_argument("--w-team", type=float, default=0.35)
#     parser.add_argument("--relevance-threshold", type=float, default=0.0)#0.70)
#     parser.add_argument("--require-semantic", action="store_true")

#     parser.add_argument("--gating", choices=["none", "static", "fuzzy", "miracle"], default="static")
#     parser.add_argument("--faiss-ceiling", type=float, default=0.656)
#     parser.add_argument("--fuzzy-low", type=float, default=0.50)
#     parser.add_argument("--fuzzy-high", type=float, default=0.75)
#     parser.add_argument("--miracle-tau", type=float, default=0.005)

#     parser.add_argument("--bert", action="store_true", help="Calculate batched BERTScore at the end.")
#     parser.add_argument("--judge-generated", action="store_true", help="Use the configured judge LLM on generated baseline and AL responses.")
#     parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM generation and response metrics; useful for testing isolation and reranking.")
#     return parser.parse_args()


# if __name__ == "__main__":
#     config = build_config(parse_args())
#     engine = ModularLOOActiveLearning(config)
#     asyncio.run(engine.run())


"""
Modular leave-one-out evaluation for feedback-driven active learning retrieval.
 
This script evaluates baseline FAISS retrieval against feedback-lift reranking
with strict per-query isolation:
  - all feedback rows involving the query ticket are removed before lift fitting
  - the query ticket is removed from retrieval candidates before generation
  - baseline and AL use the same classifier outputs, generator, and metrics
 
It is designed for:
  - comprehensive_feedback_250x250.db
  - comprehensive_feedback_v3.db
 
Examples:
  python run_modular_looe_active_learning.py --feedback-db comprehensive_feedback_250x250.db --limit 10
  python run_modular_looe_active_learning.py --feedback-db comprehensive_feedback_v3.db --lift tanh --gating fuzzy --llm-model openai/gpt-4o-mini
"""
from __future__ import annotations
 
import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
 
import numpy as np
import pandas as pd
 
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
 
try:
    from dotenv import load_dotenv
 
    load_dotenv(override=True)
except Exception:
    pass
 
rt = None
_async_generate_response_with_openai = None
async_judge_items = None
 
 
def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple)) else False:
        return None
    return obj
 
 
def _sha256_file(path: str | Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None
 
 
def _file_fingerprint(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": p.stat().st_size,
        "mtime": p.stat().st_mtime,
        "sha256": _sha256_file(p),
    }
 
 
def _mean(values: list[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None
 
 
def _std(values: list[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.std(clean)) if clean else None
 
 
def _norm_id(value: Any) -> str:
    return str(value).strip()
 
 
def _clean_team(team: Any) -> str:
    return str(team or "").replace(",", "").replace("|", "").strip()
 
 
def _ticket_text(row: pd.Series) -> str:
    return f"{row.get('Title_anon', '')} {row.get('Description_anon', '')}".strip()
 
 
def _quality_flags(generated: str, expected_team: str, predicted_team: str) -> dict:
    text = str(generated or "")
    return {
        "length": len(text),
        "has_greeting": any(g in text.lower() for g in ["dear", "hello", "thank you", "hi"]),
        "has_structure": text.count("\n-") >= 2 or text.count(":") >= 3,
        "team_match": str(expected_team) == str(predicted_team),
    }
 
 
@dataclass(frozen=True)
class LiftConfig:
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    multiplier: float = 0.60
    cap: float = 0.20
    sensitivity: float = 5.0
    lcb_k: float = 1.0
    positive_only: bool = False
 
 
@dataclass(frozen=True)
class RoutingConfig:
    name: str
    w_global: float = 0.10
    w_class: float = 0.55
    w_team: float = 0.35
    relevance_threshold: float = 0.70
    require_semantic: bool = False
 
 
@dataclass(frozen=True)
class GatingConfig:
    name: str
    faiss_ceiling: float = 0.656
    fuzzy_low: float = 0.50
    fuzzy_high: float = 0.75
    miracle_tau: float = 0.005
 
 
@dataclass(frozen=True)
class EvalConfig:
    tickets_db: str
    feedback_db: str
    results_dir: str
    top_k: int
    search_k: int
    sentence_model: str
    llm_model: str
    judge_model: Optional[str]
    lift: LiftConfig
    routing: RoutingConfig
    gating: GatingConfig
    calculate_bert: bool
    judge_generated: bool
    retrieval_only: bool
    limit: Optional[int]
    query_offset: int
    concurrency: int = 1
 
 
class LiftRegistry:
    @staticmethod
    def laplace(pos: float, neg: float, scores: pd.Series, sims: pd.Series, cfg: LiftConfig) -> float:
        weighted_neg = 0.0 if cfg.positive_only else neg
        p = (pos + cfg.alpha) / (pos + weighted_neg + cfg.alpha + cfg.beta)
        n = pos if cfg.positive_only else pos + neg
        scale = min(1.0, n / 2.0)
        return (p - 0.5) * scale * cfg.multiplier
 
    @staticmethod
    def tanh(pos: float, neg: float, scores: pd.Series, sims: pd.Series, cfg: LiftConfig) -> float:
        sim_values = sims.fillna(0.5).replace(0, 0.5).astype(float)
        evidence = ((scores.astype(float) - 0.5) * sim_values).sum()
        return math.tanh(float(evidence) / cfg.sensitivity) * cfg.cap
 
    @staticmethod
    def bayesian_lcb(pos: float, neg: float, scores: pd.Series, sims: pd.Series, cfg: LiftConfig) -> float:
        vals = scores.astype(float)
        if vals.empty:
            return 0.0
        mu = float(vals.mean()) - 0.5
        sigma = float(vals.std(ddof=0)) if len(vals) > 1 else 0.25
        uncertainty = cfg.lcb_k * sigma / math.sqrt(max(1, len(vals)))
        return (mu - uncertainty) * cfg.multiplier
 
    @classmethod
    def get(cls, name: str) -> Callable[[float, float, pd.Series, pd.Series, LiftConfig], float]:
        mapping = {
            "laplace": cls.laplace,
            "tanh": cls.tanh,
            "bayesian_lcb": cls.bayesian_lcb,
        }
        if name not in mapping:
            raise ValueError(f"Unknown lift formula '{name}'. Choose one of {sorted(mapping)}")
        return mapping[name]
 
 
class ModularLOOActiveLearning:
    def __init__(self, cfg: EvalConfig):
        global rt, _async_generate_response_with_openai, async_judge_items
        if rt is None:
            import resolution_task as _rt
            from resolution_task_async import (
                _async_generate_response_with_openai as _async_generate,
                async_judge_items as _async_judge_items,
            )
 
            rt = _rt
            _async_generate_response_with_openai = _async_generate
            async_judge_items = _async_judge_items
 
        self.cfg = cfg
        self.results_dir = Path(cfg.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
 
        if cfg.llm_model:
            os.environ["LLM_MODEL"] = cfg.llm_model
            rt.LLM_MODEL = cfg.llm_model
 
            rt.GEN_MODEL_TEMPLATE = cfg.llm_model
            rt.GEN_MODEL_PERSONAL = cfg.llm_model
            rt.GEN_MODEL_SHORT = cfg.llm_model
            rt.GEN_MODEL_STATUS = cfg.llm_model
        if cfg.judge_model:
            os.environ["JUDGE_MODEL"] = cfg.judge_model
        os.environ.setdefault("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
 
        self.kb = self._load_kb(cfg.tickets_db)
        self.feedback = self._load_feedback(cfg.feedback_db)
 
        # Extract unique ticket IDs referenced in the feedback database
        feedback_ids = set(self.feedback["query_ticket_id"].unique()).union(
            set(self.feedback["feedback_ticket_id"].unique())
        )
        self.unique_feedback_ids = {str(tid).strip() for tid in feedback_ids if tid}
 
        # BUGFIX: tickets.db's label_auto (sourced from service_subcategory) uses a
        # granular, free-text taxonomy (149 distinct raw values in this dataset) that
        # never matches the coarse 10-category taxonomy stored in the feedback DB's
        # query_class / feedback_class columns. Comparing them (as the old code did
        # via `row.get("label_auto")`) means class_df is empty for ~every ticket,
        # silently zeroing out any class-only or class-weighted lift. Build a lookup
        # straight from the feedback DB's own class columns instead -- it already
        # carries the correct coarse label for every ticket this pipeline touches.
        self._class_lookup: dict[str, str] = {}
        for tid, cls in zip(self.feedback["query_ticket_id"], self.feedback["query_class"]):
            tid, cls = str(tid).strip(), str(cls).strip().lower()
            if tid and cls and cls != "nan":
                self._class_lookup[tid] = cls
        for tid, cls in zip(self.feedback["feedback_ticket_id"], self.feedback["feedback_class"]):
            tid, cls = str(tid).strip(), str(cls).strip().lower()
            if tid and cls and cls != "nan":
                self._class_lookup.setdefault(tid, cls)
        print(f"[class-lookup] built coarse-class mapping for {len(self._class_lookup)} tickets "
              f"from feedback DB (tickets.db's label_auto taxonomy is incompatible and is no longer used for routing).")
 
        self.rag = rt.RAGSystem(self.kb, sentence_model_name=cfg.sentence_model, kb_path=cfg.tickets_db)
        self.rag.build_index()
        self.id_to_positions = self._build_id_index()
        self.query_ids = self._query_ids()
 
        self.rouge = None
        try:
            from rouge_score import rouge_scorer
 
            self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        except Exception:
            pass
 
    def _load_kb(self, tickets_db: str) -> pd.DataFrame:
        con = sqlite3.connect(tickets_db)
        try:
            df = pd.read_sql_query("SELECT * FROM tickets", con)
        finally:
            con.close()
        df = df.rename(
            columns={
                "ticket_id": "Ref",
                "title": "Title_anon",
                "description": "Description_anon",
                "service_subcategory": "label_auto",
                "team": "Team",
            }
        )
        for col in ["Ref", "sequential_id", "Title_anon", "Description_anon", "first_reply", "label_auto", "Team"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)
        return df.reset_index(drop=True)
 
    def _load_feedback(self, feedback_db: str) -> pd.DataFrame:
        con = sqlite3.connect(feedback_db)
        try:
            df = pd.read_sql_query("SELECT * FROM feedback", con)
        finally:
            con.close()
        required = [
            "query_ticket_id",
            "query_class",
            "query_team",
            "feedback_ticket_id",
            "feedback_class",
            "feedback_team",
            "score",
        ]
        for col in required:
            if col not in df.columns:
                df[col] = ""
        if "similarity" not in df.columns:
            df["similarity"] = 0.5
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.5)
        df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.5)
        for col in ["query_ticket_id", "feedback_ticket_id", "query_class", "query_team", "feedback_class", "feedback_team"]:
            df[col] = df[col].fillna("").astype(str)
 
        n_queries = df["query_ticket_id"].nunique()
        n_feedback_tickets = df["feedback_ticket_id"].nunique()
        print(
            f"[feedback-db-check] {feedback_db}: {len(df)} rows | "
            f"{n_queries} unique query tickets | {n_feedback_tickets} unique KB/feedback tickets"
        )
        if n_queries < 240 or n_feedback_tickets < 1000:
            print(
                "[feedback-db-check] WARNING: this looks smaller than the frozen "
                "comprehensive_feedback_v3.db (250 query / ~2813 KB tickets). "
                "If you intended v3, double-check --feedback-db."
            )
        return df
 
    def _build_id_index(self) -> dict[str, set[int]]:
        mapping: dict[str, set[int]] = {}
        for pos, row in self.kb.iterrows():
            for value in [row.get("Ref"), row.get("sequential_id")]:
                key = _norm_id(value)
                if key:
                    mapping.setdefault(key, set()).add(int(pos))
        return mapping
 
    def _aliases_for_id(self, ticket_id: str) -> set[str]:
        aliases = {_norm_id(ticket_id)}
        for pos in self.id_to_positions.get(_norm_id(ticket_id), set()):
            row = self.kb.iloc[pos]
            aliases.update({_norm_id(row.get("Ref")), _norm_id(row.get("sequential_id"))})
        return {a for a in aliases if a}
 
    def _query_ids(self) -> list[str]:
        ids = sorted({_norm_id(v) for v in self.feedback["query_ticket_id"].unique() if _norm_id(v)})
        if self.cfg.query_offset:
            ids = ids[self.cfg.query_offset :]
        if self.cfg.limit is not None:
            ids = ids[: self.cfg.limit]
        return ids
 
    def _find_ticket_row(self, ticket_id: str) -> Optional[tuple[int, pd.Series]]:
        aliases = self._aliases_for_id(ticket_id)
        positions: set[int] = set()
        for alias in aliases:
            positions.update(self.id_to_positions.get(alias, set()))
        if not positions:
            return None
        pos = min(positions)
        return pos, self.kb.iloc[pos]
 
    def _isolate_feedback(self, aliases: set[str]) -> tuple[pd.DataFrame, int]:
        q = self.feedback["query_ticket_id"].astype(str).isin(aliases)
        f = self.feedback["feedback_ticket_id"].astype(str).isin(aliases)
        removed = int((q | f).sum())
        return self.feedback.loc[~(q | f)].copy(), removed
 
    def _fit_lifts(
        self,
        isolated_feedback: pd.DataFrame,
        query_class: str,
        query_team: str,
        query_embedding: np.ndarray,
    ) -> dict[str, dict]:
        lift_fn = LiftRegistry.get(self.cfg.lift.name)
        routing = self.cfg.routing.name
        q_class = str(query_class or "").lower()
        q_team = _clean_team(query_team)
 
        def calc(df: pd.DataFrame) -> tuple[float, dict]:
            if df.empty:
                return 0.0, {"pos": 0.0, "neg": 0.0, "count": 0, "mean_score": None}
            pos = float((df["score"] >= 0.8).sum())
            neg = float((df["score"] <= 0.4).sum())
            lift = lift_fn(pos, neg, df["score"], df["similarity"], self.cfg.lift)
            return float(lift), {
                "pos": pos,
                "neg": neg,
                "count": int(len(df)),
                "mean_score": float(df["score"].mean()),
            }
 
        out: dict[str, dict] = {}
 
        # --- BUGFIX: canonicalize feedback_ticket_id before grouping -------------
        # comprehensive_feedback_v3.db's `feedback_ticket_id` column mixes two
        # incompatible ID schemes for the *same* physical KB tickets: some rows use
        # tickets.db's internal sequential_id ("R-2"), others use the raw original
        # ticket_id/Ref ("R-544314"). Measured on the current DB, 79% of referenced
        # KB tickets (85% of all feedback rows) have their feedback rows split
        # across BOTH forms. Grouping by the raw string (the old `target_ids` /
        # `target = isolated_feedback[... == rid]` below) silently computed the
        # Laplace lift from only ONE half of the real feedback history for most
        # tickets, and the downstream `lifts.get(rid) or lifts.get(sid)` lookup in
        # `_score_candidates` picked whichever half happened to exist under the
        # Ref-style key -- discarding the rest, not randomly, but systematically.
        #
        # Fix: resolve every feedback_ticket_id to the physical KB row it refers to
        # (via the same id_to_positions index already used for query-ticket alias
        # resolution) and group on THAT, so every candidate's lift is computed from
        # its full feedback history regardless of which naming convention tagged
        # each row.
        fb = isolated_feedback.copy()
        fb["_canonical_pos"] = fb["feedback_ticket_id"].astype(str).map(
            lambda v: min(self.id_to_positions.get(_norm_id(v), {-1}))
        )
        fb = fb[fb["_canonical_pos"] >= 0]
        # rows whose feedback_ticket_id matches neither ID scheme (e.g. stray
        # sentinel/placeholder values such as "R-0" or a leaked feedback-record ID)
        # are dropped here rather than silently mis-scored under a fabricated key.
        n_dropped = len(isolated_feedback) - len(fb)
        if n_dropped:
            pass  # expected to be tiny (2 known garbage IDs in the current DB); not worth logging per-ticket
 
        for candidate_pos, target in fb.groupby("_canonical_pos"):
            candidate_pos = int(candidate_pos)
            kb_row = self.kb.iloc[candidate_pos]
            rid = _norm_id(kb_row.get("Ref")) or _norm_id(kb_row.get("sequential_id")) or str(candidate_pos)
            # ── AFTER ───────────────────────────────────────────────────────────────────
            semantic_similarity = None
            semantic_weight = 1.0          # default: full lift weight
            if routing in {"semantic", "hybrid"} or self.cfg.routing.require_semantic:
                semantic_similarity = float(np.dot(query_embedding[0], self.rag.embeddings[candidate_pos]))
                # Soft weighting: lift is scaled by how semantically relevant
                # the candidate is. Below a hard floor (0.25) we still skip —
                # that prevents truly unrelated tickets from getting any signal.
                if semantic_similarity < 0.25:
                    continue
                semantic_weight = min(1.0, max(0.0,
                    (semantic_similarity - 0.25) / (self.cfg.routing.relevance_threshold - 0.25)
                ))
 
            class_df = target[target["query_class"].astype(str).str.lower() == q_class]
            team_df = target[target["query_team"].map(_clean_team) == q_team]
 
            scopes: list[tuple[str, float, pd.DataFrame]] = []
            if routing == "global":
                scopes.append(("global", 1.0, target))
            elif routing == "categorical":
                scopes.append(("class", self.cfg.routing.w_class, class_df))
                scopes.append(("team", self.cfg.routing.w_team, team_df))
            elif routing == "semantic":
                scopes.append(("global_semantic", 1.0, target))
            elif routing == "hybrid":
                scopes.append(("global_semantic", self.cfg.routing.w_global, target))
                scopes.append(("class_semantic", self.cfg.routing.w_class, class_df))
                scopes.append(("team_semantic", self.cfg.routing.w_team, team_df))
            elif routing == "categorical_intersection":
                class_team_df = target[
                    (target["query_class"].astype(str).str.lower() == q_class)
                    & (target["query_team"].map(_clean_team) == q_team)
                ]
                scopes.append(("class_team", 1.0, class_team_df))
            else:
                raise ValueError(f"Unknown routing model '{routing}'")
 
            weighted_lift = 0.0
            used_weight = 0.0
            scope_details = {}
            for scope_name, weight, df_scope in scopes:
                scope_lift, stats = calc(df_scope)
                scope_details[scope_name] = {"weight": weight, "lift": scope_lift, **stats}
                # if stats["count"] > 0 and weight > 0:
                #     weighted_lift += weight * scope_lift
                #     used_weight += weight
                # ── AFTER ────────────────────────────────────────────────────────────────────
                if stats["count"] > 0 and weight > 0:
                    weighted_lift += weight * scope_lift * semantic_weight   # ← apply sem weight
                    used_weight += weight
 
            if used_weight <= 0:
                continue
            lift = weighted_lift / used_weight
            lift = max(-self.cfg.lift.cap, min(self.cfg.lift.cap, lift))
            if self.cfg.lift.positive_only:
                lift = max(0.0, lift)
            all_stats = calc(target)[1]
            out[rid] = {
                "lift": float(lift),
                **all_stats,
                "routing_rows": int(all_stats["count"]),
                "semantic_similarity": semantic_similarity,
                "semantic_weight": semantic_weight,
                "scopes": scope_details,
            }
        return out
 
    def _gate_weight(self, base_top1: float, base_gap: float, max_lift: float) -> tuple[float, str, dict]:
        name = self.cfg.gating.name
        details = {"base_top1": base_top1, "base_gap": base_gap, "max_lift": max_lift}
        if name == "none":
            return 1.0, "none", details
        if name == "static":
            active = base_top1 < self.cfg.gating.faiss_ceiling
            return (1.0 if active else 0.0), ("static_active" if active else "static_blocked"), details
        if name == "fuzzy":
            low, high = self.cfg.gating.fuzzy_low, self.cfg.gating.fuzzy_high
            if base_top1 <= low:
                return 1.0, "fuzzy_full", details
            if base_top1 >= high:
                return 0.0, "fuzzy_blocked", details
            w = 1.0 - (base_top1 - low) / (high - low)
            return float(w), "fuzzy_partial", details
        if name == "miracle":
            composite = base_gap * (max_lift + 0.1)
            details["composite"] = composite
            active = composite >= self.cfg.gating.miracle_tau
            return (1.0 if active else 0.0), ("miracle_active" if active else "miracle_blocked"), details
        raise ValueError(f"Unknown gating strategy '{name}'")
 
    def _retrieve(
        self,
        query_text: str,
        query_aliases: set[str],
        predicted_class: str,
        predicted_team: str,
        isolated_feedback: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        query_embedding = self.rag.sentence_model.encode([query_text])
        import faiss
 
        faiss.normalize_L2(query_embedding)
        # search_k = min(max(self.cfg.search_k, self.cfg.top_k), len(self.kb))
        # distances, indices = self.rag.index.search(np.array(query_embedding).astype("float32"), search_k)
 
        # Force a full-table query size so we can extract all possible 
        # unique feedback candidates sorted by their underlying proximity
        search_k = len(self.kb)
        distances, indices = self.rag.index.search(np.array(query_embedding).astype("float32"), search_k)
 
        lifts = self._fit_lifts(isolated_feedback, predicted_class, predicted_team, query_embedding)
        candidates = []
        excluded_positions: set[int] = set()
        for alias in query_aliases:
            excluded_positions.update(self.id_to_positions.get(alias, set()))
 
        for rank, idx in enumerate(indices[0]):
            idx = int(idx)
            if idx in excluded_positions:
                continue
            row = self.kb.iloc[idx]
            rid = _norm_id(row.get("Ref")) or _norm_id(row.get("sequential_id")) or str(idx)
            sid = _norm_id(row.get("sequential_id"))
 
            # Pool Constraint: Only retain tickets found in the feedback database
            if rid not in self.unique_feedback_ids and sid not in self.unique_feedback_ids:
                continue
 
            lift_info = lifts.get(rid) or lifts.get(sid) or {"lift": 0.0, "pos": 0, "neg": 0, "count": 0, "mean_score": None}
            sim = float(distances[0][rank])
            candidates.append(
                {
                    "retrieved_id": rid,
                    "sequential_id": sid,
                    "Title_anon": row.get("Title_anon", ""),
                    "Description_anon": row.get("Description_anon", ""),
                    "first_reply": row.get("first_reply", ""),
                    "label_auto": row.get("label_auto", ""),
                    "Team": row.get("Team", ""),
                    "faiss_rank": int(rank + 1),
                    "faiss_score": sim,
                    "enhanced_score": sim,
                    "feedback_lift_raw": float(lift_info.get("lift", 0.0)),
                    "feedback_pos": lift_info.get("pos", 0),
                    "feedback_neg": lift_info.get("neg", 0),
                    "feedback_count": lift_info.get("count", 0),
                    "feedback_mean_score": lift_info.get("mean_score"),
                }
            )
 
        cand = pd.DataFrame(candidates)
        if cand.empty:
            return cand, cand, {"reason": "no_candidates", "al_weight": 0.0}
        
        # Both baseline and AL sort out their top_k from this identical pool
        base_sorted = cand.sort_values("faiss_score", ascending=False).head(self.cfg.top_k).copy()
        base_top1 = float(base_sorted.iloc[0]["faiss_score"]) if not base_sorted.empty else 0.0
        base_gap = 0.0
        if len(base_sorted) > 1:
            base_gap = float(base_sorted.iloc[0]["faiss_score"] - base_sorted.iloc[1]["faiss_score"])
        max_lift = float(cand["feedback_lift_raw"].abs().max()) if "feedback_lift_raw" in cand else 0.0
        al_weight, gate_reason, gate_details = self._gate_weight(base_top1, base_gap, max_lift)
 
       
 
        cand["al_weight"] = al_weight
        cand["feedback_lift"] = cand["feedback_lift_raw"] * al_weight
        cand["enhanced_score"] = cand["faiss_score"] + cand["feedback_lift"]
        cand["confidence_gated"] = al_weight == 0.0
        al_sorted = cand.sort_values(["enhanced_score", "feedback_lift", "faiss_score"], ascending=[False, False, False]).head(self.cfg.top_k).copy()
 
        metadata = {
            "gate_reason": gate_reason,
            "al_weight": al_weight,
            "gate_details": gate_details,
            "num_candidates_scored": int(len(cand)),
            "num_nonzero_lifts_in_candidates": int((cand["feedback_lift_raw"].abs() > 1e-12).sum()),
            "baseline_ids": base_sorted["retrieved_id"].astype(str).tolist(),
            "al_ids": al_sorted["retrieved_id"].astype(str).tolist(),
            "overlap_at_k": int(len(set(base_sorted["retrieved_id"].astype(str)).intersection(set(al_sorted["retrieved_id"].astype(str))))),
        }
        return base_sorted, al_sorted, metadata
 
    async def _generate(self, title: str, description: str, predicted_class: str, predicted_team: str, team_conf: float, retrieved: pd.DataFrame) -> str:
        return await _async_generate_response_with_openai(
            title,
            description,
            predicted_class,
            predicted_team,
            team_conf,
            retrieved,
            temporal_context=rt.detect_temporal_context(title, description),
            rt_module=rt,
        )
 
    async def _judge_generated(self, title: str, description: str, expected: str, label: str, response: str) -> dict:
        if not self.cfg.judge_generated:
            return {"enabled": False, "scores": [], "avg_helpfulness": None}
        item = [{"retrieved_id": label, "Title_anon": title, "Description_anon": description, "first_reply": response}]
        try:
            result = await async_judge_items(title, description, item, 1, expected_first_reply=expected)
            scores = result.get("scores", []) if isinstance(result, dict) else []
            avg = _mean([s.get("helpfulness") for s in scores if isinstance(s, dict)])
            return {"enabled": True, "scores": scores, "avg_helpfulness": avg}
        except Exception as e:
            return {"enabled": True, "error": str(e), "scores": [], "avg_helpfulness": None}
 
    def _response_metrics(self, generated: str, expected: str, expected_team: str, predicted_team: str, expected_emb: Any) -> dict:
        from sentence_transformers import util as st_util
 
        cosine = None
        if generated and expected and expected_emb is not None:
            gen_emb = self.rag.sentence_model.encode(generated, convert_to_tensor=True)
            cosine = float(st_util.cos_sim(expected_emb, gen_emb).item())
        rouge_l = None
        if self.rouge is not None and generated and expected:
            rouge_l = float(self.rouge.score(expected, generated)["rougeL"].fmeasure)
        flags = _quality_flags(generated, expected_team, predicted_team)
        return {
            "cosine_similarity": cosine,
            "rouge_l_f1": rouge_l,
            "bertscore_f1": None,
            **flags,
        }
 
    async def evaluate_one(self, query_id: str, ordinal: int, total: int) -> Optional[dict]:
        found = self._find_ticket_row(query_id)
        if found is None:
            return {"query_ticket_id": query_id, "skipped": True, "skip_reason": "query_not_found_in_tickets_db"}
        query_pos, row = found
        aliases = self._aliases_for_id(query_id)
        isolated_feedback, removed_rows = self._isolate_feedback(aliases)
 
        title = str(row.get("Title_anon", ""))
        description = str(row.get("Description_anon", ""))
        query_text = _ticket_text(row)
        expected_reply = str(row.get("first_reply", ""))
        expected_team = str(row.get("Team", ""))
        expected_class_raw = str(row.get("label_auto", ""))  # tickets.db's raw subcategory (149 values) -- kept only for logging/debugging
        expected_class = self._class_lookup.get(query_id, "")
        if not expected_class:
            print(f"[{ordinal}/{total}] {query_id}: WARNING no coarse class found in feedback DB lookup, "
                  f"class-based routing will get zero signal for this ticket (raw label_auto='{expected_class_raw[:60]}')")
 
        # using oricle class and team
        word_class = expected_class #word_class, _ = rt.classify_ticket(query_text)
        predicted_team, team_confidence = expected_team, 1.0 #rt.classify_team_with_distilbert(query_text)
 
        base_retrieval, al_retrieval, retrieval_meta = self._retrieve(
            query_text, aliases, word_class, predicted_team, isolated_feedback
        )
        if base_retrieval.empty or al_retrieval.empty:
            return {"query_ticket_id": query_id, "skipped": True, "skip_reason": "no_retrieval_candidates"}
 
        # Snapshot the model that will actually be used for this ticket's generation
        # calls. Captured here (not just once in the run config) so that if an env
        # var or module-level default ever drifts mid-run, every ticket still carries
        # proof of what model config was live at the moment it was generated.
        generation_model_config = getattr(rt, "LLM_MODEL", None)
 
        if self.cfg.retrieval_only:
            base_response = ""
            al_response = ""
        else:
            _gen_t0 = time.time()
            base_response, al_response = await asyncio.gather(
                self._generate(title, description, word_class, predicted_team, team_confidence, base_retrieval),
                self._generate(title, description, word_class, predicted_team, team_confidence, al_retrieval), 
            )
            _gen_elapsed = time.time() - _gen_t0
            if _gen_elapsed > 15:
                print(f"[{ordinal}/{total}] {query_id}: SLOW generation call took {_gen_elapsed:.1f}s "
                      f"(model={generation_model_config})")
            # base_response = await self._generate(title, description, word_class, predicted_team, team_confidence, base_retrieval) ##################izklopil al začasno#################
            
            # al_response = base_response
 
        expected_emb = self.rag.sentence_model.encode(expected_reply, convert_to_tensor=True) if expected_reply else None
        base_metrics = self._response_metrics(base_response, expected_reply, expected_team, predicted_team, expected_emb)
        al_metrics = self._response_metrics(al_response, expected_reply, expected_team, predicted_team, expected_emb)
        judge_base, judge_al = await asyncio.gather(
            self._judge_generated(title, description, expected_reply, "generated_baseline", base_response),
            self._judge_generated(title, description, expected_reply, "generated_al", al_response),
        )
        base_metrics["judge"] = judge_base
        al_metrics["judge"] = judge_al
 
        base_cos = base_metrics["cosine_similarity"]
        al_cos = al_metrics["cosine_similarity"]
        if base_cos is None or al_cos is None:
            metric_msg = "metrics=pending"
        else:
            metric_msg = f"cos base={base_cos:.4f} al={al_cos:.4f} delta={al_cos - base_cos:+.4f}"
        print(f"[{ordinal}/{total}] {query_id}: {metric_msg} gate={retrieval_meta.get('gate_reason')}")
 
        return {
            "query_ticket_id": query_id,
            "query_aliases": sorted(aliases),
            "query_kb_position": int(query_pos),
            "query": {
                "Ref": str(row.get("Ref", "")),
                "sequential_id": str(row.get("sequential_id", "")),
                "title": title,
                "description": description,
                "expected_first_reply": expected_reply,
                "expected_team": expected_team,
                "expected_class": expected_class,
            },
            "isolation": {
                "removed_feedback_rows": removed_rows,
                "remaining_feedback_rows": int(len(isolated_feedback)),
                "excluded_retrieval_aliases": sorted(aliases),
            },
            "classification": {
                "predicted_class": word_class,
                "predicted_team": predicted_team,
                "team_confidence": float(team_confidence),
            },
            "retrieval_metadata": retrieval_meta,
            "baseline": {
                "retrieval_model": "pure_faiss",
                "generation_model_config": generation_model_config,
                "retrieval": base_retrieval.to_dict(orient="records"),
                "response": base_response,
                "metrics": base_metrics,
            },
            "active_learning": {
                "routing_model": asdict(self.cfg.routing),
                "gating_method": asdict(self.cfg.gating),
                "lift_formula": asdict(self.cfg.lift),
                "retrieval_model": "faiss_plus_feedback_lift",
                "generation_model_config": generation_model_config,
                "retrieval": al_retrieval.to_dict(orient="records"),
                "response": al_response,
                "metrics": al_metrics,
            },
        }
 
    def _add_bertscore(self, results: list[dict]) -> None:
        if not self.cfg.calculate_bert or not results:
            return
        try:
            from bert_score import score as bert_score_fn
        except Exception as e:
            print(f"BERTScore unavailable: {e}")
            return
 
        base_cand, base_ref, base_idx = [], [], []
        al_cand, al_ref, al_idx = [], [], []
        for i, r in enumerate(results):
            if r.get("skipped"):
                continue
            expected = r["query"]["expected_first_reply"]
            b = r["baseline"]["response"]
            a = r["active_learning"]["response"]
            if expected and b:
                base_cand.append(b)
                base_ref.append(expected)
                base_idx.append(i)
            if expected and a:
                al_cand.append(a)
                al_ref.append(expected)
                al_idx.append(i)
 
        if base_cand:
            _, _, f1 = bert_score_fn(base_cand, base_ref, lang="en", verbose=False)
            for j, idx in enumerate(base_idx):
                results[idx]["baseline"]["metrics"]["bertscore_f1"] = float(f1[j])
        if al_cand:
            _, _, f1 = bert_score_fn(al_cand, al_ref, lang="en", verbose=False)
            for j, idx in enumerate(al_idx):
                results[idx]["active_learning"]["metrics"]["bertscore_f1"] = float(f1[j])
 
    def _summary(self, results: list[dict], started: float) -> dict:
        valid = [r for r in results if not r.get("skipped")]
        def vals(path: tuple[str, ...]) -> list[Optional[float]]:
            out = []
            for r in valid:
                cur: Any = r
                for p in path:
                    cur = cur.get(p, {}) if isinstance(cur, dict) else {}
                out.append(cur if isinstance(cur, (int, float)) else None)
            return out
 
        base_cos = vals(("baseline", "metrics", "cosine_similarity"))
        al_cos = vals(("active_learning", "metrics", "cosine_similarity"))
        base_rouge = vals(("baseline", "metrics", "rouge_l_f1"))
        al_rouge = vals(("active_learning", "metrics", "rouge_l_f1"))
        base_bert = vals(("baseline", "metrics", "bertscore_f1"))
        al_bert = vals(("active_learning", "metrics", "bertscore_f1"))
 
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "duration_s": time.time() - started,
            "total_queries_requested": len(self.query_ids),
            "total_results": len(results),
            "valid_results": len(valid),
            "skipped": len(results) - len(valid),
            "metrics": {
                "cosine_base_mean": _mean(base_cos),
                "cosine_al_mean": _mean(al_cos),
                "cosine_delta_mean": (_mean(al_cos) or 0.0) - (_mean(base_cos) or 0.0),
                "cosine_base_std": _std(base_cos),
                "cosine_al_std": _std(al_cos),
                "rouge_base_mean": _mean(base_rouge),
                "rouge_al_mean": _mean(al_rouge),
                "rouge_delta_mean": (_mean(al_rouge) or 0.0) - (_mean(base_rouge) or 0.0),
                "bert_base_mean": _mean(base_bert),
                "bert_al_mean": _mean(al_bert),
                "bert_delta_mean": (_mean(al_bert) or 0.0) - (_mean(base_bert) or 0.0),
            },
            "retrieval": {
                "mean_overlap_at_k": _mean([r.get("retrieval_metadata", {}).get("overlap_at_k") for r in valid]),
                "mean_nonzero_lift_candidates": _mean([r.get("retrieval_metadata", {}).get("num_nonzero_lifts_in_candidates") for r in valid]),
                "al_changed_rate": _mean([
                    1.0 if r.get("retrieval_metadata", {}).get("baseline_ids") != r.get("retrieval_metadata", {}).get("al_ids") else 0.0
                    for r in valid
                ]),
                "gate_counts": pd.Series([r.get("retrieval_metadata", {}).get("gate_reason", "unknown") for r in valid]).value_counts().to_dict()
                if valid
                else {},
            },
            "config": asdict(self.cfg),
            "fingerprint": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "tickets_db": _file_fingerprint(self.cfg.tickets_db),
                "feedback_db": _file_fingerprint(self.cfg.feedback_db),
                "openrouter": {
                    "base_url": os.getenv("OPENROUTER_BASE_URL"),
                    "strict_consistency": os.getenv("OPENROUTER_STRICT_CONSISTENCY"),
                    "provider": os.getenv("OPENROUTER_PROVIDER"),
                    "cache": os.getenv("OPENROUTER_ENABLE_CACHE"),
                },
            },
        }
 
    async def run(self) -> dict:
        started = time.time()
        total = len(self.query_ids)
        results: list[Optional[dict]] = [None] * total
        completed = 0
        save_lock = asyncio.Lock()
 
        sem = asyncio.Semaphore(max(1, self.cfg.concurrency))
 
        async def _worker(i: int, qid: str) -> None:
            nonlocal completed
            async with sem:
                try:
                    result = await self.evaluate_one(qid, i, total)
                except Exception as e:
                    result = {"query_ticket_id": qid, "skipped": True, "skip_reason": "exception", "error": repr(e)}
                    print(f"[{i}/{total}] {qid}: ERROR {e}")
                results[i - 1] = result
            async with save_lock:
                completed += 1
                if completed == 1 or completed % 10 == 0:
                    self._save_interim([r for r in results if r is not None])
 
        await asyncio.gather(*(_worker(i, qid) for i, qid in enumerate(self.query_ids, 1)))
        results = [r for r in results if r is not None]
 
        self._add_bertscore(results)
        summary = self._summary(results, started)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"modular_looe_{Path(self.cfg.feedback_db).stem}_{self.cfg.lift.name}_{self.cfg.routing.name}_{self.cfg.gating.name}_{timestamp}"
        details_path = self.results_dir / f"{stem}_details.json"
        summary_path = self.results_dir / f"{stem}_summary.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(results), f, indent=2, ensure_ascii=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
        summary["details_path"] = str(details_path)
        summary["summary_path"] = str(summary_path)
        print(f"Saved details: {details_path}")
        print(f"Saved summary: {summary_path}")
        return summary
 
    def _save_interim(self, results: list[dict]) -> None:
        path = self.results_dir / "modular_looe_interim.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(results), f, indent=2, ensure_ascii=False)
 
 
def build_config(args: argparse.Namespace) -> EvalConfig:
    lift = LiftConfig(
        name=args.lift,
        alpha=args.alpha,
        beta=args.beta,
        multiplier=args.lift_multiplier,
        cap=args.lift_cap,
        sensitivity=args.tanh_sensitivity,
        lcb_k=args.lcb_k,
        positive_only=args.positive_only,
    )
    routing = RoutingConfig(
        name=args.routing,
        w_global=args.w_global,
        w_class=args.w_class,
        w_team=args.w_team,
        relevance_threshold=args.relevance_threshold,
        require_semantic=args.require_semantic,
    )
    gating = GatingConfig(
        name=args.gating,
        faiss_ceiling=args.faiss_ceiling,
        fuzzy_low=args.fuzzy_low,
        fuzzy_high=args.fuzzy_high,
        miracle_tau=args.miracle_tau,
    )
    return EvalConfig(
        tickets_db=args.tickets_db,
        feedback_db=args.feedback_db,
        results_dir=args.results_dir,
        top_k=args.top_k,
        search_k=args.search_k,
        sentence_model=args.sentence_model,
        llm_model=args.llm_model,
        judge_model=args.judge_model,
        lift=lift,
        routing=routing,
        gating=gating,
        calculate_bert=args.bert,
        judge_generated=args.judge_generated,
        retrieval_only=args.retrieval_only,
        limit=args.limit,
        query_offset=args.query_offset,
        concurrency=args.concurrency,
    )
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict modular LOO evaluation for feedback-driven active learning retrieval.")
    parser.add_argument("--tickets-db", default="tickets.db")
    parser.add_argument("--feedback-db", default="comprehensive_feedback_250x250.db")
    parser.add_argument("--results-dir", default="test_results_modular_looe")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-k", type=int, default=100)
    parser.add_argument("--sentence-model", default=os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo-0613"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"))
 
    parser.add_argument("--lift", choices=["laplace", "tanh", "bayesian_lcb"], default="tanh")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lift-multiplier", type=float, default=0.60)
    parser.add_argument("--lift-cap", type=float, default=0.20)
    parser.add_argument("--tanh-sensitivity", type=float, default=5.0)
    parser.add_argument("--lcb-k", type=float, default=1.0)
    parser.add_argument("--positive-only", action="store_true")
 
    parser.add_argument("--routing", choices=["global", "categorical", "semantic", "hybrid", "categorical_intersection"], default="hybrid")
    parser.add_argument("--w-global", type=float, default=0.10)
    parser.add_argument("--w-class", type=float, default=0.55)
    parser.add_argument("--w-team", type=float, default=0.35)
    parser.add_argument("--relevance-threshold", type=float, default=0.70)
    parser.add_argument("--require-semantic", action="store_true")
 
    parser.add_argument("--gating", choices=["none", "static", "fuzzy", "miracle"], default="static")
    parser.add_argument("--faiss-ceiling", type=float, default=0.656)
    parser.add_argument("--fuzzy-low", type=float, default=0.50)
    parser.add_argument("--fuzzy-high", type=float, default=0.75)
    parser.add_argument("--miracle-tau", type=float, default=0.005)
 
    parser.add_argument("--bert", action="store_true", help="Calculate batched BERTScore at the end.")
    parser.add_argument("--judge-generated", action="store_true", help="Use the configured judge LLM on generated baseline and AL responses.")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM generation and response metrics; useful for testing isolation and reranking.")
    parser.add_argument("--concurrency", type=int, default=1,
                         help="Number of tickets processed concurrently. Default 1 preserves old strictly-sequential "
                              "behavior. Raise (e.g. 8-15) to overlap API calls across tickets -- the ticket loop "
                              "has no cross-ticket dependency, so this is safe. Watch provider rate limits.")
    return parser.parse_args()
 
 
if __name__ == "__main__":
    config = build_config(parse_args())
    engine = ModularLOOActiveLearning(config)
    asyncio.run(engine.run())