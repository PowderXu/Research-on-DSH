"""Microsoft SkillOpt environment adapter that delegates every episode to DSH."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter

from .dataloader import GitHubDocsDshDataLoader
from .rollout import DshCommandConfig, DshCommandRunner, Runner, run_batch
from .scorer import GitHubDocsSourceResolver
from .validation import load_leakage_markers


class GitHubDocsDshAdapter(EnvAdapter):
    """Optimize one arm-specific DSH retrieval skill against frozen qrels."""

    def __init__(
        self,
        split_dir: str = "",
        split_mode: str = "split_dir",
        split_seed: int = 20260821,
        seed: int = 20260821,
        limit: int = 0,
        arm: str = "hybrid",
        corpus_path: str = "evaluation/github_docs_v2/corpus.jsonl",
        dsh_binary: str = "node_modules/.bin/dsh",
        dsh_home: str = "dsh_home",
        workspace: str = "data/evaluation_raw/github-docs-corpus/content",
        model_patch: str = "dsh-techdocs-plugin/trial-openai.patch.yml",
        common_patch: str = "evaluation/harness/github_docs_dsh_common.patch.yml",
        arm_patch: str = "",
        exec_timeout: int = 180,
        workers: int = 1,
        start_backend: bool = True,
        backend_host: str = "127.0.0.1",
        backend_port: int = 0,
        cache_dir: str = "data/evaluation_cache/github_docs_v1",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        local_files_only: bool = True,
        neo4j_uri: str = "",
        neo4j_username: str = "",
        neo4j_database: str = "",
        analyst_workers: int = 1,
        failure_only: bool = False,
        minibatch_size: int = 4,
        edit_budget: int = 2,
        eval_item_ids: str = "",
        runner: Runner | None = None,
    ) -> None:
        if arm not in {"fs", "hybrid", "neo4j"}:
            raise ValueError(f"unknown GitHub Docs DSH arm: {arm}")
        if workers != 1:
            raise ValueError(
                "GitHubDocsDshAdapter currently requires workers=1 so DSH session "
                "attribution and local service traces remain deterministic"
            )
        self.arm = arm
        self.exec_timeout = int(exec_timeout)
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.eval_item_ids = tuple(
            value.strip()
            for value in str(eval_item_ids or "").split(",")
            if value.strip()
        )
        self.start_backend = bool(start_backend)
        self.backend_host = backend_host
        self.backend_port = int(backend_port or (1935 if arm == "hybrid" else 1936))
        self.cache_dir = Path(cache_dir)
        self.embedding_model = embedding_model
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_database = neo4j_database
        self.dataset_dir = Path(corpus_path).resolve().parent
        self._backend_host = None
        self._backend_service = None
        self.split_dir = Path(split_dir).resolve()
        self.dataloader = GitHubDocsDshDataLoader(
            split_dir=str(self.split_dir),
            split_mode=split_mode,
            split_seed=split_seed,
            seed=seed,
            limit=limit,
        )
        self.resolver = GitHubDocsSourceResolver(Path(corpus_path))
        self.leakage_markers = load_leakage_markers(self.split_dir)
        self.expected_name = f"github-docs-{arm}"
        self.runner = runner
        self.command_config = DshCommandConfig(
            arm=arm,
            dsh_binary=Path(dsh_binary),
            dsh_home=Path(dsh_home),
            workspace=Path(workspace),
            model_patch=Path(model_patch),
            common_patch=Path(common_patch),
            arm_patch=Path(
                arm_patch
                or f"evaluation/harness/github_docs_{arm}_system.patch.yml"
            ),
            timeout_seconds=self.exec_timeout,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)
        if self.eval_item_ids:
            requested = set(self.eval_item_ids)
            available = {
                str(item.get("id"))
                for items in self.dataloader._splits.values()
                for item in items
            }
            missing = sorted(requested - available)
            if missing:
                raise ValueError(
                    "eval_item_ids are absent from the frozen split: "
                    + ", ".join(missing)
                )
            for split, items in self.dataloader._splits.items():
                self.dataloader._splits[split] = [
                    item for item in items if str(item.get("id")) in requested
                ]
        if self.runner is None:
            trace_provider = None
            if self.arm in {"hybrid", "neo4j"} and self.start_backend:
                from kbbench.github_docs_plugin_service import (
                    GitHubDocsPluginService,
                    GitHubDocsServiceHost,
                )

                self._backend_service = GitHubDocsPluginService(
                    dataset_dir=self.dataset_dir,
                    cache_dir=self.cache_dir,
                    arm=self.arm,
                    embedding_model_name=self.embedding_model,
                    device=self.device,
                    local_files_only=self.local_files_only,
                    neo4j_uri=self.neo4j_uri
                    or os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
                    neo4j_username=self.neo4j_username
                    or os.environ.get("NEO4J_USERNAME", "neo4j"),
                    neo4j_password=os.environ.get("NEO4J_PASSWORD", "secretgraph"),
                    neo4j_database=self.neo4j_database
                    or os.environ.get("NEO4J_DATABASE", "neo4j"),
                )
                self._backend_host = GitHubDocsServiceHost(
                    self._backend_service, self.backend_host, self.backend_port
                )
                self._backend_host.start()
                trace_provider = lambda: list(self._backend_service.events)
            self.runner = DshCommandRunner(
                self.command_config, trace_provider=trace_provider
            )

    def get_dataloader(self) -> GitHubDocsDshDataLoader:
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs: Any) -> list[dict[str, Any]]:
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs: Any):
        batch = self.dataloader.build_train_batch(
            batch_size=batch_size, seed=seed, **kwargs
        )
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs: Any):
        batch = self.dataloader.build_eval_batch(
            env_num=env_num, split=split, seed=seed, **kwargs
        )
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self.runner is None:
            raise RuntimeError("GitHubDocsDshAdapter.setup() must run before rollout")
        return run_batch(
            items=list(env_manager),
            skill_content=skill_content,
            out_root=out_dir,
            arm=self.arm,
            resolver=self.resolver,
            runner=self.runner,
            expected_name=self.expected_name,
            leakage_markers=self.leakage_markers,
        )

    def build_reference_text(self, item: dict) -> str:
        # Keep qrel identities out of reflection. Per-rollout metric feedback is
        # attached by rollout.run_batch after scoring.
        return (
            f"Evidence category: {item.get('task_type', 'unknown')}; "
            f"evidence structure: {item.get('evidence_structure', 'unknown')}; "
            f"qrel count: {item.get('qrel_count', 'unknown')}."
        )

    def get_task_types(self) -> list[str]:
        return [
            "single_page_direct",
            "single_page_section",
            "single_page_variant_anchor",
            "single_page_composed_anchor",
            "multi_page_dispersed",
            "multi_page_linked",
        ]
