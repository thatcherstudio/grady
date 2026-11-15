"""Suite and models files: Pydantic schemas and loaders.

Every load error names the file, the case (index and id, or JSONL line number) and the
cause, so a broken suite is fixable from the message alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from grady.scorers import Scorer, ScorerArgError, build_scorer

ID_PATTERN = r"^[A-Za-z0-9_.-]+$"
LABEL_PATTERN = r"^[A-Za-z0-9_.:-]+$"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512


class ConfigError(Exception):
    """A suite or models file is missing or invalid. The message is shown to the user as is."""


# --- Suite -------------------------------------------------------------------------------------


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0, le=2)
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, gt=0)
    system: str | None = None


class ScoreSpec(BaseModel):
    """One score entry: ``{kind: arg}`` in the suite file."""

    model_config = ConfigDict(frozen=True)

    kind: str
    arg: Any

    def build(self) -> Scorer:
        return build_scorer(self.kind, self.arg)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=ID_PATTERN)
    prompt: str = Field(min_length=1)
    system: str | None = None
    temperature: float | None = Field(None, ge=0, le=2)
    max_tokens: int | None = Field(None, gt=0)
    score: list[ScoreSpec] = Field(min_length=1)

    @field_validator("score", mode="before")
    @classmethod
    def _parse_score_entries(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("score: must be a list of score entries")
        specs: list[ScoreSpec] = []
        for i, entry in enumerate(value):
            if isinstance(entry, ScoreSpec):
                specs.append(entry)
                continue
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(
                    f'score[{i}]: each entry must be a one-key mapping like `contains: "text"`'
                )
            ((kind, arg),) = entry.items()
            try:
                build_scorer(kind, arg)
            except ScorerArgError as exc:
                raise ValueError(f"score[{i}]: {exc}") from None
            specs.append(ScoreSpec(kind=kind, arg=arg))
        return specs


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: str = Field(pattern=ID_PATTERN)
    description: str | None = None
    defaults: Defaults = Field(default_factory=Defaults)
    cases: list[Case] = Field(min_length=1)

    def system_for(self, case: Case) -> str | None:
        return case.system if case.system is not None else self.defaults.system

    def temperature_for(self, case: Case) -> float:
        return case.temperature if case.temperature is not None else self.defaults.temperature

    def max_tokens_for(self, case: Case) -> int:
        return case.max_tokens if case.max_tokens is not None else self.defaults.max_tokens


def load_suite(path: Path) -> Suite:
    """Load a ``.yaml``/``.yml`` or ``.jsonl`` suite file."""
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        raw = _read_yaml(path)
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: expected a mapping with 'suite' and 'cases' keys")
        return _suite_from_mapping(path, raw)
    if suffix == ".jsonl":
        return _suite_from_jsonl(path)
    raise ConfigError(
        f"{path}: unsupported suite file type {path.suffix!r} (use .yaml, .yml or .jsonl)"
    )


def _suite_from_mapping(path: Path, raw: Mapping[str, Any]) -> Suite:
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ConfigError(f"{path}: 'cases' must be a non-empty list")
    cases = [_validate_case(path, f"cases[{i}]", item) for i, item in enumerate(raw_cases)]
    suite = _validate_suite(path, {**raw, "cases": cases})
    _check_unique_ids(path, cases, [f"cases[{i}]" for i in range(len(cases))])
    return suite


def _suite_from_jsonl(path: Path) -> Suite:
    cases: list[Case] = []
    where: list[str] = []
    for lineno, line in enumerate(_read_text(path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from None
        cases.append(_validate_case(path, str(lineno), item, sep=":"))
        where.append(f"line {lineno}")
    if not cases:
        raise ConfigError(f"{path}: no cases found")
    suite = _validate_suite(path, {"suite": path.stem, "cases": cases})
    _check_unique_ids(path, cases, where)
    return suite


def _validate_case(path: Path, where: str, item: object, sep: str = ": ") -> Case:
    if not isinstance(item, dict):
        raise ConfigError(f"{path}{sep}{where}: each case must be a mapping")
    label = where
    if isinstance(item.get("id"), str):
        label += f" (id {item['id']!r})"
    try:
        return Case.model_validate(item)
    except ValidationError as exc:
        raise ConfigError(f"{path}{sep}{label}: {format_validation_error(exc)}") from None


def _validate_suite(path: Path, data: Mapping[str, Any]) -> Suite:
    try:
        return Suite.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {format_validation_error(exc)}") from None


def _check_unique_ids(path: Path, cases: Sequence[Case], where: Sequence[str]) -> None:
    seen: dict[str, str] = {}
    for case, loc in zip(cases, where, strict=True):
        if case.id in seen:
            raise ConfigError(f"{path}: duplicate case id {case.id!r} ({seen[case.id]} and {loc})")
        seen[case.id] = loc


# --- Models ------------------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """One endpoint + model. Holds the *name* of the key's env var, never the key itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(pattern=LABEL_PATTERN)
    base_url: str
    model: str = Field(min_length=1)
    api_key_env: str | None = None
    input_price_per_mtok: float | None = Field(None, ge=0)
    output_price_per_mtok: float | None = Field(None, ge=0)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https:// (got {value!r})")
        return value.rstrip("/")


class ModelsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelSpec] = Field(min_length=1)


def load_models(path: Path) -> list[ModelSpec]:
    raw = _read_yaml(path)
    try:
        models = ModelsFile.model_validate(raw).models
    except ValidationError as exc:
        raise ConfigError(f"{path}: {format_validation_error(exc)}") from None
    seen: set[str] = set()
    for model in models:
        if model.label in seen:
            raise ConfigError(f"{path}: duplicate model label {model.label!r}")
        seen.add(model.label)
    return models


def select_models(models: Sequence[ModelSpec], labels: Sequence[str]) -> list[ModelSpec]:
    """Pick models by label, in the order given. No labels means every model."""
    if not labels:
        return list(models)
    by_label = {m.label: m for m in models}
    unknown = [label for label in labels if label not in by_label]
    if unknown:
        known = ", ".join(by_label)
        raise ConfigError(f"unknown model label(s): {', '.join(unknown)} (known: {known})")
    return [by_label[label] for label in dict.fromkeys(labels)]


def resolve_api_keys(
    models: Sequence[ModelSpec], environ: Mapping[str, str]
) -> dict[str, str | None]:
    """Map each model label to its API key, failing before any request if one is missing."""
    keys: dict[str, str | None] = {}
    for model in models:
        if model.api_key_env is None:
            keys[model.label] = None
            continue
        key = environ.get(model.api_key_env, "")
        if not key:
            raise ConfigError(
                f"model {model.label!r} needs an API key in ${model.api_key_env}, which is not set"
            )
        keys[model.label] = key
    return keys


# --- Helpers -----------------------------------------------------------------------------------


def format_validation_error(exc: ValidationError) -> str:
    """Compact, one-line rendering of a Pydantic error.

    Messages from Grady's own validators already say where they apply, so they're shown
    without the field path; Pydantic's built-in errors get ``field.path: message``.
    """
    parts = []
    for err in exc.errors():
        msg = str(err["msg"])
        if err["type"] == "value_error":
            parts.append(msg.removeprefix("Value error, "))
            continue
        loc = ".".join(str(p) for p in err["loc"])
        if err["type"] == "extra_forbidden":
            msg = "unknown key"
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: cannot read file: {exc}") from None


def _read_yaml(path: Path) -> object:
    text = _read_text(path)
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None
