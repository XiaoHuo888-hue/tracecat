"""Cross-workspace reference resolution for synced documents.

Synced documents reference workspace-local targets (model catalog entries, MCP
integrations) that never sync themselves. This module owns the single
resolution ladder for those references: keep a locally-valid binding, accept a
validated explicit selection, auto-resolve a unique natural-key match, or
require a choice. Every binding is verified against live workspace state
before use.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.orm import load_only, raiseload

from tracecat.agent.catalog.service import AgentCatalogService
from tracecat.agent.catalog.types import ModelKey
from tracecat.db.models import MCPIntegration, WorkspaceSyncResourceMapping
from tracecat.dsl.enums import PlatformAction
from tracecat.sync import (
    CatalogMappingAffectedPreset,
    CatalogMappingAffectedWorkflow,
    CatalogMappingCandidate,
    CatalogMappingRequirement,
    CatalogMappingRequirementReason,
    McpIntegrationMappingAffectedPreset,
    McpIntegrationMappingAffectedWorkflow,
    McpIntegrationMappingCandidate,
    McpIntegrationMappingRequirement,
    McpIntegrationMappingRequirementReason,
    PullDiagnostic,
)
from tracecat.workspace_sync.adapters.base import SyncMappingService
from tracecat.workspace_sync.enums import ReferenceKind
from tracecat.workspace_sync.schemas import (
    AGENT_PRESET_ROOT,
    AgentPresetResourceSpec,
    AgentPresetVersionResourceSpec,
    McpIntegrationRefMeta,
    WorkflowResourceSpec,
)
from tracecat.workspace_sync.types import (
    AgentPresetCatalogReference,
    AgentPresetMcpIntegrationReference,
    CatalogReference,
    CorrelatedAgentPresets,
    CorrelatedMcpIntegrationRefs,
    McpIntegrationReference,
    WorkflowCatalogReference,
    WorkflowMcpIntegrationReference,
)
from tracecat.workspace_sync.workflow import workflow_source_path

AGENT_PRESET_VERSIONS_DIR = "versions"


def _version_source_path(source_id: str, version: int) -> str:
    """Return the repository path for an agent preset version manifest."""
    return f"{AGENT_PRESET_ROOT}/{source_id}/{AGENT_PRESET_VERSIONS_DIR}/{version}.yml"


def _parse_uuid(value: object) -> uuid.UUID | None:
    """Parse a reference into a UUID; non-UUID refs are left uncorrelated."""
    try:
        return uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError):
        return None


async def correlate_catalog_ids(
    workspace_service: SyncMappingService,
    presets: dict[str, AgentPresetResourceSpec],
    workflows: dict[str, WorkflowResourceSpec] | None = None,
    *,
    requested_catalog_mappings: Mapping[uuid.UUID, uuid.UUID] | None = None,
) -> CorrelatedAgentPresets:
    """Re-map preset and workflow catalog UUIDs to enabled local rows.

    Catalog UUIDs are deployment-local. Preserve an incoming UUID when it is
    already enabled in this workspace and its model tuple matches the local
    row. Otherwise, accept a valid per-pull selection or use the only matching
    local candidate. Multiple candidates require an explicit selection rather
    than an arbitrary UUID tie-break.

    When remapping a catalog UUID, clear the snapshot's deployment-local
    ``base_url`` in both preset versions and workflow agent actions so the
    target's catalog credentials are never routed to the source deployment's
    endpoint. Unresolvable selections produce diagnostics before import writes
    begin instead of reaching a foreign-key violation.
    """
    requested_catalog_mappings = requested_catalog_mappings or {}
    workflows = workflows or {}
    catalog_service = AgentCatalogService(session=workspace_service.session)
    references_by_catalog_id: dict[uuid.UUID, list[CatalogReference]] = {}
    present_catalog_ids: set[uuid.UUID] = set()
    diagnostics: list[PullDiagnostic] = []
    invalid_catalog_ids: set[uuid.UUID] = set()
    missing_tuple_diagnostics: list[tuple[uuid.UUID, PullDiagnostic]] = []

    for source_id, preset in sorted(presets.items()):
        for version_number, version in sorted(preset.versions.items()):
            catalog_id = version.catalog_id
            if catalog_id is None:
                continue
            present_catalog_ids.add(catalog_id)
            if not version.model_provider or not version.model_name:
                # The tuple is only needed to correlate a non-local UUID.
                # Defer the diagnostic until local enablement is known.
                missing_tuple_diagnostics.append(
                    (
                        catalog_id,
                        PullDiagnostic(
                            workflow_path=_version_source_path(
                                source_id, version_number
                            ),
                            workflow_title=preset.name,
                            error_type="validation",
                            message=(
                                f"Agent preset {preset.slug!r} version "
                                f"{version_number} references a non-local "
                                "model catalog entry but does not include "
                                "model_provider and model_name for "
                                "correlation."
                            ),
                            details={
                                "preset_slug": preset.slug,
                                "preset_version": version_number,
                                "catalog_id": str(catalog_id),
                            },
                        ),
                    )
                )
                continue
            references_by_catalog_id.setdefault(catalog_id, []).append(
                AgentPresetCatalogReference(
                    path=_version_source_path(source_id, version_number),
                    preset_slug=preset.slug,
                    preset_name=preset.name,
                    version_number=version_number,
                    model_key=ModelKey(
                        version.model_provider,
                        version.model_name,
                    ),
                )
            )

    workflow_action_types = (PlatformAction.AI_AGENT, PlatformAction.AI_ACTION)
    for source_id, workflow in sorted(workflows.items()):
        for action in workflow.definition.actions:
            if action.action not in workflow_action_types:
                continue
            nested_model = action.args.get("model")
            nested_model = nested_model if isinstance(nested_model, dict) else None
            merged_args = (
                {**action.args, **nested_model}
                if nested_model is not None
                else action.args
            )
            raw_catalog_id = merged_args.get("catalog_id")
            model_provider = merged_args.get("model_provider")
            model_name = merged_args.get("model_name")
            if raw_catalog_id is None or not model_provider or not model_name:
                continue
            try:
                catalog_id = uuid.UUID(str(raw_catalog_id))
            except (TypeError, ValueError):
                # Not a literal UUID (e.g. a template expression evaluated
                # at runtime): leave the action untouched.
                continue
            present_catalog_ids.add(catalog_id)
            references_by_catalog_id.setdefault(catalog_id, []).append(
                WorkflowCatalogReference(
                    path=workflow_source_path(source_id),
                    workflow_source_id=source_id,
                    workflow_title=workflow.definition.title,
                    action_ref=action.ref,
                    model_key=ModelKey(
                        str(model_provider),
                        str(model_name),
                    ),
                )
            )

    enabled_catalog_models = await catalog_service.enabled_catalog_models(
        org_id=workspace_service.organization_id,
        workspace_id=workspace_service.workspace_id,
        catalog_ids=present_catalog_ids,
    )

    for catalog_id, diagnostic in missing_tuple_diagnostics:
        if catalog_id in enabled_catalog_models:
            continue
        diagnostics.append(diagnostic)
        invalid_catalog_ids.add(catalog_id)

    sorted_references_by_catalog_id = sorted(references_by_catalog_id.items())
    for catalog_id, references in sorted_references_by_catalog_id:
        model_keys = {reference.model_key for reference in references}
        if len(model_keys) <= 1:
            continue
        first = references[0]
        diagnostics.append(
            PullDiagnostic(
                workflow_path=first.path,
                workflow_title=_catalog_reference_title(first),
                error_type="validation",
                message=(
                    f"Source catalog entry {catalog_id} is used with conflicting "
                    "model_provider and model_name values. A source catalog UUID "
                    "must identify exactly one model."
                ),
                details={
                    "code": "catalog_model_identity_conflict",
                    "catalog_id": str(catalog_id),
                    "models": [
                        {
                            "model_provider": model_key.model_provider,
                            "model_name": model_key.model_name,
                        }
                        for model_key in sorted(model_keys)
                    ],
                },
            )
        )
        invalid_catalog_ids.add(catalog_id)

    for catalog_id, references in sorted_references_by_catalog_id:
        if catalog_id in invalid_catalog_ids:
            continue
        local_model_key = enabled_catalog_models.get(catalog_id)
        if local_model_key is None:
            continue
        manifest_model_key = references[0].model_key
        if manifest_model_key == local_model_key:
            continue
        first = references[0]
        diagnostics.append(
            PullDiagnostic(
                workflow_path=first.path,
                workflow_title=_catalog_reference_title(first),
                error_type="validation",
                message=(
                    f"Enabled local catalog entry {catalog_id} identifies "
                    f"{local_model_key.model_provider!r} / "
                    f"{local_model_key.model_name!r}, but the repository "
                    f"manifest identifies {manifest_model_key.model_provider!r} / "
                    f"{manifest_model_key.model_name!r}."
                ),
                details={
                    "code": "catalog_model_identity_mismatch",
                    "catalog_id": str(catalog_id),
                    "manifest_model": {
                        "model_provider": manifest_model_key.model_provider,
                        "model_name": manifest_model_key.model_name,
                    },
                    "local_model": {
                        "model_provider": local_model_key.model_provider,
                        "model_name": local_model_key.model_name,
                    },
                },
            )
        )
        invalid_catalog_ids.add(catalog_id)

    models = {
        references[0].model_key
        for catalog_id, references in references_by_catalog_id.items()
        if catalog_id not in invalid_catalog_ids
        and catalog_id not in enabled_catalog_models
    }
    candidates_by_model = await catalog_service.catalog_candidates_by_models(
        org_id=workspace_service.organization_id,
        workspace_id=workspace_service.workspace_id,
        models=models,
    )
    resolved_catalog_ids: dict[uuid.UUID, uuid.UUID] = {}
    requirements: list[CatalogMappingRequirement] = []

    for catalog_id, references in sorted_references_by_catalog_id:
        if catalog_id in invalid_catalog_ids:
            continue
        if catalog_id in enabled_catalog_models:
            continue
        model_key = references[0].model_key
        candidates = candidates_by_model.get(model_key, [])
        requested_target = requested_catalog_mappings.get(catalog_id)

        if requested_target is not None:
            candidate_ids = {candidate.catalog_id for candidate in candidates}
            if requested_target in candidate_ids:
                resolved_catalog_ids[catalog_id] = requested_target
                continue
            _append_catalog_mapping_requirement(
                requirements=requirements,
                diagnostics=diagnostics,
                catalog_id=catalog_id,
                model_key=model_key,
                references=references,
                candidates=candidates,
                reason="invalid_selection",
            )
            continue

        if len(candidates) == 1:
            resolved_catalog_ids[catalog_id] = candidates[0].catalog_id
            continue
        if len(candidates) > 1:
            _append_catalog_mapping_requirement(
                requirements=requirements,
                diagnostics=diagnostics,
                catalog_id=catalog_id,
                model_key=model_key,
                references=references,
                candidates=candidates,
                reason="ambiguous",
            )
            continue

        _append_unavailable_catalog_diagnostics(
            diagnostics=diagnostics,
            catalog_id=catalog_id,
            references=references,
        )

    unused_requested_catalog_ids = set(requested_catalog_mappings) - present_catalog_ids
    for catalog_id in sorted(unused_requested_catalog_ids):
        diagnostics.append(
            PullDiagnostic(
                workflow_path="",
                workflow_title=None,
                error_type="validation",
                message=(
                    f"Catalog mapping selection for source {catalog_id} does not "
                    "appear in this repository snapshot."
                ),
                details={
                    "catalog_id": str(catalog_id),
                    "code": "catalog_mapping_source_not_found",
                },
            )
        )

    if not resolved_catalog_ids:
        return CorrelatedAgentPresets(
            presets=presets,
            workflows=workflows,
            diagnostics=diagnostics,
            requirements=requirements,
        )

    correlated_presets: dict[str, AgentPresetResourceSpec] = {}
    for source_id, preset in sorted(presets.items()):
        correlated_versions: dict[int, AgentPresetVersionResourceSpec] = {}
        for version_number, version in sorted(preset.versions.items()):
            local_catalog_id = (
                resolved_catalog_ids.get(version.catalog_id)
                if version.catalog_id is not None
                else None
            )
            correlated_versions[version_number] = (
                version
                if local_catalog_id is None
                else version.model_copy(
                    update={"catalog_id": local_catalog_id, "base_url": None}
                )
            )

        correlated_presets[source_id] = preset.model_copy(
            update={"versions": correlated_versions}
        )

    correlated_workflows: dict[str, WorkflowResourceSpec] = {}
    for source_id, workflow in sorted(workflows.items()):
        correlated_action_specs = list(workflow.definition.actions)
        workflow_rewritten = False
        for index, action in enumerate(workflow.definition.actions):
            if action.action not in (
                PlatformAction.AI_AGENT,
                PlatformAction.AI_ACTION,
            ):
                continue
            nested_model = action.args.get("model")
            nested_model = nested_model if isinstance(nested_model, dict) else None
            merged_args = (
                {**action.args, **nested_model}
                if nested_model is not None
                else action.args
            )
            raw_catalog_id = merged_args.get("catalog_id")
            if (
                not raw_catalog_id
                or not merged_args.get("model_provider")
                or not merged_args.get("model_name")
            ):
                continue
            try:
                catalog_id = uuid.UUID(str(raw_catalog_id))
            except (TypeError, ValueError):
                continue
            local_catalog_id = resolved_catalog_ids.get(catalog_id)
            if local_catalog_id is None:
                continue

            new_catalog_id = str(local_catalog_id)
            new_args = dict(action.args)
            new_args["catalog_id"] = new_catalog_id
            if "base_url" in new_args:
                new_args["base_url"] = None
            if nested_model is not None:
                new_model = dict(nested_model)
                if "catalog_id" in new_model:
                    new_model["catalog_id"] = new_catalog_id
                if "base_url" in new_model:
                    new_model["base_url"] = None
                new_args["model"] = new_model
            correlated_action_specs[index] = action.model_copy(
                update={"args": new_args}
            )
            workflow_rewritten = True

        correlated_workflows[source_id] = (
            workflow
            if not workflow_rewritten
            else workflow.model_copy(
                update={
                    "definition": workflow.definition.model_copy(
                        update={"actions": correlated_action_specs}
                    )
                }
            )
        )

    return CorrelatedAgentPresets(
        presets=correlated_presets,
        workflows=correlated_workflows,
        diagnostics=diagnostics,
        requirements=requirements,
    )


def _append_catalog_mapping_requirement(
    *,
    requirements: list[CatalogMappingRequirement],
    diagnostics: list[PullDiagnostic],
    catalog_id: uuid.UUID,
    model_key: ModelKey,
    references: list[CatalogReference],
    candidates: list[CatalogMappingCandidate],
    reason: CatalogMappingRequirementReason,
) -> None:
    """Append one grouped mapping requirement and its blocking diagnostic."""
    if not candidates:
        _append_unavailable_catalog_diagnostics(
            diagnostics=diagnostics,
            catalog_id=catalog_id,
            references=references,
        )
        return

    if reason == "ambiguous":
        message = (
            f"Model {model_key.model_provider!r} / {model_key.model_name!r} "
            f"matches {len(candidates)} enabled target catalogs. Choose the "
            "target model before applying this pull."
        )
    else:
        message = (
            f"The selected target is not an enabled match for model "
            f"{model_key.model_provider!r} / {model_key.model_name!r}. "
            "Choose an available target before applying this pull."
        )
    first = references[0]
    diagnostics.append(
        PullDiagnostic(
            workflow_path=first.path,
            workflow_title=_catalog_reference_title(first),
            error_type="dependency",
            message=message,
            details={
                "code": "catalog_mapping_required",
                "catalog_id": str(catalog_id),
                "model_provider": model_key.model_provider,
                "model_name": model_key.model_name,
                "reason": reason,
            },
        )
    )
    requirements.append(
        CatalogMappingRequirement(
            source_catalog_id=catalog_id,
            model_provider=model_key.model_provider,
            model_name=model_key.model_name,
            reason=reason,
            message=message,
            candidates=list(candidates),
            affected_presets=[
                CatalogMappingAffectedPreset(
                    preset_slug=reference.preset_slug,
                    preset_name=reference.preset_name,
                    version=reference.version_number,
                    path=reference.path,
                )
                for reference in references
                if isinstance(reference, AgentPresetCatalogReference)
            ],
            affected_workflows=[
                CatalogMappingAffectedWorkflow(
                    workflow_source_id=reference.workflow_source_id,
                    workflow_path=reference.path,
                    workflow_title=reference.workflow_title,
                    action_ref=reference.action_ref,
                )
                for reference in references
                if isinstance(reference, WorkflowCatalogReference)
            ],
        )
    )


def _append_unavailable_catalog_diagnostics(
    *,
    diagnostics: list[PullDiagnostic],
    catalog_id: uuid.UUID,
    references: list[CatalogReference],
) -> None:
    """Append per-reference diagnostics when no candidate is available."""
    for reference in references:
        if isinstance(reference, AgentPresetCatalogReference):
            message = (
                f"Agent preset {reference.preset_slug!r} version "
                f"{reference.version_number} requires model "
                f"{reference.model_key.model_provider!r} / "
                f"{reference.model_key.model_name!r}, but no matching "
                "enabled model is configured for this workspace."
            )
            details = {
                "preset_slug": reference.preset_slug,
                "preset_version": reference.version_number,
                "catalog_id": str(catalog_id),
                "model_provider": reference.model_key.model_provider,
                "model_name": reference.model_key.model_name,
            }
        else:
            message = (
                f"Workflow {reference.workflow_title!r} action "
                f"{reference.action_ref!r} requires model "
                f"{reference.model_key.model_provider!r} / "
                f"{reference.model_key.model_name!r}, but no matching "
                "enabled model is configured for this workspace."
            )
            details = {
                "workflow_source_id": reference.workflow_source_id,
                "action_ref": reference.action_ref,
                "catalog_id": str(catalog_id),
                "model_provider": reference.model_key.model_provider,
                "model_name": reference.model_key.model_name,
            }
        diagnostics.append(
            PullDiagnostic(
                workflow_path=reference.path,
                workflow_title=_catalog_reference_title(reference),
                error_type="dependency",
                message=message,
                details=details,
            )
        )


def _catalog_reference_title(reference: CatalogReference) -> str:
    """Return the owning preset or workflow title for a catalog reference."""
    if isinstance(reference, AgentPresetCatalogReference):
        return reference.preset_name
    return reference.workflow_title


async def correlate_mcp_integration_refs(
    workspace_service: SyncMappingService,
    presets: dict[str, AgentPresetResourceSpec],
    workflows: dict[str, WorkflowResourceSpec] | None = None,
    *,
    requested_mcp_integration_mappings: Mapping[uuid.UUID, uuid.UUID] | None = None,
) -> CorrelatedMcpIntegrationRefs:
    """Re-map preset and workflow MCP integration UUIDs to local rows.

    MCP integration UUIDs are workspace-local, and the integrations themselves
    never sync. A persisted sync mapping resolves silently; otherwise an exact
    slug/server_type/auth_type match against the manifest's correlation hint
    auto-resolves without persisting. Anything else requires an explicit
    selection, which is persisted so later pulls resolve silently.
    """
    requested_mappings = requested_mcp_integration_mappings or {}
    workflows = workflows or {}
    diagnostics: list[PullDiagnostic] = []
    references: dict[uuid.UUID, list[McpIntegrationReference]] = {}
    meta_by_source_id: dict[uuid.UUID, McpIntegrationRefMeta] = {}
    conflicting_meta_ids: set[uuid.UUID] = set()

    for source_id, preset in sorted(presets.items()):
        for version_number, version in sorted(preset.versions.items()):
            version_meta = version.mcp_integration_meta or {}
            for ref in version.mcp_integrations:
                integration_id = _parse_uuid(ref)
                if integration_id is None:
                    continue
                meta = version_meta.get(ref)
                if meta is not None:
                    known = meta_by_source_id.get(integration_id)
                    if known is None:
                        meta_by_source_id[integration_id] = meta
                    elif known != meta:
                        # One source UUID must identify one integration.
                        conflicting_meta_ids.add(integration_id)
                references.setdefault(integration_id, []).append(
                    AgentPresetMcpIntegrationReference(
                        path=_version_source_path(source_id, version_number),
                        preset_slug=preset.slug,
                        preset_name=preset.name,
                        version_number=version_number,
                        meta=meta,
                    )
                )

    workflow_action_types = (PlatformAction.AI_AGENT, PlatformAction.AI_ACTION)
    for source_id, workflow in sorted(workflows.items()):
        for action in workflow.definition.actions:
            if action.action not in workflow_action_types:
                continue
            raw_refs = action.args.get("mcp_integrations")
            if not isinstance(raw_refs, list):
                continue
            for ref in raw_refs:
                integration_id = _parse_uuid(ref)
                if integration_id is None:
                    continue
                references.setdefault(integration_id, []).append(
                    WorkflowMcpIntegrationReference(
                        path=workflow_source_path(source_id),
                        workflow_source_id=source_id,
                        workflow_title=workflow.definition.title,
                        action_ref=action.ref,
                    )
                )

    present_source_ids = set(references)
    unused_requested = set(requested_mappings) - present_source_ids
    for source_integration_id in sorted(unused_requested):
        diagnostics.append(
            PullDiagnostic(
                workflow_path="",
                workflow_title=None,
                error_type="validation",
                message=(
                    f"MCP integration mapping selection for source "
                    f"{source_integration_id} does not appear in this "
                    "repository snapshot."
                ),
                details={
                    "code": "mcp_integration_mapping_source_not_found",
                    "mcp_integration_id": str(source_integration_id),
                },
            )
        )

    if not references:
        return CorrelatedMcpIntegrationRefs(
            presets=presets,
            workflows=workflows,
            diagnostics=diagnostics,
            requirements=[],
            resolved={},
            persist={},
        )

    existing = await _mcp_integration_mappings(workspace_service, present_source_ids)
    local_integrations = await _local_mcp_integrations(workspace_service)
    by_natural_key: dict[tuple[str, str, str], list[MCPIntegration]] = {}
    for row in local_integrations:
        key = (row.slug, row.server_type, str(row.auth_type))
        by_natural_key.setdefault(key, []).append(row)

    resolved: dict[uuid.UUID, uuid.UUID] = {}
    persist: dict[uuid.UUID, uuid.UUID] = {}
    requirements: list[McpIntegrationMappingRequirement] = []
    local_ids = {row.id for row in local_integrations}

    for source_integration_id, refs in sorted(references.items()):
        requested_target = requested_mappings.get(source_integration_id)
        meta = meta_by_source_id.get(source_integration_id)
        if requested_target is not None:
            if requested_target in local_ids:
                resolved[source_integration_id] = requested_target
                persist[source_integration_id] = requested_target
                continue
            _append_mcp_integration_requirement(
                requirements=requirements,
                diagnostics=diagnostics,
                source_integration_id=source_integration_id,
                meta=meta,
                references=refs,
                local_integrations=local_integrations,
                reason="invalid_selection",
            )
            continue

        mapped = existing.get(source_integration_id)
        if mapped is not None:
            if mapped in local_ids:
                resolved[source_integration_id] = mapped
                continue
            # Stale mapping target; fall through to re-resolution.

        if source_integration_id in conflicting_meta_ids:
            # Neither copy of the hint can be trusted; require a choice.
            _append_mcp_integration_requirement(
                requirements=requirements,
                diagnostics=diagnostics,
                source_integration_id=source_integration_id,
                meta=None,
                references=refs,
                local_integrations=local_integrations,
                reason="conflicting_metadata",
            )
            continue

        if meta is not None:
            matches = by_natural_key.get(
                (meta.slug, meta.server_type, meta.auth_type), []
            )
            if len(matches) == 1:
                # Deterministic natural-key hit; intentionally not persisted.
                resolved[source_integration_id] = matches[0].id
                continue

        _append_mcp_integration_requirement(
            requirements=requirements,
            diagnostics=diagnostics,
            source_integration_id=source_integration_id,
            meta=meta,
            references=refs,
            local_integrations=local_integrations,
            reason="unresolved",
        )

    if not resolved:
        return CorrelatedMcpIntegrationRefs(
            presets=presets,
            workflows=workflows,
            diagnostics=diagnostics,
            requirements=requirements,
            resolved={},
            persist=persist,
        )

    local_meta_by_id = {
        row.id: McpIntegrationRefMeta(
            slug=row.slug,
            server_type=row.server_type,
            auth_type=str(row.auth_type),
            name=row.name,
        )
        for row in local_integrations
    }
    correlated_presets: dict[str, AgentPresetResourceSpec] = {}
    for source_id, preset in sorted(presets.items()):
        correlated_versions: dict[int, AgentPresetVersionResourceSpec] = {}
        for version_number, version in sorted(preset.versions.items()):
            rewritten = [
                str(resolved[parsed])
                if (parsed := _parse_uuid(ref)) is not None and parsed in resolved
                else ref
                for ref in version.mcp_integrations
            ]
            # Rewrite hint keys and values with the resolved rows so repeat
            # previews converge on the local projection instead of diffing on
            # source-workspace metadata forever.
            rewritten_meta: dict[str, McpIntegrationRefMeta] | None = None
            if version.mcp_integration_meta is not None:
                rewritten_meta = {}
                for ref, ref_meta in version.mcp_integration_meta.items():
                    parsed = _parse_uuid(ref)
                    if parsed is not None and parsed in resolved:
                        target = resolved[parsed]
                        rewritten_meta[str(target)] = local_meta_by_id.get(
                            target, ref_meta
                        )
                    else:
                        rewritten_meta[ref] = ref_meta
                rewritten_meta = dict(sorted(rewritten_meta.items()))
            correlated_versions[version_number] = (
                version
                if rewritten == version.mcp_integrations
                and rewritten_meta == version.mcp_integration_meta
                else version.model_copy(
                    update={
                        "mcp_integrations": rewritten,
                        "mcp_integration_meta": rewritten_meta,
                    }
                )
            )
        correlated_presets[source_id] = preset.model_copy(
            update={"versions": correlated_versions}
        )

    correlated_workflows: dict[str, WorkflowResourceSpec] = {}
    for source_id, workflow in sorted(workflows.items()):
        action_specs = list(workflow.definition.actions)
        rewritten_workflow = False
        for index, action in enumerate(workflow.definition.actions):
            if action.action not in workflow_action_types:
                continue
            raw_refs = action.args.get("mcp_integrations")
            if not isinstance(raw_refs, list):
                continue
            rewritten_refs = [
                str(resolved[parsed])
                if (parsed := _parse_uuid(ref)) is not None and parsed in resolved
                else ref
                for ref in raw_refs
            ]
            if rewritten_refs == raw_refs:
                continue
            new_args = dict(action.args)
            new_args["mcp_integrations"] = rewritten_refs
            action_specs[index] = action.model_copy(update={"args": new_args})
            rewritten_workflow = True

        correlated_workflows[source_id] = (
            workflow
            if not rewritten_workflow
            else workflow.model_copy(
                update={
                    "definition": workflow.definition.model_copy(
                        update={"actions": action_specs}
                    )
                }
            )
        )

    return CorrelatedMcpIntegrationRefs(
        presets=correlated_presets,
        workflows=correlated_workflows,
        diagnostics=diagnostics,
        requirements=requirements,
        resolved=resolved,
        persist=persist,
    )


async def _local_mcp_integrations(
    workspace_service: SyncMappingService,
) -> list[MCPIntegration]:
    """Return the workspace's MCP integrations ordered by slug.

    Loads only the correlation metadata columns; encrypted credential columns
    and the OAuth relationship must never enter this path.
    """
    stmt = (
        select(MCPIntegration)
        .options(
            load_only(
                MCPIntegration.id,
                MCPIntegration.slug,
                MCPIntegration.name,
                MCPIntegration.server_type,
                MCPIntegration.auth_type,
            ),
            raiseload(MCPIntegration.oauth_integration),
        )
        .where(MCPIntegration.workspace_id == workspace_service.workspace_id)
        .order_by(MCPIntegration.slug.asc())
    )
    return list((await workspace_service.session.scalars(stmt)).all())


async def _mcp_integration_mappings(
    workspace_service: SyncMappingService,
    source_ids: set[uuid.UUID],
) -> dict[uuid.UUID, uuid.UUID]:
    """Return persisted source-to-local MCP integration mappings."""
    if not source_ids:
        return {}
    stmt = select(WorkspaceSyncResourceMapping).where(
        WorkspaceSyncResourceMapping.workspace_id == workspace_service.workspace_id,
        WorkspaceSyncResourceMapping.provider
        == workspace_service._mapping_provider_value,
        WorkspaceSyncResourceMapping.resource_type
        == ReferenceKind.MCP_INTEGRATION.value,
        WorkspaceSyncResourceMapping.source_id.in_(
            {str(source_id) for source_id in source_ids}
        ),
    )
    mappings: dict[uuid.UUID, uuid.UUID] = {}
    for row in (await workspace_service.session.scalars(stmt)).all():
        source_uuid = _parse_uuid(row.source_id)
        if source_uuid is not None:
            mappings[source_uuid] = row.local_id
    return mappings


def _append_mcp_integration_requirement(
    *,
    requirements: list[McpIntegrationMappingRequirement],
    diagnostics: list[PullDiagnostic],
    source_integration_id: uuid.UUID,
    meta: McpIntegrationRefMeta | None,
    references: list[McpIntegrationReference],
    local_integrations: list[MCPIntegration],
    reason: McpIntegrationMappingRequirementReason,
) -> None:
    """Append one grouped MCP mapping requirement and its blocking diagnostic."""
    first = references[0]
    label = f"{meta.slug!r}" if meta is not None else str(source_integration_id)
    if not local_integrations:
        diagnostics.append(
            PullDiagnostic(
                workflow_path=first.path,
                workflow_title=_mcp_reference_title(first),
                error_type="dependency",
                message=(
                    f"MCP integration {label} is referenced by this snapshot, "
                    "but no MCP integrations are configured for this workspace."
                ),
                details={
                    "code": "mcp_integration_mapping_required",
                    "mcp_integration_id": str(source_integration_id),
                    "reason": reason,
                },
            )
        )
        return

    # Slug-matched candidate first so the likely target leads the picker.
    candidates = sorted(
        local_integrations,
        key=lambda row: (
            meta is None or row.slug != meta.slug,
            row.slug,
        ),
    )
    if reason == "invalid_selection":
        message = (
            f"The selected target is not an MCP integration in this workspace. "
            f"Choose an available MCP integration for source {label}."
        )
    elif reason == "conflicting_metadata":
        message = (
            f"MCP integration {label} carries conflicting correlation "
            "metadata across this snapshot's files. Choose the target "
            "integration explicitly."
        )
    else:
        message = (
            f"MCP integration {label} could not be matched to a local MCP "
            "integration. Choose the target integration before applying "
            "this pull."
        )
    diagnostics.append(
        PullDiagnostic(
            workflow_path=first.path,
            workflow_title=_mcp_reference_title(first),
            error_type="dependency",
            message=message,
            details={
                "code": "mcp_integration_mapping_required",
                "mcp_integration_id": str(source_integration_id),
                "reason": reason,
            },
        )
    )
    requirements.append(
        McpIntegrationMappingRequirement(
            source_mcp_integration_id=source_integration_id,
            slug=meta.slug if meta else None,
            name=meta.name if meta else None,
            server_type=meta.server_type if meta else None,
            auth_type=meta.auth_type if meta else None,
            reason=reason,
            message=message,
            candidates=[
                McpIntegrationMappingCandidate(
                    mcp_integration_id=row.id,
                    slug=row.slug,
                    name=row.name,
                    server_type=row.server_type,
                    auth_type=str(row.auth_type),
                )
                for row in candidates
            ],
            affected_presets=[
                McpIntegrationMappingAffectedPreset(
                    preset_slug=reference.preset_slug,
                    preset_name=reference.preset_name,
                    version=reference.version_number,
                    path=reference.path,
                )
                for reference in references
                if isinstance(reference, AgentPresetMcpIntegrationReference)
            ],
            affected_workflows=[
                McpIntegrationMappingAffectedWorkflow(
                    workflow_source_id=reference.workflow_source_id,
                    workflow_path=reference.path,
                    workflow_title=reference.workflow_title,
                    action_ref=reference.action_ref,
                )
                for reference in references
                if isinstance(reference, WorkflowMcpIntegrationReference)
            ],
        )
    )


def _mcp_reference_title(reference: McpIntegrationReference) -> str:
    """Return the owning preset or workflow title for an MCP reference."""
    if isinstance(reference, AgentPresetMcpIntegrationReference):
        return reference.preset_name
    return reference.workflow_title
