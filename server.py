import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import yaml
from gooddata_pandas import GoodPandas
from gooddata_sdk import (
    GoodDataSdk,
    CatalogDependentEntitiesRequest,
    CatalogEntityIdentifier,
)
from gooddata_sdk.compute.model.base import Filter, ObjId
from gooddata_sdk.compute.model.filter import (
    PositiveAttributeFilter,
    RelativeDateFilter,
)
from gooddata_api_client.model.scan_sql_request import ScanSqlRequest
from ldm_quality_check import (
    has_no_description,
    obfuscated_title_check,
    semantic_similarity_check,
)
from visualization_converter import convert
import uuid

# Load environment variables from .env file
load_dotenv()

# Initialize the MCP server
mcp = FastMCP("Demo")

# Initialize GoodData SDK using environment variables for host and token
GD_HOST = os.environ.get("GOODDATA_HOST")
GD_TOKEN = os.environ.get("GOODDATA_TOKEN")
GD_WORKSPACE = os.environ.get("GOODDATA_WORKSPACE")
GD_DATA_SOURCE = os.environ.get("GOODDATA_DATA_SOURCE")
gd = GoodDataSdk.create(host_=GD_HOST, token_=GD_TOKEN)
gp = GoodPandas(host=GD_HOST, token=GD_TOKEN) if GD_HOST and GD_TOKEN else None


def _workspace_or_default(workspace_id: Optional[str] = None) -> str:
    workspace = workspace_id or GD_WORKSPACE
    if not workspace:
        raise ValueError(
            "Workspace ID is not configured. Set GOODDATA_WORKSPACE or pass workspace_id."
        )
    return workspace


@lru_cache(maxsize=8)
def _get_ldm(workspace_id: str):
    return gd.catalog_workspace_content.get_declarative_ldm(workspace_id=workspace_id)


@lru_cache(maxsize=8)
def _get_analytics_model(workspace_id: str):
    return gd.catalog_workspace_content.get_declarative_analytics_model(
        workspace_id=workspace_id
    )


@lru_cache(maxsize=8)
def _label_index(workspace_id: str) -> Dict[str, Dict[str, Any]]:
    ldm = _get_ldm(workspace_id)
    index: Dict[str, Dict[str, Any]] = {}
    datasets = getattr(getattr(ldm, "ldm", None), "datasets", []) or []
    for dataset in datasets:
        dataset_id = getattr(dataset, "id", None)
        dataset_title = getattr(dataset, "title", dataset_id)
        dataset_type = getattr(dataset, "type", getattr(dataset, "dataset", None))
        attributes = getattr(dataset, "attributes", []) or []
        for attribute in attributes:
            attribute_id = getattr(attribute, "id", None)
            attribute_title = getattr(attribute, "title", attribute_id)
            default_view = getattr(attribute, "default_view", None)
            if default_view and getattr(default_view, "id", None):
                label_id = default_view.id
                index.setdefault(
                    label_id,
                    {
                        "attribute_id": attribute_id,
                        "attribute_title": attribute_title,
                        "dataset_id": dataset_id,
                        "dataset_title": dataset_title,
                        "dataset_type": dataset_type,
                        "label_title": getattr(default_view, "title", label_id),
                    },
                )
            for label in getattr(attribute, "labels", []) or []:
                label_id = getattr(label, "id", None)
                if not label_id:
                    continue
                index[label_id] = {
                    "attribute_id": attribute_id,
                    "attribute_title": attribute_title,
                    "dataset_id": dataset_id,
                    "dataset_title": dataset_title,
                    "dataset_type": dataset_type,
                    "label_title": getattr(label, "title", label_id),
                }
    return index


@lru_cache(maxsize=8)
def _attribute_default_labels(workspace_id: str) -> Dict[str, str]:
    index = _label_index(workspace_id)
    mapping: Dict[str, str] = {}
    for label_id, meta in index.items():
        attr_id = meta.get("attribute_id")
        if attr_id and attr_id not in mapping:
            mapping[attr_id] = label_id
    return mapping


@lru_cache(maxsize=8)
def _metric_index(workspace_id: str) -> Dict[str, Dict[str, Any]]:
    analytics = _get_analytics_model(workspace_id)
    metrics = getattr(getattr(analytics, "analytics", None), "metrics", []) or []
    index: Dict[str, Dict[str, Any]] = {}
    for metric in metrics:
        metric_id = getattr(metric, "id", None)
        if not metric_id:
            continue
        index[metric_id] = {
            "title": getattr(metric, "title", metric_id),
            "description": getattr(metric, "description", ""),
        }
    return index


def _search_objects(
    question: Optional[str], object_types: List[str], workspace_id: str, limit: int = 5
) -> List[Dict[str, Any]]:
    if not question:
        return []
    try:
        result = gd.compute.search_ai(
            workspace_id=workspace_id, question=question, object_types=object_types
        )
        results = getattr(result, "results", []) or []
        return list(results)[:limit]
    except Exception:
        return []


def _normalize_label_identifier(identifier: str, workspace_id: str) -> Optional[str]:
    if not identifier:
        return None
    if identifier.startswith("label/"):
        return identifier
    if identifier.startswith("attribute/"):
        return _attribute_default_labels(workspace_id).get(identifier)
    return identifier


def _normalize_metric_identifier(identifier: Optional[str]) -> Optional[str]:
    if not identifier:
        return None
    return identifier if identifier.startswith("metric/") else f"metric/{identifier}"


def _resolve_metrics(
    prompt: Optional[str], metrics: Optional[List[str]], workspace_id: str
) -> Tuple[List[str], List[Dict[str, Any]]]:
    metric_ids = [_normalize_metric_identifier(m) for m in (metrics or []) if m]
    metric_ids = [m for m in metric_ids if m]
    suggestions: List[Dict[str, Any]] = []
    if metric_ids:
        return metric_ids, suggestions
    suggestions = _search_objects(prompt, ["metric"], workspace_id, limit=3)
    metric_ids = [
        item.get("id")
        for item in suggestions
        if item.get("type") == "metric" and item.get("id")
    ]
    if metric_ids:
        metric_ids = [_normalize_metric_identifier(metric_ids[0])]
    return metric_ids, suggestions


def _resolve_dimensions(
    prompt: Optional[str],
    dimensions: Optional[List[str]],
    workspace_id: str,
    limit: int = 3,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    label_ids: List[str] = []
    suggestions: List[Dict[str, Any]] = []
    for dim in dimensions or []:
        label_id = _normalize_label_identifier(dim, workspace_id)
        if label_id:
            label_ids.append(label_id)
    if label_ids:
        # Deduplicate preserving order
        seen = set()
        label_ids = [lid for lid in label_ids if not (lid in seen or seen.add(lid))]
        return label_ids, suggestions
    suggestions = _search_objects(
        prompt, ["label", "attribute"], workspace_id, limit=limit
    )
    for item in suggestions:
        identifier = item.get("id")
        if not identifier:
            continue
        if item.get("type") == "attribute":
            identifier = _attribute_default_labels(workspace_id).get(identifier)
        if identifier:
            label_ids.append(identifier)
    # Deduplicate preserving order
    seen: set[str] = set()
    unique = []
    for lid in label_ids:
        if lid and lid not in seen:
            unique.append(lid)
            seen.add(lid)
    return unique, suggestions


def _select_date_label(
    label_ids: List[str], date_dataset_id: Optional[str], workspace_id: str
) -> Tuple[Optional[str], Optional[str]]:
    """Return (dataset_id, label_id) for date filtering and grouping."""
    index = _label_index(workspace_id)
    candidates: List[Tuple[int, str, str]] = []
    preferred_dataset = date_dataset_id
    for label_id, meta in index.items():
        dataset_id = meta.get("dataset_id")
        dataset_type = (
            (meta.get("dataset_type") or "").lower() if meta.get("dataset_type") else ""
        )
        label_title = (meta.get("label_title") or "").lower()
        if dataset_type == "date" or (
            dataset_id and dataset_id.lower().startswith("date")
        ):
            score = 0
            label_id_lower = label_id.lower()
            if "month" in label_id_lower or "month" in label_title:
                score -= 2
            if "snapshot" in label_id_lower or "snapshot" in label_title:
                score -= 1
            candidates.append((score, dataset_id, label_id))

    # If any of the provided dimensions already cover a date dataset, prioritize it
    for lid in label_ids:
        meta = index.get(lid)
        if not meta:
            continue
        dataset_id = meta.get("dataset_id")
        dataset_type = (
            (meta.get("dataset_type") or "").lower() if meta.get("dataset_type") else ""
        )
        if date_dataset_id and dataset_id == date_dataset_id:
            return dataset_id, lid
        if dataset_type == "date" or (
            dataset_id and dataset_id.lower().startswith("date")
        ):
            return dataset_id, lid

    if preferred_dataset:
        for score, dataset_id, label_id in sorted(candidates):
            if dataset_id == preferred_dataset:
                return dataset_id, label_id

    if candidates:
        candidates.sort()
        return candidates[0][1], candidates[0][2]

    return None, None


def _dataset_obj_id(dataset_id: str) -> ObjId:
    if dataset_id.startswith("dataset/"):
        return ObjId(id=dataset_id.split("/", 1)[1], type="dataset")
    return ObjId(id=dataset_id, type="dataset")


def _build_relative_date_filter(
    dataset_id: str, months: int, include_current: bool
) -> RelativeDateFilter:
    if months <= 0:
        raise ValueError("months must be positive")
    to_value = 0 if include_current else -1
    from_value = -(months - 1) if include_current else -months
    return RelativeDateFilter(
        dataset=_dataset_obj_id(dataset_id),
        granularity="MONTH",
        from_shift=from_value,
        to_shift=to_value,
    )


def _build_attribute_filters(
    attribute_filters: Optional[Dict[str, List[str]]], workspace_id: str
) -> Tuple[List[PositiveAttributeFilter], List[Dict[str, Any]]]:
    if not attribute_filters:
        return [], []
    filters: List[PositiveAttributeFilter] = []
    meta: List[Dict[str, Any]] = []
    for identifier, values in attribute_filters.items():
        label_id = _normalize_label_identifier(identifier, workspace_id)
        if not label_id or not values:
            continue
        filters.append(PositiveAttributeFilter(label=label_id, values=values))
        meta.append(
            {
                "type": "attribute",
                "label_id": label_id,
                "source_identifier": identifier,
                "values": values,
            }
        )
    return filters, meta


def _alias_from_title(title: Optional[str], fallback: str) -> str:
    base = (title or fallback or "value").strip()
    if not base:
        base = fallback or "value"
    normalized = "".join(ch if ch.isalnum() else "_" for ch in base)
    normalized = normalized.strip("_") or fallback.replace("/", "_")
    # Compress consecutive underscores
    parts = [segment for segment in normalized.split("_") if segment]
    alias = "_".join(parts) if parts else fallback.replace("/", "_")
    return alias


@mcp.tool(
    name="analyze_ldm",
    description="Analyze the declarative Logical Data Model (LDM) for missing or well-defined descriptions on attributes and facts. Returns counts and examples.",
)
def analyze_ldm() -> dict:
    """Analyze the declarative LDM for missing/well-defined descriptions of attributes and facts."""
    try:
        declarative_ldm = gd.catalog_workspace_content.get_declarative_ldm(
            workspace_id=GD_WORKSPACE
        )
        datasets = getattr(declarative_ldm.ldm, "datasets", [])
        missing_descriptions_attributes = []
        missing_descriptions_facts = []
        obfuscated_title_attributes = []
        obfuscated_title_facts = []
        similar_attributes = []
        similar_facts = []
        for ds in datasets:
            similar_attributes = semantic_similarity_check(
                ds.attributes
            ).semantically_similar_pairs
            similar_facts = semantic_similarity_check(
                ds.facts
            ).semantically_similar_pairs
            for attr in ds.attributes:
                if has_no_description(attr):
                    missing_descriptions_attributes.append(
                        {"title": attr.title, "id": attr.id}
                    )
                obfuscated_title_result = obfuscated_title_check(attr)
                if obfuscated_title_result.is_obfuscated:
                    obfuscated_title_attributes.append(
                        {
                            "title": attr.title,
                            "reason": obfuscated_title_result.reason,
                            "id": attr.id,
                        }
                    )
            for fact in ds.facts:
                if has_no_description(fact):
                    missing_descriptions_facts.append(
                        {"title": fact.title, "id": fact.id}
                    )
                obfuscated_title_result = obfuscated_title_check(fact)
                if obfuscated_title_result.is_obfuscated:
                    obfuscated_title_facts.append(
                        {
                            "title": fact.title,
                            "reason": obfuscated_title_result.reason,
                            "id": fact.id,
                        }
                    )
        result = {
            "missing_descriptions_attributes": len(missing_descriptions_attributes),
            "missing_descriptions_facts ": len(missing_descriptions_facts),
            "missing_descriptions_attributes_examples": missing_descriptions_attributes[
                :5
            ],
            "missing_descriptions_facts_examples": missing_descriptions_facts[:5],
            "obfuscated_title_attributes": len(obfuscated_title_attributes),
            "obfuscated_title_facts": len(obfuscated_title_facts),
            "obfuscated_title_attributes_examples": obfuscated_title_attributes[:5],
            "obfuscated_title_facts_examples": obfuscated_title_facts[:5],
            "similar_attributes": similar_attributes,
            "similar_facts": similar_facts,
        }
        return yaml.safe_dump(result, sort_keys=False, allow_unicode=True)
    except Exception as e:
        return yaml.safe_dump({"error": str(e)}, sort_keys=False, allow_unicode=True)


@mcp.tool(
    name="analyze_field",
    description="Analyze the specific field in the Logical Data Model (LDM)",
)
def analyze_field(dataset_id: str, field_id: str) -> dict:
    """Gather info about a specific field: DB name, dataset, title, description, and sample data."""
    try:
        # Fetch LDM info
        declarative_ldm = gd.catalog_workspace_content.get_declarative_ldm(
            workspace_id=GD_WORKSPACE
        )
        field_meta = None
        for ds in getattr(declarative_ldm.ldm, "datasets", []):
            if ds.id == dataset_id:
                for attr in getattr(ds, "attributes", []):
                    if attr.id == field_id:
                        field_meta = {
                            "dataset_id": ds.id,
                            "dataset_title": ds.title,
                            "field_id": attr.id,
                            "field_title": attr.title,
                            "field_description": getattr(attr, "description", None),
                            "source_column": getattr(attr, "source_column", None),
                            "source_table": ds.data_source_table_id.path[-1],
                        }
                        break

        if not field_meta:
            raise Exception(f"Field {field_id} not found in LDM")
        # Sample data
        sql_request = ScanSqlRequest(
            sql=f"SELECT DISTINCT \"{field_meta['source_column']}\" FROM \"{field_meta['source_table']}\" ORDER BY RANDOM() LIMIT 10;",
        )
        result = gd.client.actions_api.scan_sql(GD_DATA_SOURCE, sql_request)
        sample_data = ", ".join([row[0] for row in result["data_preview"]])
        result = {"field_meta": field_meta, "sample_data": sample_data}
        return yaml.safe_dump(result, sort_keys=False, allow_unicode=True)
    except Exception as e:
        return yaml.safe_dump({"error": str(e)}, sort_keys=False, allow_unicode=True)


@mcp.tool(
    name="patch_ldm",
    description="Patch (update) the title and/or description of a dataset or attribute in the Logical Data Model (LDM). Persists changes.",
)
def patch_ldm(object_id: str, title: str = None, description: str = None) -> dict:
    """Patch the title and/or description of a dataset or attribute in the LDM."""
    try:
        # Fetch current LDM
        declarative_ldm = gd.catalog_workspace_content.get_declarative_ldm(
            workspace_id=GD_WORKSPACE
        )
        updated = False
        for ds in getattr(declarative_ldm.ldm, "datasets", []):
            if ds.id == object_id:
                if title:
                    ds.title = title
                if description:
                    ds.description = description
                updated = True
                break
            for attr in getattr(ds, "attributes", []):
                if attr.id == object_id:
                    if title:
                        attr.title = title
                    if description:
                        attr.description = description
                    updated = True
                    break
        if updated:
            gd.catalog_workspace_content.put_declarative_ldm(
                workspace_id=GD_WORKSPACE, ldm=declarative_ldm
            )
            return {"status": "OK"}
        else:
            return {"error": "Field not found"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="explain_metric",
    description="Explain how a given metric is computed, including its MAQL expression, description, and where it is used across dashboards and insights.",
)
def explain_metric(metric_id: str) -> dict:
    """
    Explain how a given metric is computed and where it is used.
    Unfold nested metrics and translate MAQL (not implemented).
    """
    try:
        declarative_analytics = (
            gd.catalog_workspace_content.get_declarative_analytics_model(
                workspace_id=GD_WORKSPACE
            )
        )
        metrics = declarative_analytics.analytics.metrics

        # 1. Find MAQL for the metric (try id and local_identifier)
        maql = None
        description = None
        local_identifier = None
        found_metric = None
        for m in metrics:
            if m.id == metric_id or getattr(m, "local_identifier", None) == metric_id:
                maql = m.content.get("maql")
                description = m.description
                found_metric = m
                break

        result_dependencies = (
            gd.catalog_workspace_content.get_dependent_entities_graph_from_entry_points(
                GD_WORKSPACE,
                CatalogDependentEntitiesRequest(
                    identifiers=[CatalogEntityIdentifier(id=metric_id, type="metric")]
                ),
            )
        )
        used_in = [(i.title, i.type) for i in result_dependencies.graph.nodes]

        whole_graph = gd.catalog_workspace_content.get_dependent_entities_graph(
            GD_WORKSPACE
        )

        uses_ids = [
            (edge[0].id, edge[0].type)
            for edge in whole_graph.graph.edges
            if edge[1].id == metric_id and edge[1].type == "metric"
        ]
        uses = [
            (i.title, i.type)
            for i in whole_graph.graph.nodes
            if (i.id, i.type) in uses_ids
        ]
        # TODO: it would be helpful to fetch uses descriptions

        result = {
            "metric_id": metric_id,
            "maql": maql,
            "description": description,
            "usage_total_count": len(used_in),
            "usage_example": used_in[:10],  # limit to 10 usages for brevity
            "uses": uses,
            "uses_total_count": len(uses),
        }
        return yaml.safe_dump(result, sort_keys=False, allow_unicode=True)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="fetch_metrics_dataset",
    description=(
        "Fetch metrics with optional natural-language resolution of metrics and dimensions. "
        "Returns data as CSV (for large datasets) or JSON with a preview table for display. "
        "Automatically uses CSV format for datasets with more than 100 rows."
    ),
)
def fetch_metrics_dataset(
    prompt: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    months: int = 3,
    include_current: bool = True,
    date_dataset_id: Optional[str] = None,
    attribute_filters: Optional[Dict[str, List[str]]] = None,
    limit: Optional[int] = None,
    sort: Optional[List[str]] = None,
    workspace_id: Optional[str] = None,
    output_format: str = "auto",
) -> Dict[str, Any]:
    """Return metric datasets with preview table. Uses CSV for large datasets (>100 rows) automatically."""

    try:
        if gp is None:
            return {
                "error": "GoodPandas is not initialized. Set GOODDATA_HOST and GOODDATA_TOKEN to enable data access.",
            }

        workspace = _workspace_or_default(workspace_id)

        resolved_metrics, metric_suggestions = _resolve_metrics(
            prompt, metrics, workspace
        )
        resolved_dimensions, dimension_suggestions = _resolve_dimensions(
            prompt, dimensions, workspace, limit=3
        )

        metric_suggestions = metric_suggestions or []
        dimension_suggestions = dimension_suggestions or []

        if not resolved_metrics:
            return {
                "error": "No metric identified. Provide a metric id in `metrics` or adjust the prompt.",
                "suggestions": {"metrics": metric_suggestions},
            }

        label_index = _label_index(workspace)
        dataset_id_for_filter, date_label = _select_date_label(
            resolved_dimensions, date_dataset_id, workspace
        )
        if date_label and date_label not in resolved_dimensions:
            resolved_dimensions.append(date_label)

        attribute_filters_list, attribute_filter_meta = _build_attribute_filters(
            attribute_filters, workspace
        )
        filters: List[Filter] = list(attribute_filters_list)
        filter_meta: List[Dict[str, Any]] = []
        if dataset_id_for_filter:
            try:
                filters.append(
                    _build_relative_date_filter(
                        dataset_id_for_filter, months, include_current
                    )
                )
                filter_meta.append(
                    {
                        "type": "relative_date",
                        "dataset_id": dataset_id_for_filter,
                        "window_months": months,
                        "include_current": include_current,
                    }
                )
            except ValueError as exc:
                return {"error": str(exc)}
        else:
            filter_meta.append(
                {
                    "type": "relative_date",
                    "applied": False,
                    "reason": "No date dataset detected. Pass date_dataset_id or include a date label in dimensions.",
                }
            )

        columns: Dict[str, str] = {}
        column_meta: List[Dict[str, Any]] = []
        used_aliases: Dict[str, int] = {}

        def _unique_alias(base: str) -> str:
            count = used_aliases.get(base, 0)
            if count == 0:
                used_aliases[base] = 1
                return base
            count += 1
            used_aliases[base] = count
            return f"{base}_{count}"

        metric_meta = _metric_index(workspace)

        for label_id in resolved_dimensions:
            meta = label_index.get(label_id, {})
            alias = _alias_from_title(meta.get("label_title"), label_id)
            alias = _unique_alias(alias)
            formatted_id = (
                label_id if label_id.startswith("label/") else f"label/{label_id}"
            )
            columns[alias] = formatted_id
            column_meta.append(
                {
                    "alias": alias,
                    "id": formatted_id,
                    "type": "label",
                    "dataset_id": meta.get("dataset_id"),
                    "dataset_title": meta.get("dataset_title"),
                    "attribute_id": meta.get("attribute_id"),
                    "attribute_title": meta.get("attribute_title"),
                }
            )

        for metric_id in resolved_metrics:
            meta = metric_meta.get(metric_id, {})
            alias = _alias_from_title(meta.get("title"), metric_id)
            alias = _unique_alias(alias)
            columns[alias] = metric_id
            column_meta.append(
                {
                    "alias": alias,
                    "id": metric_id,
                    "type": "metric",
                    "title": meta.get("title", metric_id),
                    "description": meta.get("description"),
                }
            )

        frames = gp.data_frames(workspace)
        df = frames.not_indexed(columns=columns, filter_by=filters or None)

        if sort:
            for key in reversed(sort):
                if not key:
                    continue
                ascending = True
                column_name = key
                if key.startswith("-"):
                    ascending = False
                    column_name = key[1:]
                elif key.startswith("+"):
                    column_name = key[1:]
                if column_name in getattr(df, "columns", []):
                    df = df.sort_values(by=column_name, ascending=ascending)

        # Compute total rows before any limiting
        total_row_count = len(df) if hasattr(df, "__len__") else 0

        # Decide whether caller explicitly wants CSV; if so and limit is omitted,
        # do not apply the default 50-row limit (export all rows).
        wants_csv = (output_format or "auto").lower() == "csv"

        # Apply a default limit of 50 rows unless the caller specifies otherwise.
        # Convention: limit <= 0 means "no limit" (return all rows).
        if limit is None and wants_csv:
            effective_limit = None
        else:
            effective_limit = 50 if (limit is None) else limit

        if effective_limit is not None and effective_limit > 0 and hasattr(df, "head"):
            df = df.head(effective_limit)

        row_count = len(df) if hasattr(df, "__len__") else 0

        # Always create a preview table (first 30 rows) for display
        preview_df = df.head(30) if hasattr(df, "head") else df
        preview_table = (
            preview_df.to_string(index=False)
            if hasattr(preview_df, "to_string")
            else str(preview_df)
        )

        # Auto-detect format: use CSV for large datasets (>100 rows) or if explicitly requested
        if output_format.lower() == "auto":
            use_csv = row_count > 100
        else:
            use_csv = output_format.lower() == "csv"

        if use_csv:
            csv_string = df.to_csv(index=False) if hasattr(df, "to_csv") else str(df)
            response: Dict[str, Any] = {
                "csv": csv_string,
                "preview": preview_table,
                "meta": {
                    "workspace_id": workspace,
                    "columns": column_meta,
                    "metrics": resolved_metrics,
                    "dimensions": resolved_dimensions,
                    "filters": filter_meta + attribute_filter_meta,
                    "row_count": row_count,
                    "total_row_count": total_row_count,
                    "preview_rows": len(preview_df)
                    if hasattr(preview_df, "__len__")
                    else 0,
                },
            }
        else:
            records = df.to_dict(orient="records") if hasattr(df, "to_dict") else df
            response: Dict[str, Any] = {
                "data": records,
                "preview": preview_table,
                "meta": {
                    "workspace_id": workspace,
                    "columns": column_meta,
                    "metrics": resolved_metrics,
                    "dimensions": resolved_dimensions,
                    "filters": filter_meta + attribute_filter_meta,
                    "row_count": row_count,
                    "total_row_count": total_row_count,
                },
            }

        suggestions: Dict[str, Any] = {}
        if metric_suggestions:
            suggestions["metrics"] = metric_suggestions
        if dimension_suggestions:
            suggestions["dimensions"] = dimension_suggestions
        if suggestions:
            response["suggestions"] = suggestions

        return response
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="search",
    description="Search facts, metrics, attributes, date instances, visualizations or dashboards in the workspace.",
)
def search(term: str, types: list[str] = []) -> dict:
    """
    Use the GoodData SDK to search for facts, metrics, attributes, date instances, visualizations or dashboards in the workspace.
    """
    try:
        return {
            "result": [
                {
                    "id": result["id"],
                    "title": result["title"],
                    "description": result.get("description", None),
                    "type": result["type"],
                    "visualization_type": result.get("visualization_type", None),
                    "match_score": result.get("score", 0.0),
                }
                for result in gd.compute.search_ai(
                    workspace_id=GD_WORKSPACE, question=term, object_types=types
                ).results
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="create_visualization",
    description="Creates a visualization using a prompt and adds it directly to the GoodData workspace. Returns a confirmation and the new visualization's ID.",
)
def create_visualization(prompt: str) -> dict:
    """
    Calls the GoodData AI compute engine to create a visualization and adds it to the workspace.
    Returns a confirmation message and the new visualization's ID.
    """
    try:
        result = gd.compute.ai_chat_stream(workspace_id=GD_WORKSPACE, question=prompt)
        visualization = [chunk for chunk in result if "createdVisualizations" in chunk]
        if len(visualization) == 0:
            return {"error": "No visualization object found in AI chat output."}
        visualization = visualization[0].get("createdVisualizations", {})
        visualization_converted = convert(visualization)
        if len(visualization_converted) == 0:
            return {"error": "Conversion failed."}
        declarative_workspace = gd.catalog_workspace.get_declarative_workspace(
            workspace_id=GD_WORKSPACE
        )
        if hasattr(declarative_workspace.analytics, "visualization_objects"):
            declarative_workspace.analytics.visualization_objects.append(
                visualization_converted
            )
        else:
            declarative_workspace.analytics.visualization_objects = [
                visualization_converted
            ]
        gd.catalog_workspace.put_declarative_workspace(
            workspace_id=GD_WORKSPACE, workspace=declarative_workspace
        )

        return {
            "message": f"Visualization '{visualization_converted.get('title')}' added to workspace.",
            "id": visualization_converted.get("id"),
            "url": f"{GD_HOST}/analyze/#/{GD_WORKSPACE}/{visualization_converted['id']}/edit",
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="add_visualization_to_dashboard",
    description="Add a visualization to a dashboard. Requires the visualization_id and dashboard_id as inputs.",
)
def add_visualization_to_dashboard(visualization_id: str, dashboard_id: str) -> str:
    """
    You must provide the visualization_id of an existing visualization (ask for it if not provided). This tool will then place it on the first dashboard. It does not generate or search for the visualization_id itself. Returns a YAML message confirming the visual has been placed in the dashboard.
    """
    try:
        declarative_workspace = gd.catalog_workspace.get_declarative_workspace(
            workspace_id=GD_WORKSPACE
        )
        dashboards = getattr(
            declarative_workspace.analytics, "analytical_dashboards", []
        )
        if not dashboards:
            return yaml.safe_dump(
                {"error": "No dashboards found in workspace."},
                sort_keys=False,
                allow_unicode=True,
            )
        dashboard = next((d for d in dashboards if d.id == dashboard_id), None)
        if not dashboard:
            return yaml.safe_dump(
                {"error": f"Dashboard {dashboard_id} not found in workspace."},
                sort_keys=False,
                allow_unicode=True,
            )
        layout = dashboard.content.get("layout", {})
        sections = layout.get("sections", [])

        # Use the first item in the first section as a template
        if sections and sections[0]["items"]:
            from copy import deepcopy

            template_item = deepcopy(sections[0]["items"][0])
            # Update only the fields needed for the new visualization
            widget = template_item["widget"]
            widget["insight"]["identifier"]["id"] = visualization_id
            widget["title"] = f"Visualization {visualization_id}"
            # Generate a new unique localIdentifier if present
            if "localIdentifier" in widget:
                widget["localIdentifier"] = str(uuid.uuid4())
            template_item["widget"] = widget
            # Prepend the new item
            sections[0]["items"] = [template_item] + sections[0]["items"]
        else:
            # Fallback: create a minimal valid item if no template exists
            new_item = {
                "size": {"xl": {"gridWidth": 12}},
                "type": "IDashboardLayoutItem",
                "widget": {
                    "type": "insight",
                    "insight": {
                        "identifier": {
                            "id": visualization_id,
                            "type": "visualizationObject",
                        }
                    },
                    "title": f"Visualization {visualization_id}",
                    "localIdentifier": str(uuid.uuid4()),
                    "configuration": {
                        "description": {
                            "includeMetrics": False,
                            "source": "widget",
                            "visible": True,
                        },
                        "hideTitle": False,
                    },
                    "properties": {},
                },
            }
            sections.append({"items": [new_item], "type": "IDashboardLayoutSection"})
        layout["sections"] = sections
        dashboard.content["layout"] = layout
        gd.catalog_workspace.put_declarative_workspace(
            workspace_id=GD_WORKSPACE, workspace=declarative_workspace
        )
        result = {
            "message": f"Visualization {visualization_id} has been placed in the dashboard.",
            "visualization_id": visualization_id,
            "url": f"{GD_HOST}/dashboards/#/workspace/{GD_WORKSPACE}/dashboard/{dashboard_id}",
        }
        return yaml.safe_dump(result, sort_keys=False, allow_unicode=True)
    except Exception as e:
        return yaml.safe_dump({"error": str(e)}, sort_keys=False, allow_unicode=True)


# Reset logging settings that MCP made because we want to use our own logging configuration configured in the bootstrap script
logging.basicConfig(force=True, handlers=[], level=logging.NOTSET)
