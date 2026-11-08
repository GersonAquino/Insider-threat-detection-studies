from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from common import (
    CERT_DATETIME_FORMAT,
    compact_text,
    email_domains,
    is_external_email,
    is_external_url,
    label_from_bool,
    load_config,
    parse_cert_datetime,
    resolve_path,
    sanitize_for_prompt,
    sql_literal_path,
    sql_string,
    stable_int,
    url_domain,
    volume_class,
    write_json,
    write_jsonl,
)


CERT_TABLES = {
    "logon": "logon.csv",
    "device": "device.csv",
    "file": "file.csv",
    "email": "email.csv",
    "http": "http.csv",
}


def require_duckdb():
    try:
        import duckdb
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "This pipeline requires duckdb. Install it with: python -m pip install duckdb pyyaml"
        ) from exc
    return duckdb


def answer_scenario_from_path(path: Path) -> dict[str, str]:
    stem = path.stem
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].startswith("r"):
        return {
            "dataset": parts[0][1:],
            "scenario": parts[1],
            "answer_user": "-".join(parts[2:]),
            "answer_file": path.name,
        }
    parent_parts = path.parent.name.split("-")
    return {
        "dataset": parent_parts[0][1:] if parent_parts and parent_parts[0].startswith("r") else "",
        "scenario": parent_parts[1] if len(parent_parts) > 1 else "",
        "answer_user": stem,
        "answer_file": path.name,
    }


def load_answer_user_days(cfg: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    pattern = str(resolve_path(cfg, cfg["dataset"]["answers_glob"]))
    answer_paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not answer_paths:
        raise FileNotFoundError(f"No answer files matched: {pattern}")

    user_days: dict[tuple[str, str], dict[str, Any]] = {}
    skipped_rows = 0
    total_answer_events = 0
    for path in answer_paths:
        scenario = answer_scenario_from_path(path)
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            for line_number, row in enumerate(reader, start=1):
                if len(row) < 5:
                    skipped_rows += 1
                    continue
                source = row[0].strip().lower()
                event_id = row[1].strip()
                try:
                    ts = parse_cert_datetime(row[2])
                except Exception:
                    skipped_rows += 1
                    continue
                user = row[3].strip()
                if not user:
                    skipped_rows += 1
                    continue
                day = ts.date().isoformat()
                key = (user, day)
                payload = user_days.setdefault(
                    key,
                    {
                        "user": user,
                        "date": day,
                        "scenario": set(),
                        "answer_reference": [],
                        "malicious_event_count": 0,
                    },
                )
                if scenario.get("scenario"):
                    payload["scenario"].add(str(scenario["scenario"]))
                payload["malicious_event_count"] += 1
                payload["answer_reference"].append(
                    {
                        "source": source,
                        "event_id": event_id,
                        "timestamp": ts.isoformat(sep=" "),
                        "answer_file": scenario["answer_file"],
                        "answer_line": line_number,
                    }
                )
                total_answer_events += 1

    for payload in user_days.values():
        payload["scenario"] = sorted(payload["scenario"])

    meta = {
        "answer_glob": pattern,
        "answer_files": [str(path) for path in answer_paths],
        "num_answer_files": len(answer_paths),
        "num_answer_events": total_answer_events,
        "num_positive_user_days_from_answers": len(user_days),
        "skipped_answer_rows": skipped_rows,
    }
    return user_days, meta


def connect_duckdb(cfg: dict[str, Any]):
    duckdb = require_duckdb()
    db_path = cfg["duckdb"].get("database") or ":memory:"
    con = duckdb.connect(database=db_path)
    con.execute(f"PRAGMA threads={int(cfg['duckdb'].get('threads', 1))}")
    memory_limit = str(cfg["duckdb"].get("memory_limit", "")).strip()
    if memory_limit:
        con.execute(f"PRAGMA memory_limit='{sql_string(memory_limit)}'")
    return con


def read_csv_expr(path: Path) -> str:
    return (
        f"read_csv_auto('{sql_literal_path(path)}', "
        "header=true, all_varchar=true, ignore_errors=true, union_by_name=true)"
    )


def create_event_views(con, cfg: dict[str, Any]) -> None:
    raw_dir = resolve_path(cfg, cfg["dataset"]["raw_dir"])
    include_content = bool(cfg["dataset"].get("include_content_preview", True))

    paths = {name: raw_dir / filename for name, filename in CERT_TABLES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing CERT raw CSV files: " + ", ".join(missing))

    content_file = "content" if include_content else "NULL::VARCHAR"
    content_email = "content" if include_content else "NULL::VARCHAR"
    content_http = "content" if include_content else "NULL::VARCHAR"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW logon_events AS
        SELECT
            'logon'::VARCHAR AS source,
            id::VARCHAR AS event_id,
            try_strptime("date", '{CERT_DATETIME_FORMAT}') AS ts,
            trim(user)::VARCHAR AS user,
            pc::VARCHAR AS pc,
            activity::VARCHAR AS activity,
            NULL::VARCHAR AS filename,
            NULL::VARCHAR AS url,
            NULL::VARCHAR AS email_to,
            NULL::VARCHAR AS email_cc,
            NULL::VARCHAR AS email_bcc,
            NULL::VARCHAR AS email_from,
            NULL::VARCHAR AS size,
            NULL::VARCHAR AS attachments,
            NULL::VARCHAR AS content
        FROM {read_csv_expr(paths["logon"])}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW device_events AS
        SELECT
            'device'::VARCHAR AS source,
            id::VARCHAR AS event_id,
            try_strptime("date", '{CERT_DATETIME_FORMAT}') AS ts,
            trim(user)::VARCHAR AS user,
            pc::VARCHAR AS pc,
            activity::VARCHAR AS activity,
            NULL::VARCHAR AS filename,
            NULL::VARCHAR AS url,
            NULL::VARCHAR AS email_to,
            NULL::VARCHAR AS email_cc,
            NULL::VARCHAR AS email_bcc,
            NULL::VARCHAR AS email_from,
            NULL::VARCHAR AS size,
            NULL::VARCHAR AS attachments,
            NULL::VARCHAR AS content
        FROM {read_csv_expr(paths["device"])}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW file_events AS
        SELECT
            'file'::VARCHAR AS source,
            id::VARCHAR AS event_id,
            try_strptime("date", '{CERT_DATETIME_FORMAT}') AS ts,
            trim(user)::VARCHAR AS user,
            pc::VARCHAR AS pc,
            'File'::VARCHAR AS activity,
            filename::VARCHAR AS filename,
            NULL::VARCHAR AS url,
            NULL::VARCHAR AS email_to,
            NULL::VARCHAR AS email_cc,
            NULL::VARCHAR AS email_bcc,
            NULL::VARCHAR AS email_from,
            NULL::VARCHAR AS size,
            NULL::VARCHAR AS attachments,
            {content_file}::VARCHAR AS content
        FROM {read_csv_expr(paths["file"])}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW email_events AS
        SELECT
            'email'::VARCHAR AS source,
            id::VARCHAR AS event_id,
            try_strptime("date", '{CERT_DATETIME_FORMAT}') AS ts,
            trim(user)::VARCHAR AS user,
            pc::VARCHAR AS pc,
            'Email'::VARCHAR AS activity,
            NULL::VARCHAR AS filename,
            NULL::VARCHAR AS url,
            "to"::VARCHAR AS email_to,
            cc::VARCHAR AS email_cc,
            bcc::VARCHAR AS email_bcc,
            "from"::VARCHAR AS email_from,
            size::VARCHAR AS size,
            attachments::VARCHAR AS attachments,
            {content_email}::VARCHAR AS content
        FROM {read_csv_expr(paths["email"])}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW http_events AS
        SELECT
            'http'::VARCHAR AS source,
            id::VARCHAR AS event_id,
            try_strptime("date", '{CERT_DATETIME_FORMAT}') AS ts,
            trim(user)::VARCHAR AS user,
            pc::VARCHAR AS pc,
            'Http'::VARCHAR AS activity,
            NULL::VARCHAR AS filename,
            url::VARCHAR AS url,
            NULL::VARCHAR AS email_to,
            NULL::VARCHAR AS email_cc,
            NULL::VARCHAR AS email_bcc,
            NULL::VARCHAR AS email_from,
            NULL::VARCHAR AS size,
            NULL::VARCHAR AS attachments,
            {content_http}::VARCHAR AS content
        FROM {read_csv_expr(paths["http"])}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW all_events AS
        SELECT * FROM logon_events
        UNION ALL SELECT * FROM device_events
        UNION ALL SELECT * FROM file_events
        UNION ALL SELECT * FROM email_events
        UNION ALL SELECT * FROM http_events
        """
    )


def create_answer_table(con, answer_user_days: dict[tuple[str, str], dict[str, Any]]) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE answer_user_days (
            user VARCHAR,
            day DATE,
            malicious_event_count INTEGER,
            scenario VARCHAR
        )
        """
    )
    rows = [
        (
            payload["user"],
            payload["date"],
            int(payload["malicious_event_count"]),
            ",".join(payload.get("scenario") or []),
        )
        for payload in answer_user_days.values()
    ]
    if rows:
        con.executemany("INSERT INTO answer_user_days VALUES (?, CAST(? AS DATE), ?, ?)", rows)


def create_feature_tables(con, cfg: dict[str, Any]) -> None:
    sampling = cfg["sampling"]
    off_start = int(sampling["off_hours_start"])
    off_end = int(sampling["off_hours_end"])
    many_files = int(sampling["many_files_threshold"])
    bins = [int(value) for value in sampling.get("volume_bins", [20, 100, 300])]

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE user_day_features AS
        WITH daily AS (
            SELECT
                user,
                CAST(ts AS DATE) AS day,
                COUNT(*)::BIGINT AS total_events,
                SUM(CASE WHEN source = 'logon' THEN 1 ELSE 0 END)::BIGINT AS logon_count,
                SUM(CASE WHEN source = 'device' THEN 1 ELSE 0 END)::BIGINT AS device_count,
                SUM(CASE WHEN source = 'file' THEN 1 ELSE 0 END)::BIGINT AS file_count,
                SUM(CASE WHEN source = 'email' THEN 1 ELSE 0 END)::BIGINT AS email_count,
                SUM(CASE WHEN source = 'http' THEN 1 ELSE 0 END)::BIGINT AS http_count,
                SUM(CASE WHEN source = 'device' AND lower(coalesce(activity, '')) = 'connect' THEN 1 ELSE 0 END)::BIGINT
                    AS device_connect_count,
                SUM(CASE WHEN date_part('hour', ts) < {off_start} OR date_part('hour', ts) >= {off_end} THEN 1 ELSE 0 END)::BIGINT
                    AS off_hours_count,
                COUNT(DISTINCT pc)::BIGINT AS pc_count,
                MIN(ts) AS first_ts,
                MAX(ts) AS last_ts
            FROM all_events
            WHERE ts IS NOT NULL AND user IS NOT NULL AND trim(user) <> ''
            GROUP BY user, CAST(ts AS DATE)
        )
        SELECT
            *,
            CASE
                WHEN total_events < {bins[0]} THEN 'low'
                WHEN total_events < {bins[1]} THEN 'medium'
                WHEN total_events < {bins[2]} THEN 'high'
                ELSE 'very_high'
            END AS volume_bin,
            (
                CASE WHEN device_connect_count > 0 THEN 3.0 ELSE 0.0 END
                + CASE WHEN file_count >= {many_files} THEN 3.0 ELSE file_count * 0.04 END
                + CASE WHEN off_hours_count > 0 THEN 2.0 ELSE 0.0 END
                + CASE WHEN email_count > 0 THEN 0.8 ELSE 0.0 END
                + CASE WHEN http_count > 0 THEN 0.8 ELSE 0.0 END
                + CASE WHEN pc_count > 2 THEN 1.2 ELSE 0.0 END
            )::DOUBLE AS hard_score
        FROM daily
        """
    )


def fetch_dicts(con, sql: str) -> list[dict[str, Any]]:
    cur = con.execute(sql)
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def day_to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def feature_payload(row: dict[str, Any]) -> dict[str, Any]:
    numeric = [
        "total_events",
        "logon_count",
        "device_count",
        "file_count",
        "email_count",
        "http_count",
        "device_connect_count",
        "off_hours_count",
        "pc_count",
    ]
    payload: dict[str, Any] = {
        "volume_bin": str(row.get("volume_bin") or ""),
        "hard_score": float(row.get("hard_score") or 0.0),
    }
    for key in numeric:
        payload[key] = int(row.get(key) or 0)
    return payload


def build_case_id(dataset_name: str, user: str, day: str) -> str:
    safe_user = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in user)
    return f"{dataset_name}_{safe_user}_{day}"


def add_selected_case(
    selected: dict[tuple[str, str], dict[str, Any]],
    row: dict[str, Any],
    reason: str,
    cfg: dict[str, Any],
    answer_user_days: dict[tuple[str, str], dict[str, Any]],
) -> None:
    user = str(row["user"]).strip()
    day = day_to_iso(row["day"])
    key = (user, day)
    label = key in answer_user_days
    case = selected.get(key)
    if case is None:
        case = {
            "case_id": build_case_id(cfg["dataset"]["name"], user, day),
            "user": user,
            "date": day,
            "label": label_from_bool(label),
            "features": feature_payload(row),
            "sampling_reasons": [],
        }
        selected[key] = case
    if reason not in case["sampling_reasons"]:
        case["sampling_reasons"].append(reason)


def select_user_days(
    con,
    cfg: dict[str, Any],
    answer_user_days: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    sampling = cfg["sampling"]

    positive_rows = fetch_dicts(
        con,
        """
        SELECT f.*
        FROM user_day_features f
        INNER JOIN answer_user_days a ON a.user = f.user AND a.day = f.day
        ORDER BY f.user, f.day
        """,
    )
    for row in positive_rows:
        add_selected_case(selected, row, "positive_answer_user_day", cfg, answer_user_days)

    temporal_window = int(sampling["temporal_window_days"])
    temporal_limit = int(sampling["temporal_negatives_per_positive_day"])
    if temporal_limit > 0:
        rows = fetch_dicts(
            con,
            f"""
            WITH ranked AS (
                SELECT
                    f.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.user, p.day
                        ORDER BY abs(date_diff('day', f.day, p.day)) ASC,
                                 f.hard_score DESC,
                                 f.total_events DESC,
                                 f.day ASC
                    ) AS rn
                FROM user_day_features f
                INNER JOIN answer_user_days p ON p.user = f.user
                LEFT JOIN answer_user_days a ON a.user = f.user AND a.day = f.day
                WHERE a.user IS NULL
                  AND f.day <> p.day
                  AND abs(date_diff('day', f.day, p.day)) BETWEEN 1 AND {temporal_window}
            )
            SELECT *
            FROM ranked
            WHERE rn <= {temporal_limit}
            """,
        )
        for row in rows:
            add_selected_case(selected, row, "temporal_near_negative", cfg, answer_user_days)

    same_user_limit = int(sampling["same_user_negatives_per_positive_user"])
    if same_user_limit > 0:
        rows = fetch_dicts(
            con,
            f"""
            WITH positive_users AS (
                SELECT DISTINCT user FROM answer_user_days
            ),
            ranked AS (
                SELECT
                    f.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.user
                        ORDER BY f.hard_score DESC, f.total_events DESC, f.day ASC
                    ) AS rn
                FROM user_day_features f
                INNER JOIN positive_users pu ON pu.user = f.user
                LEFT JOIN answer_user_days a ON a.user = f.user AND a.day = f.day
                WHERE a.user IS NULL
            )
            SELECT *
            FROM ranked
            WHERE rn <= {same_user_limit}
            """,
        )
        for row in rows:
            add_selected_case(selected, row, "same_user_negative", cfg, answer_user_days)

    similar_limit = int(sampling["similar_user_negatives_per_positive_day"])
    similar_window = int(sampling["similar_temporal_window_days"])
    if similar_limit > 0:
        rows = fetch_dicts(
            con,
            f"""
            WITH pos AS (
                SELECT
                    p.user AS p_user,
                    p.day AS p_day,
                    f.total_events AS p_total_events,
                    f.logon_count AS p_logon_count,
                    f.device_count AS p_device_count,
                    f.file_count AS p_file_count,
                    f.email_count AS p_email_count,
                    f.http_count AS p_http_count,
                    f.volume_bin AS p_volume_bin
                FROM answer_user_days p
                INNER JOIN user_day_features f ON f.user = p.user AND f.day = p.day
            ),
            ranked AS (
                SELECT
                    f.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY pos.p_user, pos.p_day
                        ORDER BY
                            abs(f.total_events - pos.p_total_events) ASC,
                            (
                                abs(f.logon_count - pos.p_logon_count)
                                + abs(f.device_count - pos.p_device_count)
                                + abs(f.file_count - pos.p_file_count)
                                + abs(f.email_count - pos.p_email_count)
                                + abs(f.http_count - pos.p_http_count)
                            ) ASC,
                            f.hard_score DESC,
                            f.day ASC
                    ) AS rn
                FROM pos
                INNER JOIN user_day_features f
                    ON f.user <> pos.p_user
                   AND f.volume_bin = pos.p_volume_bin
                   AND f.day BETWEEN pos.p_day - INTERVAL '{similar_window} days'
                                 AND pos.p_day + INTERVAL '{similar_window} days'
                LEFT JOIN answer_user_days a ON a.user = f.user AND a.day = f.day
                WHERE a.user IS NULL
            )
            SELECT *
            FROM ranked
            WHERE rn <= {similar_limit}
            """,
        )
        for row in rows:
            add_selected_case(selected, row, "similar_user_negative", cfg, answer_user_days)

    hard_limit = int(sampling["hard_negative_limit"])
    if hard_limit > 0:
        rows = fetch_dicts(
            con,
            f"""
            SELECT f.*
            FROM user_day_features f
            LEFT JOIN answer_user_days a ON a.user = f.user AND a.day = f.day
            WHERE a.user IS NULL
            ORDER BY f.hard_score DESC, f.total_events DESC, f.day ASC, f.user ASC
            LIMIT {hard_limit}
            """,
        )
        for row in rows:
            add_selected_case(selected, row, "hard_negative", cfg, answer_user_days)

    cases = list(selected.values())
    max_cases = int(sampling.get("max_main_user_days") or 0)
    limited = False
    if max_cases > 0 and len(cases) > max_cases:
        positives = [case for case in cases if case["label"] == "malicious"]
        negatives = [case for case in cases if case["label"] != "malicious"]
        reason_priority = {
            "temporal_near_negative": 0,
            "same_user_negative": 1,
            "similar_user_negative": 2,
            "hard_negative": 3,
        }

        def neg_key(case: dict[str, Any]) -> tuple[int, float, int]:
            priority = min(reason_priority.get(reason, 9) for reason in case["sampling_reasons"])
            return (
                priority,
                -float(case["features"].get("hard_score") or 0.0),
                stable_int(case["case_id"], int(sampling["random_seed"])),
            )

        remaining = max(0, max_cases - len(positives))
        negatives = sorted(negatives, key=neg_key)[:remaining]
        cases = positives + negatives
        limited = True

    cases.sort(key=lambda item: (item["date"], item["user"]))
    reason_counts = Counter(reason for case in cases for reason in case["sampling_reasons"])
    label_counts = Counter(case["label"] for case in cases)
    volume_counts = Counter(str(case["features"].get("volume_bin") or "") for case in cases)
    report = {
        "num_selected_user_days": len(cases),
        "limited_by_max_main_user_days": limited,
        "label_counts": dict(label_counts),
        "sampling_reason_counts": dict(reason_counts),
        "volume_bin_counts": dict(volume_counts),
        "num_positive_user_days_with_raw_events": len(positive_rows),
        "num_positive_user_days_from_answers": len(answer_user_days),
        "num_positive_user_days_missing_raw_events": max(0, len(answer_user_days) - len(positive_rows)),
    }
    return cases, report


def assign_splits(cases: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    split_cfg = cfg["split"]
    seed = int(cfg["sampling"]["random_seed"])
    test_size = float(split_cfg["test_size"])
    target_test = max(1, round(len(cases) * test_size)) if cases else 0
    target_test_pos = round(sum(case["label"] == "malicious" for case in cases) * test_size)

    if bool(split_cfg.get("user_disjoint", True)):
        by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            by_user[case["user"]].append(case)
        user_stats = []
        for user, rows in by_user.items():
            pos = sum(row["label"] == "malicious" for row in rows)
            user_stats.append(
                {
                    "user": user,
                    "rows": rows,
                    "count": len(rows),
                    "positive_count": pos,
                    "hash": stable_int(user, seed),
                }
            )
        positive_users = sorted([item for item in user_stats if item["positive_count"] > 0], key=lambda item: item["hash"])
        benign_users = sorted([item for item in user_stats if item["positive_count"] == 0], key=lambda item: item["hash"])

        test_users: set[str] = set()
        test_count = 0
        test_pos = 0

        for item in positive_users:
            if len(positive_users) > 1 and test_pos < max(1, target_test_pos):
                test_users.add(item["user"])
                test_count += item["count"]
                test_pos += item["positive_count"]

        for item in benign_users:
            if test_count >= target_test:
                break
            test_users.add(item["user"])
            test_count += item["count"]

        if not test_users and user_stats:
            item = min(user_stats, key=lambda row: (abs(row["count"] - target_test), row["hash"]))
            test_users.add(item["user"])

        for case in cases:
            case["split"] = "test" if case["user"] in test_users else "train"
    else:
        positives = [case for case in cases if case["label"] == "malicious"]
        negatives = [case for case in cases if case["label"] != "malicious"]
        positives.sort(key=lambda item: stable_int(item["case_id"], seed))
        negatives.sort(key=lambda item: stable_int(item["case_id"], seed))
        test_ids = set()
        for group in (positives, negatives):
            take = round(len(group) * test_size)
            for case in group[:take]:
                test_ids.add(case["case_id"])
        if not test_ids and cases:
            test_ids.add(min(cases, key=lambda item: stable_int(item["case_id"], seed))["case_id"])
        for case in cases:
            case["split"] = "test" if case["case_id"] in test_ids else "train"

    split_counts = {
        split: {
            "total": sum(case["split"] == split for case in cases),
            "malicious": sum(case["split"] == split and case["label"] == "malicious" for case in cases),
            "benign": sum(case["split"] == split and case["label"] == "benign" for case in cases),
            "users": len({case["user"] for case in cases if case["split"] == split}),
        }
        for split in ("train", "test")
    }
    return {
        "target_test_size": target_test,
        "target_test_positive": target_test_pos,
        "user_disjoint": bool(split_cfg.get("user_disjoint", True)),
        "split_counts": split_counts,
        "user_overlap_count": len(
            {case["user"] for case in cases if case["split"] == "train"}
            & {case["user"] for case in cases if case["split"] == "test"}
        ),
    }


def create_selected_case_table(con, cases: list[dict[str, Any]]) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE selected_case_keys (
            case_id VARCHAR,
            user VARCHAR,
            day DATE
        )
        """
    )
    rows = [(case["case_id"], case["user"], case["date"]) for case in cases]
    if rows:
        con.executemany("INSERT INTO selected_case_keys VALUES (?, ?, CAST(? AS DATE))", rows)


def fetch_context_summaries(con, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    context_days = int(cfg["dataset"].get("context_days", 7))
    rows = fetch_dicts(
        con,
        f"""
        SELECT
            s.case_id,
            COUNT(f.day)::BIGINT AS active_days,
            COALESCE(SUM(f.total_events), 0)::BIGINT AS total_events,
            COALESCE(SUM(f.logon_count), 0)::BIGINT AS logon_count,
            COALESCE(SUM(f.device_count), 0)::BIGINT AS device_count,
            COALESCE(SUM(f.file_count), 0)::BIGINT AS file_count,
            COALESCE(SUM(f.email_count), 0)::BIGINT AS email_count,
            COALESCE(SUM(f.http_count), 0)::BIGINT AS http_count,
            COALESCE(SUM(f.device_connect_count), 0)::BIGINT AS device_connect_count,
            COALESCE(SUM(f.off_hours_count), 0)::BIGINT AS off_hours_count,
            COALESCE(MAX(f.hard_score), 0.0)::DOUBLE AS max_daily_hard_score
        FROM selected_case_keys s
        LEFT JOIN user_day_features f
          ON f.user = s.user
         AND f.day >= s.day - INTERVAL '{context_days} days'
         AND f.day < s.day
        GROUP BY s.case_id
        """,
    )
    context: dict[str, dict[str, Any]] = {}
    for row in rows:
        context[row["case_id"]] = {
            "window_days": context_days,
            "active_days": int(row["active_days"] or 0),
            "event_counts": {
                "total": int(row["total_events"] or 0),
                "logon": int(row["logon_count"] or 0),
                "device": int(row["device_count"] or 0),
                "file": int(row["file_count"] or 0),
                "email": int(row["email_count"] or 0),
                "http": int(row["http_count"] or 0),
            },
            "signal_counts": {
                "device_connect": int(row["device_connect_count"] or 0),
                "off_hours": int(row["off_hours_count"] or 0),
                "max_daily_hard_score": float(row["max_daily_hard_score"] or 0.0),
            },
        }
    return context


def fetch_selected_events(con) -> dict[str, list[dict[str, Any]]]:
    rows = fetch_dicts(
        con,
        """
        SELECT
            s.case_id,
            e.source,
            e.event_id,
            e.ts,
            e.pc,
            e.activity,
            e.filename,
            e.url,
            e.email_to,
            e.email_cc,
            e.email_bcc,
            e.email_from,
            e.size,
            e.attachments,
            e.content
        FROM selected_case_keys s
        INNER JOIN all_events e
          ON e.user = s.user
         AND CAST(e.ts AS DATE) = s.day
        WHERE e.ts IS NOT NULL
        ORDER BY s.case_id, e.ts, e.event_id
        """,
    )
    events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = str(row.pop("case_id"))
        events[case_id].append(row)
    return events


def load_ldap_profiles(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw_dir = resolve_path(cfg, cfg["dataset"]["raw_dir"])
    ldap_dir = raw_dir / "LDAP"
    profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not ldap_dir.exists():
        return profiles

    for path in sorted(ldap_dir.glob("*.csv")):
        profile_month = path.stem
        try:
            datetime.strptime(profile_month + "-01", "%Y-%m-%d")
        except ValueError:
            continue
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                user = str(row.get("user_id") or "").strip()
                if not user:
                    continue
                profiles[user].append(
                    {
                        "profile_month": profile_month,
                        "employee_name": row.get("employee_name"),
                        "email": row.get("email"),
                        "role": row.get("role"),
                        "business_unit": row.get("business_unit"),
                        "functional_unit": row.get("functional_unit"),
                        "department": row.get("department"),
                        "team": row.get("team"),
                        "supervisor": row.get("supervisor"),
                    }
                )
    for user in profiles:
        profiles[user].sort(key=lambda item: str(item["profile_month"]))
    return profiles


def load_psychometric_profiles(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_dir = resolve_path(cfg, cfg["dataset"]["raw_dir"])
    path = raw_dir / "psychometric.csv"
    profiles: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return profiles

    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            user = str(row.get("user_id") or "").strip()
            if not user:
                continue
            profiles[user] = {
                "employee_name": row.get("employee_name"),
                "O": row.get("O"),
                "C": row.get("C"),
                "E": row.get("E"),
                "A": row.get("A"),
                "N": row.get("N"),
            }
    return profiles


def latest_ldap_profile_for_day(
    profiles: list[dict[str, Any]],
    target_day: str,
) -> dict[str, Any] | None:
    target_month = target_day[:7]
    eligible = [profile for profile in profiles if str(profile.get("profile_month") or "") <= target_month]
    if not eligible:
        return None
    return eligible[-1]


def build_profile_events_by_case(cfg: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    enabled_sources = {str(value).lower() for value in cfg["dataset"].get("profile_sources", [])}
    ldap_profiles = load_ldap_profiles(cfg) if "ldap" in enabled_sources else {}
    psychometric_profiles = load_psychometric_profiles(cfg) if "psychometric" in enabled_sources else {}

    events_by_case: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        case_events: list[dict[str, Any]] = []
        profile_ts = datetime.strptime(case["date"] + " 00:00:00", "%Y-%m-%d %H:%M:%S")
        user = case["user"]

        ldap_profile = latest_ldap_profile_for_day(ldap_profiles.get(user, []), case["date"])
        if ldap_profile:
            case_events.append(
                {
                    "source": "ldap",
                    "event_id": f"ldap:{user}:{ldap_profile['profile_month']}",
                    "ts": profile_ts,
                    "pc": None,
                    "activity": "LDAP profile snapshot",
                    **ldap_profile,
                }
            )

        psychometric_profile = psychometric_profiles.get(user)
        if psychometric_profile:
            case_events.append(
                {
                    "source": "psychometric",
                    "event_id": f"psychometric:{user}",
                    "ts": profile_ts,
                    "pc": None,
                    "activity": "Psychometric profile",
                    **psychometric_profile,
                }
            )

        events_by_case[case["case_id"]] = case_events
    return events_by_case


def source_specific_event(row: dict[str, Any], cfg: dict[str, Any], index: int) -> dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    internal_email_domains = [str(value) for value in dataset_cfg.get("internal_email_domains", [])]
    internal_http_domains = [str(value) for value in dataset_cfg.get("internal_http_domains", [])]
    suspicious_keywords = [str(value).lower() for value in dataset_cfg.get("suspicious_http_keywords", [])]
    source = str(row.get("source") or "")
    ts = row.get("ts")
    timestamp = ts.isoformat(sep=" ") if hasattr(ts, "isoformat") else str(ts)
    activity = "" if row.get("activity") is None else str(row.get("activity"))
    event = {
        "event_id": row.get("event_id"),
        "timestamp": timestamp,
        "source": source,
        "pc": row.get("pc"),
        "activity": activity,
    }
    hour = int(timestamp[11:13]) if len(timestamp) >= 13 and timestamp[11:13].isdigit() else 12
    off_start = int(cfg["sampling"]["off_hours_start"])
    off_end = int(cfg["sampling"]["off_hours_end"])
    event["off_hours"] = bool(hour < off_start or hour >= off_end)

    if source == "device":
        event["is_usb_activity"] = activity.lower() in {"connect", "disconnect"}
    elif source == "file":
        filename = row.get("filename")
        event["filename"] = filename
        if filename:
            suffix = Path(str(filename)).suffix.lower().lstrip(".")
            event["file_extension"] = suffix
        preview = compact_text(row.get("content"), int(dataset_cfg["content_preview_chars"]))
        if preview:
            event["content_preview"] = preview
    elif source == "email":
        recipients = ";".join(
            str(value or "")
            for value in (row.get("email_to"), row.get("email_cc"), row.get("email_bcc"))
        )
        event["from"] = row.get("email_from")
        event["to"] = row.get("email_to")
        event["cc"] = row.get("email_cc")
        event["bcc"] = row.get("email_bcc")
        event["recipient_domains"] = sorted(set(email_domains(recipients)))
        event["external_email"] = is_external_email(recipients, internal_email_domains)
        if row.get("size") not in {None, ""}:
            event["size"] = row.get("size")
        if row.get("attachments") not in {None, ""}:
            event["attachments"] = row.get("attachments")
        preview = compact_text(row.get("content"), int(dataset_cfg["content_preview_chars"]))
        if preview:
            event["content_preview"] = preview
    elif source == "http":
        url = row.get("url")
        domain = url_domain(url)
        lower_url = str(url or "").lower()
        lower_content = str(row.get("content") or "").lower()
        event["url"] = url
        event["domain"] = domain
        event["external_http"] = is_external_url(url, internal_http_domains)
        event["suspicious_http_keyword"] = any(
            keyword in lower_url or keyword in lower_content for keyword in suspicious_keywords
        )
        preview = compact_text(row.get("content"), int(dataset_cfg["content_preview_chars"]))
        if preview:
            event["content_preview"] = preview
    elif source == "logon":
        event["is_logon"] = activity.lower() == "logon"
        event["is_logoff"] = activity.lower() == "logoff"
    elif source == "ldap":
        event["is_profile_record"] = True
        event["off_hours"] = False
        event["profile_month"] = row.get("profile_month")
        event["employee_name"] = row.get("employee_name")
        event["email"] = row.get("email")
        event["role"] = row.get("role")
        event["business_unit"] = row.get("business_unit")
        event["functional_unit"] = row.get("functional_unit")
        event["department"] = row.get("department")
        event["team"] = row.get("team")
        event["supervisor"] = row.get("supervisor")
    elif source == "psychometric":
        event["is_profile_record"] = True
        event["off_hours"] = False
        event["employee_name"] = row.get("employee_name")
        event["scores"] = {
            "O": row.get("O"),
            "C": row.get("C"),
            "E": row.get("E"),
            "A": row.get("A"),
            "N": row.get("N"),
        }

    event["_original_index"] = index
    event["_priority"] = event_priority(event)
    return event


def event_priority(event: dict[str, Any]) -> float:
    source = str(event.get("source") or "")
    score = 0.0
    if source == "device":
        score += 80.0
    if source == "file":
        score += 45.0
    if source == "email" and event.get("external_email"):
        score += 50.0
    if source == "http" and event.get("external_http"):
        score += 35.0
    if event.get("suspicious_http_keyword"):
        score += 70.0
    if source == "logon" and event.get("off_hours"):
        score += 60.0
    if source in {"ldap", "psychometric"}:
        score += 15.0
    if event.get("off_hours"):
        score += 10.0
    return score


def compute_event_counts(events: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    source_counts = Counter(str(event.get("source") or "") for event in events)
    pc_values = {str(event.get("pc")) for event in events if event.get("pc")}
    total = len(events)
    bins = [int(value) for value in cfg["sampling"].get("volume_bins", [20, 100, 300])]
    return {
        "total": total,
        "by_source": dict(source_counts),
        "volume_bin": volume_class(total, bins),
        "distinct_pc_count": len(pc_values),
        "has_device": source_counts.get("device", 0) > 0,
        "has_file": source_counts.get("file", 0) > 0,
        "has_ldap": source_counts.get("ldap", 0) > 0,
        "has_psychometric": source_counts.get("psychometric", 0) > 0,
        "has_profile": source_counts.get("ldap", 0) > 0 or source_counts.get("psychometric", 0) > 0,
        "has_external_email": any(event.get("external_email") for event in events),
        "has_external_http": any(event.get("external_http") for event in events),
        "has_off_hours": any(event.get("off_hours") for event in events),
        "has_off_hours_logon": any(event.get("source") == "logon" and event.get("off_hours") for event in events),
        "has_suspicious_http": any(event.get("suspicious_http_keyword") for event in events),
    }


def truncate_events(events: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_events = int(cfg["truncation"]["max_events_per_user_day"])
    keep_start = int(cfg["truncation"]["keep_start_events"])
    keep_end = int(cfg["truncation"]["keep_end_events"])
    original_count = len(events)
    if original_count <= max_events:
        clean = [{k: v for k, v in event.items() if not k.startswith("_")} for event in events]
        return clean, {
            "original_event_count": original_count,
            "kept_event_count": original_count,
            "omitted_event_count": 0,
            "strategy": "none",
        }

    keep: set[int] = {
        idx
        for idx, event in enumerate(events)
        if str(event.get("source") or "") in {"ldap", "psychometric"}
    }
    keep.update(range(min(keep_start, original_count)))
    keep.update(range(max(0, original_count - keep_end), original_count))
    if len(keep) > max_events:
        ordered = sorted(keep)
        left = ordered[: max_events // 2]
        right = ordered[-(max_events - len(left)) :]
        keep = set(left + right)
    remaining_slots = max(0, max_events - len(keep))
    ranked = sorted(
        [idx for idx in range(original_count) if idx not in keep],
        key=lambda idx: (-float(events[idx].get("_priority") or 0.0), int(events[idx].get("_original_index") or idx)),
    )
    keep.update(ranked[:remaining_slots])
    truncated = [
        {k: v for k, v in events[idx].items() if not k.startswith("_")}
        for idx in sorted(keep, key=lambda item: int(events[item].get("_original_index") or item))
    ]
    return truncated, {
        "original_event_count": original_count,
        "kept_event_count": len(truncated),
        "omitted_event_count": original_count - len(truncated),
        "strategy": "priority_device_file_external_offhours_boundaries",
    }


def build_samples(
    cfg: dict[str, Any],
    cases: list[dict[str, Any]],
    answer_user_days: dict[tuple[str, str], dict[str, Any]],
    raw_events_by_case: dict[str, list[dict[str, Any]]],
    profile_events_by_case: dict[str, list[dict[str, Any]]],
    context_by_case: dict[str, dict[str, Any]],
    *,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    truncation_rows: list[dict[str, Any]] = []
    for case in cases:
        raw_rows = profile_events_by_case.get(case["case_id"], []) + raw_events_by_case.get(case["case_id"], [])
        formatted_events = [
            source_specific_event(row, cfg, index)
            for index, row in enumerate(raw_rows)
        ]
        event_counts = compute_event_counts(formatted_events, cfg)
        events, truncation = truncate_events(formatted_events, cfg)
        answer_payload = answer_user_days.get((case["user"], case["date"]), {})
        metadata = {
            "variant": variant,
            "split": case["split"],
            "sampling_reasons": case["sampling_reasons"],
            "selection_features": case["features"],
            "event_counts": event_counts,
            "events_in_payload": len(events),
            "truncation": truncation,
        }
        if case["label"] == "malicious":
            metadata["scenario"] = answer_payload.get("scenario", [])
            metadata["malicious_event_count"] = int(answer_payload.get("malicious_event_count", 0))
            metadata["answer_reference"] = answer_payload.get("answer_reference", [])

        sample = {
            "case_id": case["case_id"],
            "user": case["user"],
            "date": case["date"],
            "events": events,
            "label": case["label"],
            "metadata": metadata,
        }
        if variant == "with_context":
            sample["context"] = {
                "type": "previous_7_days_summary",
                "includes_future_events": False,
                "summary": context_by_case.get(case["case_id"], {}),
            }
        samples.append(sample)
        truncation_rows.append(
            {
                "case_id": case["case_id"],
                "split": case["split"],
                "label": case["label"],
                **truncation,
            }
        )

    total_original = sum(row["original_event_count"] for row in truncation_rows)
    total_kept = sum(row["kept_event_count"] for row in truncation_rows)
    truncated_cases = [row for row in truncation_rows if row["omitted_event_count"] > 0]
    report = {
        "variant": variant,
        "num_cases": len(samples),
        "num_truncated_cases": len(truncated_cases),
        "total_original_events": total_original,
        "total_kept_events": total_kept,
        "total_omitted_events": total_original - total_kept,
        "max_omitted_events_single_case": max((row["omitted_event_count"] for row in truncation_rows), default=0),
        "most_truncated_cases": sorted(
            truncated_cases,
            key=lambda row: row["omitted_event_count"],
            reverse=True,
        )[:25],
    }
    return samples, report


def write_variant_outputs(
    base_output_dir: Path,
    variant: str,
    samples: list[dict[str, Any]],
    reports: dict[str, Any],
) -> dict[str, str]:
    out_dir = base_output_dir / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    main_full = out_dir / "main_full.jsonl"
    main_prompt = out_dir / "main_prompt.jsonl"
    train_full = out_dir / "train_full.jsonl"
    train_prompt = out_dir / "train_prompt.jsonl"
    test_full = out_dir / "test_full.jsonl"
    test_prompt = out_dir / "test_prompt.jsonl"

    train = [sample for sample in samples if sample["metadata"]["split"] == "train"]
    test = [sample for sample in samples if sample["metadata"]["split"] == "test"]
    write_jsonl(main_full, samples)
    write_jsonl(main_prompt, [sanitize_for_prompt(sample) for sample in samples])
    write_jsonl(train_full, train)
    write_jsonl(train_prompt, [sanitize_for_prompt(sample) for sample in train])
    write_jsonl(test_full, test)
    write_jsonl(test_prompt, [sanitize_for_prompt(sample) for sample in test])

    write_json(out_dir / "subset_manifest.json", reports["manifest"])
    write_json(out_dir / "sampling_report.json", reports["sampling"])
    write_json(out_dir / "leakage_report.json", reports["leakage"])
    write_json(out_dir / "truncation_report.json", reports["truncation"])

    return {
        "main_full": str(main_full),
        "main_prompt": str(main_prompt),
        "train_full": str(train_full),
        "train_prompt": str(train_prompt),
        "test_full": str(test_full),
        "test_prompt": str(test_prompt),
        "subset_manifest": str(out_dir / "subset_manifest.json"),
        "sampling_report": str(out_dir / "sampling_report.json"),
        "leakage_report": str(out_dir / "leakage_report.json"),
        "truncation_report": str(out_dir / "truncation_report.json"),
    }


def leakage_report(cases: list[dict[str, Any]], samples: list[dict[str, Any]], split_report: dict[str, Any]) -> dict[str, Any]:
    train_keys = {(case["user"], case["date"]) for case in cases if case["split"] == "train"}
    test_keys = {(case["user"], case["date"]) for case in cases if case["split"] == "test"}
    train_users = {case["user"] for case in cases if case["split"] == "train"}
    test_users = {case["user"] for case in cases if case["split"] == "test"}
    case_ids = [case["case_id"] for case in cases]
    prompt_has_label = any("label" in sanitize_for_prompt(sample) for sample in samples)
    prompt_has_answer_ref = any(
        "answer_reference" in sanitize_for_prompt(sample).get("metadata", {})
        for sample in samples
    )
    return {
        "duplicate_case_ids": len(case_ids) - len(set(case_ids)),
        "user_day_overlap_between_train_test": sorted(f"{user}|{day}" for user, day in (train_keys & test_keys)),
        "user_overlap_count": len(train_users & test_users),
        "user_overlap_sample": sorted(train_users & test_users)[:50],
        "user_disjoint_requested": bool(split_report["user_disjoint"]),
        "prompt_files_include_label": prompt_has_label,
        "prompt_files_include_answer_reference": prompt_has_answer_ref,
        "test_prompt_labels_removed": True,
        "answer_files_used_only_for_labels_and_reports": True,
        "with_context_uses_future_events": False,
        "notes": [
            "Splits are assigned after the main subset is built.",
            "No raw answer rows or test labels are written to *_prompt.jsonl files.",
            "with_context uses aggregate summaries from days < target date only.",
        ],
    }


def run(config_path: str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    answer_user_days, answer_meta = load_answer_user_days(cfg)
    con = connect_duckdb(cfg)
    create_event_views(con, cfg)
    create_answer_table(con, answer_user_days)
    create_feature_tables(con, cfg)
    cases, sampling_report = select_user_days(con, cfg, answer_user_days)
    split_report = assign_splits(cases, cfg)
    create_selected_case_table(con, cases)
    context_by_case = fetch_context_summaries(con, cfg)
    raw_events_by_case = fetch_selected_events(con)
    profile_events_by_case = build_profile_events_by_case(cfg, cases)

    base_output_dir = resolve_path(cfg, cfg["dataset"]["output_dir"])
    base_output_dir.mkdir(parents=True, exist_ok=True)
    variants = list(cfg["dataset"].get("variants") or ["day_only"])
    output_files: dict[str, dict[str, str]] = {}
    manifest_base = {
        "dataset_name": cfg["dataset"]["name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": cfg["project_root"],
        "raw_dir": str(resolve_path(cfg, cfg["dataset"]["raw_dir"])),
        "output_dir": str(base_output_dir),
        "variants": variants,
        "answer_files": answer_meta,
        "split": split_report,
        "num_cases": len(cases),
        "label_counts": dict(Counter(case["label"] for case in cases)),
        "config": cfg,
    }

    top_level_manifests: dict[str, Any] = {}
    for variant in variants:
        samples, trunc_report = build_samples(
            cfg,
            cases,
            answer_user_days,
            raw_events_by_case,
            profile_events_by_case,
            context_by_case,
            variant=variant,
        )
        leak_report = leakage_report(cases, samples, split_report)
        manifest = dict(manifest_base)
        manifest["variant"] = variant
        reports = {
            "manifest": manifest,
            "sampling": {**sampling_report, "split": split_report},
            "leakage": leak_report,
            "truncation": trunc_report,
        }
        output_files[variant] = write_variant_outputs(base_output_dir, variant, samples, reports)
        top_level_manifests[variant] = {
            "files": output_files[variant],
            "num_cases": len(samples),
            "num_train": sum(sample["metadata"]["split"] == "train" for sample in samples),
            "num_test": sum(sample["metadata"]["split"] == "test" for sample in samples),
        }

    root_manifest = dict(manifest_base)
    root_manifest["variant_outputs"] = top_level_manifests
    write_json(base_output_dir / "subset_manifest.json", root_manifest)
    return root_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build standalone CERT user-day JSONL subsets for local LLM evaluation.")
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config. Defaults are used when omitted.")
    args = parser.parse_args(argv)
    manifest = run(args.config)
    print(f"Wrote CERT LLM subset manifest: {manifest['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
