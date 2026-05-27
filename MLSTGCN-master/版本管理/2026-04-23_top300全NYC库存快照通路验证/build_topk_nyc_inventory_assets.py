import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_trip_files(trip_dir, trip_pattern):
    paths = sorted(glob.glob(os.path.join(trip_dir, trip_pattern)))
    if not paths:
        raise FileNotFoundError("No trip files matched under %s with pattern %s" % (trip_dir, trip_pattern))
    return paths


def list_snapshot_files(snapshot_root):
    morning = sorted(glob.glob(os.path.join(snapshot_root, "morning", "citibike_cn_*.csv")))
    evening = sorted(glob.glob(os.path.join(snapshot_root, "evening", "citibike_cn_*.csv")))
    if not morning and not evening:
        raise FileNotFoundError("No snapshot files found under %s" % snapshot_root)
    return morning, evening


def normalize_float(raw_value):
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_int(raw_value):
    value = normalize_float(raw_value)
    if value is None:
        return None
    return int(round(value))


def parse_snapshot_station_catalog(snapshot_files):
    station_catalog = {}
    for path in snapshot_files:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                station_name = (row.get("站点名称") or "").strip()
                if not station_name:
                    continue

                entry = station_catalog.setdefault(
                    station_name,
                    {
                        "snapshot_id_counter": Counter(),
                        "capacity_counter": Counter(),
                        "lat_counter": Counter(),
                        "lon_counter": Counter(),
                    },
                )
                snapshot_station_id = (row.get("站点ID") or "").strip()
                if snapshot_station_id:
                    entry["snapshot_id_counter"][snapshot_station_id] += 1
                capacity = normalize_int(row.get("总桩数"))
                if capacity is not None:
                    entry["capacity_counter"][capacity] += 1
                lat = normalize_float(row.get("纬度"))
                if lat is not None:
                    entry["lat_counter"][round(lat, 6)] += 1
                lon = normalize_float(row.get("经度"))
                if lon is not None:
                    entry["lon_counter"][round(lon, 6)] += 1
    return station_catalog


def most_common_or_none(counter_obj):
    if not counter_obj:
        return None
    return counter_obj.most_common(1)[0][0]


def first_pass_station_activity(trip_files, snapshot_station_names):
    station_activity = {}
    for trip_path in trip_files:
        with open(trip_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                start_station_name = (row.get("start_station_name") or "").strip()
                if start_station_name in snapshot_station_names:
                    start_entry = station_activity.setdefault(
                        start_station_name,
                        {
                            "order_id_counter": Counter(),
                            "start_count": 0,
                            "end_count": 0,
                            "lat_counter": Counter(),
                            "lon_counter": Counter(),
                        },
                    )
                    start_station_id = (row.get("start_station_id") or "").strip()
                    if start_station_id:
                        start_entry["order_id_counter"][start_station_id] += 1
                    start_entry["start_count"] += 1
                    lat = normalize_float(row.get("start_lat"))
                    if lat is not None:
                        start_entry["lat_counter"][round(lat, 6)] += 1
                    lon = normalize_float(row.get("start_lng"))
                    if lon is not None:
                        start_entry["lon_counter"][round(lon, 6)] += 1

                end_station_name = (row.get("end_station_name") or "").strip()
                if end_station_name in snapshot_station_names:
                    end_entry = station_activity.setdefault(
                        end_station_name,
                        {
                            "order_id_counter": Counter(),
                            "start_count": 0,
                            "end_count": 0,
                            "lat_counter": Counter(),
                            "lon_counter": Counter(),
                        },
                    )
                    end_station_id = (row.get("end_station_id") or "").strip()
                    if end_station_id:
                        end_entry["order_id_counter"][end_station_id] += 1
                    end_entry["end_count"] += 1
                    lat = normalize_float(row.get("end_lat"))
                    if lat is not None:
                        end_entry["lat_counter"][round(lat, 6)] += 1
                    lon = normalize_float(row.get("end_lng"))
                    if lon is not None:
                        end_entry["lon_counter"][round(lon, 6)] += 1
    return station_activity


def build_mapping_records(station_activity, snapshot_catalog, topk):
    ranked_names = []
    for station_name, activity in station_activity.items():
        total_flow = activity["start_count"] + activity["end_count"]
        if total_flow <= 0:
            continue
        ranked_names.append((station_name, total_flow, activity["start_count"], activity["end_count"]))
    ranked_names.sort(key=lambda item: (-item[1], -item[2], item[0]))

    selected_names = [item[0] for item in ranked_names[:topk]]
    records = []
    for node_id, station_name in enumerate(selected_names):
        activity = station_activity[station_name]
        snapshot_info = snapshot_catalog.get(station_name, {})
        order_ids = sorted(activity["order_id_counter"].keys())
        snapshot_ids = sorted(snapshot_info.get("snapshot_id_counter", {}).keys())
        record = {
            "Node_ID": node_id,
            "station_name": station_name,
            "order_station_ids": "|".join(order_ids),
            "snapshot_station_id": most_common_or_none(snapshot_info.get("snapshot_id_counter", Counter())),
            "snapshot_station_ids": "|".join(snapshot_ids),
            "lat": most_common_or_none(activity["lat_counter"]) or most_common_or_none(snapshot_info.get("lat_counter", Counter())),
            "lon": most_common_or_none(activity["lon_counter"]) or most_common_or_none(snapshot_info.get("lon_counter", Counter())),
            "capacity": most_common_or_none(snapshot_info.get("capacity_counter", Counter())),
            "total_start_count": activity["start_count"],
            "total_end_count": activity["end_count"],
            "total_flow_count": activity["start_count"] + activity["end_count"],
            "matched_snapshot": station_name in snapshot_catalog,
            "has_order_id_conflict": len(order_ids) > 1,
            "has_snapshot_id_conflict": len(snapshot_ids) > 1,
        }
        records.append(record)
    return records


def append_rows(path, rows, fieldnames, write_header=False):
    mode = "w" if write_header else "a"
    with open(path, mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def second_pass_daily_summaries(trip_files, selected_names, output_dir):
    selected_set = set(selected_names)
    daily_netflow_path = os.path.join(output_dir, "daily_netflow_topk.csv")
    daily_od_path = os.path.join(output_dir, "daily_od_pairs_topk.csv")
    global_od_path = os.path.join(output_dir, "global_od_pairs_topk.csv")
    daily_net_fields = ["date", "station_name", "daily_outflow", "daily_inflow", "daily_netflow"]
    daily_od_fields = ["date", "start_station_name", "end_station_name", "daily_trips"]
    global_od_counter = Counter()
    wrote_net_header = False
    wrote_od_header = False

    for trip_path in trip_files:
        station_day_counter = defaultdict(lambda: {"daily_outflow": 0, "daily_inflow": 0})
        day_od_counter = Counter()
        with open(trip_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                started_at = (row.get("started_at") or "").strip()
                if not started_at:
                    continue
                current_date = started_at[:10]
                start_station_name = (row.get("start_station_name") or "").strip()
                end_station_name = (row.get("end_station_name") or "").strip()

                if start_station_name in selected_set:
                    station_day_counter[(current_date, start_station_name)]["daily_outflow"] += 1
                if end_station_name in selected_set:
                    station_day_counter[(current_date, end_station_name)]["daily_inflow"] += 1
                if start_station_name in selected_set and end_station_name in selected_set:
                    day_od_counter[(current_date, start_station_name, end_station_name)] += 1
                    global_od_counter[(start_station_name, end_station_name)] += 1

        daily_net_rows = []
        for (current_date, station_name), counts in sorted(station_day_counter.items()):
            outflow = counts["daily_outflow"]
            inflow = counts["daily_inflow"]
            daily_net_rows.append(
                {
                    "date": current_date,
                    "station_name": station_name,
                    "daily_outflow": outflow,
                    "daily_inflow": inflow,
                    "daily_netflow": inflow - outflow,
                }
            )
        if daily_net_rows:
            append_rows(daily_netflow_path, daily_net_rows, daily_net_fields, write_header=not wrote_net_header)
            wrote_net_header = True

        daily_od_rows = []
        for (current_date, start_station_name, end_station_name), trip_count in sorted(day_od_counter.items()):
            daily_od_rows.append(
                {
                    "date": current_date,
                    "start_station_name": start_station_name,
                    "end_station_name": end_station_name,
                    "daily_trips": trip_count,
                }
            )
        if daily_od_rows:
            append_rows(daily_od_path, daily_od_rows, daily_od_fields, write_header=not wrote_od_header)
            wrote_od_header = True

    global_od_rows = [
        {
            "start_station_name": start_station_name,
            "end_station_name": end_station_name,
            "total_trips": trip_count,
        }
        for (start_station_name, end_station_name), trip_count in sorted(global_od_counter.items())
    ]
    append_rows(global_od_path, global_od_rows, ["start_station_name", "end_station_name", "total_trips"], write_header=True)
    return daily_netflow_path, daily_od_path, global_od_path


def build_snapshot_daily_features(selected_names, morning_files, evening_files, mapping_df, output_dir):
    selected_set = set(selected_names)
    rows_by_key = {}
    mapping_lookup = {
        row["station_name"]: {
            "snapshot_station_id": row.get("snapshot_station_id"),
            "capacity": row.get("capacity"),
            "lat": row.get("lat"),
            "lon": row.get("lon"),
        }
        for row in mapping_df.to_dict("records")
    }

    def consume_snapshot_file(path, period_prefix):
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                station_name = (row.get("站点名称") or "").strip()
                if station_name not in selected_set:
                    continue
                date_text = (row.get("抓取时间") or "").strip()[:10]
                if not date_text:
                    continue
                key = (date_text, station_name)
                base = rows_by_key.setdefault(
                    key,
                    {
                        "date": date_text,
                        "station_name": station_name,
                        "snapshot_station_id": mapping_lookup[station_name].get("snapshot_station_id"),
                        "capacity": mapping_lookup[station_name].get("capacity"),
                        "lat": mapping_lookup[station_name].get("lat"),
                        "lon": mapping_lookup[station_name].get("lon"),
                        "morning_bikes": None,
                        "morning_docks": None,
                        "evening_bikes": None,
                        "evening_docks": None,
                    },
                )
                base["snapshot_station_id"] = (row.get("站点ID") or "").strip() or base["snapshot_station_id"]
                capacity = normalize_int(row.get("总桩数"))
                if capacity is not None:
                    base["capacity"] = capacity
                lat = normalize_float(row.get("纬度"))
                if lat is not None:
                    base["lat"] = lat
                lon = normalize_float(row.get("经度"))
                if lon is not None:
                    base["lon"] = lon
                bikes = normalize_int(row.get("当前可用车辆"))
                docks = normalize_int(row.get("当前可用空位"))
                if period_prefix == "morning":
                    base["morning_bikes"] = bikes
                    base["morning_docks"] = docks
                else:
                    base["evening_bikes"] = bikes
                    base["evening_docks"] = docks

    for path in morning_files:
        consume_snapshot_file(path, "morning")
    for path in evening_files:
        consume_snapshot_file(path, "evening")

    output_path = os.path.join(output_dir, "snapshot_daily_features_topk.csv")
    df = pd.DataFrame(sorted(rows_by_key.values(), key=lambda item: (item["date"], item["station_name"])))
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def build_station_static_features(mapping_df, output_dir):
    output_path = os.path.join(output_dir, "station_static_features_topk.csv")
    static_df = mapping_df[
        ["station_name", "snapshot_station_id", "capacity", "lat", "lon"]
    ].drop_duplicates(subset=["station_name"], keep="first")
    static_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def write_summary(output_dir, summary):
    summary_path = os.path.join(output_dir, "build_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="二月份数据处理")
    parser.add_argument("--trip_subdir", default="纽约单车订单数据")
    parser.add_argument("--trip_pattern", default="20*-citibike-tripdata*.csv")
    parser.add_argument("--output_subdir", default="nyc_top300_inventory_validation")
    parser.add_argument("--topk", type=int, default=300)
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    trip_dir = os.path.join(source_dir, args.trip_subdir)
    output_dir = os.path.join(source_dir, args.output_subdir)
    ensure_dir(output_dir)

    trip_files = list_trip_files(trip_dir, args.trip_pattern)
    morning_files, evening_files = list_snapshot_files(source_dir)
    snapshot_catalog = parse_snapshot_station_catalog(morning_files + evening_files)
    station_activity = first_pass_station_activity(trip_files, set(snapshot_catalog.keys()))
    mapping_records = build_mapping_records(station_activity, snapshot_catalog, args.topk)
    mapping_df = pd.DataFrame(mapping_records)

    mapping_path = os.path.join(output_dir, "GNN_Node_Mapping_topk.csv")
    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")

    selected_names = mapping_df["station_name"].tolist()
    daily_netflow_path, daily_od_path, global_od_path = second_pass_daily_summaries(trip_files, selected_names, output_dir)
    snapshot_daily_feature_path = build_snapshot_daily_features(selected_names, morning_files, evening_files, mapping_df, output_dir)
    station_static_path = build_station_static_features(mapping_df, output_dir)

    summary = {
        "trip_files": len(trip_files),
        "snapshot_morning_files": len(morning_files),
        "snapshot_evening_files": len(evening_files),
        "snapshot_station_names": len(snapshot_catalog),
        "matched_station_names": len(station_activity),
        "topk": args.topk,
        "selected_stations": len(mapping_df),
        "order_id_conflict_stations": int(mapping_df["has_order_id_conflict"].sum()),
        "snapshot_id_conflict_stations": int(mapping_df["has_snapshot_id_conflict"].sum()),
        "mapping_file": mapping_path,
        "daily_netflow_file": daily_netflow_path,
        "daily_od_pairs_file": daily_od_path,
        "global_od_pairs_file": global_od_path,
        "snapshot_daily_feature_file": snapshot_daily_feature_path,
        "station_static_feature_file": station_static_path,
    }
    summary_path = write_summary(output_dir, summary)

    print("Built top-k NYC inventory assets.")
    print("Output dir:", output_dir)
    print("Trip files:", len(trip_files))
    print("Snapshot catalog stations:", len(snapshot_catalog))
    print("Matched stations:", len(station_activity))
    print("Selected stations:", len(mapping_df))
    print("Order-ID conflicts:", int(mapping_df["has_order_id_conflict"].sum()))
    print("Snapshot-ID conflicts:", int(mapping_df["has_snapshot_id_conflict"].sum()))
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
