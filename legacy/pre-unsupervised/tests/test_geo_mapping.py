"""Tests for StateGeoMapper: join validation, explicit renames, no silent drops."""

import json

import pytest

from backend.src.metrics.geo_mapping import StateGeoMapper


@pytest.fixture
def small_geojson(tmp_path):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"id": "IN1", "name": "Kerala"}, "geometry": None},
            {"type": "Feature", "properties": {"id": "IN2", "name": "Orissa"}, "geometry": None},
            {"type": "Feature", "properties": {"id": "IN3", "name": "Unmapped State"}, "geometry": None},
        ],
    }
    path = tmp_path / "geo.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def duplicate_geojson(tmp_path):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"id": "IN1", "name": "Kerala"}, "geometry": None},
            {"type": "Feature", "properties": {"id": "IN2", "name": "Kerala"}, "geometry": None},
        ],
    }
    path = tmp_path / "geo.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestConstruction:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            StateGeoMapper(geojson_path=tmp_path / "missing.json")

    def test_duplicate_names_raise(self, duplicate_geojson):
        with pytest.raises(ValueError, match="Duplicate"):
            StateGeoMapper(geojson_path=duplicate_geojson)

    def test_not_a_feature_collection_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"type": "Point"}), encoding="utf-8")

        with pytest.raises(ValueError, match="FeatureCollection"):
            StateGeoMapper(geojson_path=path)


class TestNameMapping:
    def test_direct_match(self, small_geojson):
        mapper = StateGeoMapper(geojson_path=small_geojson, name_map={})
        assert mapper.to_geojson_name("Kerala") == "Kerala"

    def test_explicit_rename_applied(self, small_geojson):
        mapper = StateGeoMapper(
            geojson_path=small_geojson, name_map={"Odisha": "Orissa"}
        )
        assert mapper.to_geojson_name("Odisha") == "Orissa"

    def test_unmapped_state_returns_none(self, small_geojson):
        mapper = StateGeoMapper(geojson_path=small_geojson, name_map={})
        assert mapper.to_geojson_name("Nonexistent State") is None

    def test_geojson_id_lookup(self, small_geojson):
        mapper = StateGeoMapper(geojson_path=small_geojson, name_map={})
        assert mapper.geojson_id("Kerala") == "IN1"
        assert mapper.geojson_id("Nonexistent") is None


class TestAttachAndValidate:
    def test_attach_marks_unmatched_without_dropping(self, small_geojson):
        mapper = StateGeoMapper(geojson_path=small_geojson, name_map={})
        records = [
            {"state": "Kerala", "fraud_rate": 0.05},
            {"state": "NotOnMap", "fraud_rate": 0.03},
        ]
        enriched = mapper.attach(records)

        assert len(enriched) == 2
        assert enriched[0]["matched"] is True
        assert enriched[1]["matched"] is False
        assert enriched[1]["fraud_rate"] == 0.03  # original metrics preserved

    def test_validate_reports_unmatched_and_renames(self, small_geojson):
        mapper = StateGeoMapper(
            geojson_path=small_geojson, name_map={"Odisha": "Orissa"}
        )
        records = [{"state": "Kerala"}, {"state": "Odisha"}, {"state": "Ghost State"}]

        report = mapper.validate(records)

        assert report["ok"] is False
        assert report["unmatched_dataset_states"] == ["Ghost State"]
        assert report["renames_applied"] == {"Odisha": "Orissa"}
        assert "Unmapped State" in report["geojson_states_without_data"]

    def test_validate_ok_when_everything_matches(self, small_geojson):
        mapper = StateGeoMapper(
            geojson_path=small_geojson, name_map={"Odisha": "Orissa"}
        )
        records = [{"state": "Kerala"}, {"state": "Odisha"}]

        report = mapper.validate(records)

        assert report["ok"] is True
        assert report["unmatched_dataset_states"] == []

    def test_duplicate_state_in_records_is_reported(self, small_geojson):
        mapper = StateGeoMapper(geojson_path=small_geojson, name_map={})
        records = [{"state": "Kerala"}, {"state": "Kerala"}]

        report = mapper.validate(records)

        assert report["duplicate_states"] == ["Kerala"]
        assert report["ok"] is False


class TestRealGeoJSON:
    def test_real_dataset_states_all_matched(self, full_dataset):
        from backend.src.modeling.config import GEOJSON_PATH

        if not GEOJSON_PATH.exists():
            pytest.skip(f"GeoJSON not found at {GEOJSON_PATH}")

        mapper = StateGeoMapper()
        records = [{"state": s} for s in full_dataset["State"].unique()]

        report = mapper.validate(records)

        assert report["unmatched_dataset_states"] == []
        assert report["duplicate_states"] == []
