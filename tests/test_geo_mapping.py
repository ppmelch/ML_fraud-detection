"""Tests for StateGeoMapper: US join validation, top-level id, no silent drops."""

import json

import pytest

from backend.src.metrics.geo_mapping import StateGeoMapper


@pytest.fixture
def small_geojson(tmp_path):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "CA", "properties": {"name": "California"}, "geometry": None},
            {"type": "Feature", "id": "TX", "properties": {"name": "Texas"}, "geometry": None},
            {"type": "Feature", "id": "NY", "properties": {"name": "New York"}, "geometry": None},
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
            {"type": "Feature", "id": "CA", "properties": {"name": "California"}, "geometry": None},
            {"type": "Feature", "id": "CA2", "properties": {"name": "California"}, "geometry": None},
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
        assert mapper.to_geojson_name("California") == "California"

    def test_unmapped_returns_none(self, small_geojson):
        mapper = StateGeoMapper(geojson_path=small_geojson, name_map={})
        assert mapper.to_geojson_name("Atlantis") is None

    def test_id_read_from_feature_top_level(self, small_geojson):
        mapper = StateGeoMapper(geojson_path=small_geojson, name_map={})
        assert mapper.geojson_id("California") == "CA"
        assert mapper.geojson_id("Atlantis") is None


class TestAttachAndValidate:
    def test_attach_marks_unmatched_without_dropping(self, small_geojson):
        mapper = StateGeoMapper(geojson_path=small_geojson, name_map={})
        records = [
            {"state": "California", "anomaly_rate": 0.05},
            {"state": "NotOnMap", "anomaly_rate": 0.03},
        ]
        enriched = mapper.attach(records)

        assert len(enriched) == 2
        assert enriched[0]["matched"] is True
        assert enriched[0]["geojson_id"] == "CA"
        assert enriched[1]["matched"] is False
        assert enriched[1]["anomaly_rate"] == 0.03

    def test_validate_reports_unmatched(self, small_geojson):
        mapper = StateGeoMapper(geojson_path=small_geojson, name_map={})
        report = mapper.validate([{"state": "California"}, {"state": "Ghost"}])

        assert report["ok"] is False
        assert report["unmatched_dataset_states"] == ["Ghost"]

    def test_validate_ok_when_all_match(self, small_geojson):
        mapper = StateGeoMapper(geojson_path=small_geojson, name_map={})
        report = mapper.validate([{"state": "California"}, {"state": "Texas"}])

        assert report["ok"] is True


class TestRealGeoJSON:
    def test_every_city_to_state_value_matches_a_feature(self):
        from backend.src.modeling.config import CITY_TO_STATE, GEOJSON_PATH

        if not GEOJSON_PATH.exists():
            pytest.skip("GeoJSON not present")

        mapper = StateGeoMapper()
        report = mapper.validate(
            [{"state": s} for s in sorted(set(CITY_TO_STATE.values()))]
        )

        assert report["unmatched_dataset_states"] == []
        assert report["ok"] is True
