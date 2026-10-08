import math

import pytest

from backend.agents.landcover_api import retrieve_landcover_evidence


class Value:
    def __init__(self, value):
        self.value = value

    def getInfo(self):
        return self.value


class FakeImage:
    def __init__(self, histogram):
        self.histogram = histogram

    def select(self, _band):
        return self

    def reduceRegion(self, **_kwargs):
        return Value({"Map": self.histogram} if self.histogram is not None else {})


class FakeEE:
    class Geometry:
        @staticmethod
        def Rectangle(coordinates):
            return coordinates

    class Reducer:
        @staticmethod
        def frequencyHistogram():
            return "frequencyHistogram"

    def __init__(self, histogram=None, init_error=None):
        self.histogram = histogram
        self.init_error = init_error
        self.initialized_project = None

    def Initialize(self, project):
        self.initialized_project = project
        if self.init_error:
            raise self.init_error

    def ImageCollection(self, _collection):
        image = FakeImage(self.histogram)
        return type("Collection", (), {"first": lambda _self: image})()


def test_successful_landcover_response_contract():
    result = retrieve_landcover_evidence(
        10.1234,
        76.1234,
        gee_project="test-project",
        ee_module=FakeEE({"50": 60, "40": 30, "10": 10}),
    )
    assert result["agent"] == "landcover"
    assert result["status"] == "success"
    assert result["source"] == "ESA WorldCover"
    assert result["location"] == {"latitude": 10.1234, "longitude": 76.1234}
    assert result["land_cover"] == {
        "primary_class": "Built-up",
        "class_code": 50,
        "coverage_percentage": 60.0,
    }
    assert result["evidence"]


def test_class_code_and_class_name_mapping():
    result = retrieve_landcover_evidence(
        10,
        76,
        gee_project="test-project",
        ee_module=FakeEE({"10": 1, "20": 2, "90": 3, "95": 4, "100": 5}),
    )
    classes = {item["class_code"]: item["class_name"] for item in result["classes"]}
    assert classes == {
        10: "Tree cover",
        20: "Shrubland",
        90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen",
    }


def test_coverage_statistics_are_calculated_from_pixel_counts():
    result = retrieve_landcover_evidence(
        10,
        76,
        gee_project="test-project",
        ee_module=FakeEE({"30": 1, "40": 3}),
    )
    assert result["classes"] == [
        {"class_code": 40, "class_name": "Cropland", "percentage": 75.0},
        {"class_code": 30, "class_name": "Grassland", "percentage": 25.0},
    ]


def test_spatial_scope_is_explicit_and_configurable():
    result = retrieve_landcover_evidence(
        10,
        76,
        radius_km=10,
        gee_project="test-project",
        ee_module=FakeEE({"80": 1}),
    )
    area = result["search_area"]
    assert area["radius_km"] == 10
    assert area["measurement"] == "WorldCover pixel-class distribution within the bounding box"
    assert area["bounding_box"]["west"] < 76 < area["bounding_box"]["east"]


@pytest.mark.parametrize("latitude,longitude", [(91, 76), (10, 181), (math.nan, 76)])
def test_invalid_coordinates_return_structured_failure(latitude, longitude):
    result = retrieve_landcover_evidence(latitude, longitude, gee_project="test-project", ee_module=FakeEE())
    assert result["status"] == "failed"
    assert result["metrics"] is None
    assert result["evidence"] == []


def test_invalid_spatial_parameter_returns_failure():
    result = retrieve_landcover_evidence(10, 76, radius_km=0, gee_project="test-project", ee_module=FakeEE())
    assert result["status"] == "failed"
    assert "radius_km" in result["error"]


def test_no_data_response_returns_failure():
    result = retrieve_landcover_evidence(10, 76, gee_project="test-project", ee_module=FakeEE(None))
    assert result["status"] == "failed"
    assert result["metrics"] is None
    assert "no class data" in result["error"]


def test_malformed_response_returns_failure():
    result = retrieve_landcover_evidence(10, 76, gee_project="test-project", ee_module=FakeEE({"999": 2}))
    assert result["status"] == "failed"
    assert "unsupported class code" in result["error"]


def test_authentication_or_configuration_failure(monkeypatch):
    # This test intentionally exercises missing configuration, independent of
    # any developer-local GEE_PROJECT environment variable.
    monkeypatch.delenv("GEE_PROJECT", raising=False)
    no_project = retrieve_landcover_evidence(10, 76, ee_module=FakeEE({"50": 1}))
    init_error = retrieve_landcover_evidence(
        10,
        76,
        gee_project="test-project",
        ee_module=FakeEE({"50": 1}, init_error=RuntimeError("not authenticated")),
    )
    assert no_project["status"] == "failed"
    assert "GEE_PROJECT" in no_project["error"]
    assert init_error["status"] == "failed"
    assert "not authenticated" in init_error["error"]


def test_empty_pixel_counts_return_failure():
    result = retrieve_landcover_evidence(10, 76, gee_project="test-project", ee_module=FakeEE({"50": 0}))
    assert result["status"] == "failed"
    assert "no valid pixels" in result["error"]
