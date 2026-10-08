from pathlib import Path

import pytest

from backend.services import gee_retriever as retriever


class Value:
    def __init__(self, value):
        self.value = value

    def getInfo(self):
        return self.value


class FakeGeometry:
    def coordinates(self):
        return Value([[0, 0], [1, 0], [1, 1], [0, 1]])


class FakeImage:
    def __init__(self, acquired, cloud):
        self.acquired = acquired
        self.cloud = cloud

    def date(self):
        return self

    def format(self, _pattern):
        return Value(self.acquired)

    def get(self, _name):
        return Value(self.cloud)

    def select(self, _bands):
        return self

    def reproject(self, **_kwargs):
        return self

    def projection(self):
        return self

    def crs(self):
        return Value("EPSG:32643")


class FakeCollection:
    def __init__(self, image):
        self.image = image

    def filterBounds(self, _aoi):
        return self

    def filterDate(self, _start, _end):
        return self

    def filter(self, _filter):
        return self

    def sort(self, _field):
        return self

    def size(self):
        return Value(0 if self.image is None else 1)

    def first(self):
        return self.image


class FakeEE:
    class Filter:
        @staticmethod
        def lt(_field, _value):
            return object()

    class Geometry:
        @staticmethod
        def Rectangle(_coordinates):
            return FakeGeometry()

    _unset = object()

    def __init__(self, before_image, after_image=_unset, init_error=None):
        self.before_image = before_image
        self.after_image = before_image if after_image is self._unset else after_image
        self.init_error = init_error
        self.collection_calls = 0

    def Initialize(self, project):
        assert project == "test-project"
        if self.init_error:
            raise self.init_error

    def ImageCollection(self, _collection_id):
        self.collection_calls += 1
        return FakeCollection(self.before_image if self.collection_calls == 1 else self.after_image)

    @staticmethod
    def Image(image):
        return image


def write_download(image, _aoi, destination: Path, _name, _crs):
    destination.write_bytes(b"mock-geotiff")


def test_successful_retrieval_writes_two_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(retriever, "ee", FakeEE(FakeImage("2025-01-15", 3), FakeImage("2025-02-10", 4)))
    result = retriever.retrieve_sentinel2_images(
        "GS-TEST1234", 10.1234, 76.1234, "2025-01-01", "2025-02-01",
        gee_project="test-project", artifact_root=tmp_path, downloader=write_download,
    )
    assert result["success"] is True
    assert result["metadata"]["bands"] == ["B2", "B3", "B4", "B8", "B11", "B12"]
    assert result["metadata"]["grid"] == {"crs": "EPSG:32643", "scale_meters": 10}
    assert Path(result["before_image_path"]).exists()
    assert Path(result["after_image_path"]).exists()
    assert (tmp_path / "GS-TEST1234" / "metadata.json").exists()


def test_empty_before_collection(monkeypatch, tmp_path):
    monkeypatch.setattr(retriever, "ee", FakeEE(None, FakeImage("2025-02-10", 4)))
    result = retriever.retrieve_sentinel2_images(
        "GS-EMPTYBEFORE", 10, 76, "2025-01-01", "2025-02-01",
        gee_project="test-project", artifact_root=tmp_path, downloader=write_download,
    )
    assert result == {
        "success": False,
        "error": "No suitable Sentinel-2 image found for the requested period",
        "stage": "before_query",
    }


def test_empty_after_collection(monkeypatch, tmp_path):
    monkeypatch.setattr(retriever, "ee", FakeEE(FakeImage("2025-01-15", 3), None))
    result = retriever.retrieve_sentinel2_images(
        "GS-EMPTYAFTER", 10, 76, "2025-01-01", "2025-02-01",
        gee_project="test-project", artifact_root=tmp_path, downloader=write_download,
    )
    assert result["success"] is False
    assert result["stage"] == "after_query"


def test_gee_initialization_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(retriever, "ee", FakeEE(None, init_error=RuntimeError("not authenticated")))
    result = retriever.retrieve_sentinel2_images(
        "GS-INITFAIL", 10, 76, "2025-01-01", "2025-02-01",
        gee_project="test-project", artifact_root=tmp_path,
    )
    assert result["success"] is False
    assert result["stage"] == "gee_initialization"
    assert "not authenticated" in result["error"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"latitude": 91}, "latitude must be between -90 and 90"),
        ({"before_date": "not-a-date"}, "before_date must be a valid ISO date"),
        ({"after_date": "2025-01-01"}, "before_date must be earlier than after_date"),
        ({"gee_project": None}, "GEE_PROJECT is required"),
    ],
)
def test_invalid_configuration_or_input(monkeypatch, tmp_path, kwargs, message):
    # The missing-project case must be deterministic even when the developer's
    # shell has a real GEE_PROJECT configured.
    monkeypatch.delenv("GEE_PROJECT", raising=False)
    monkeypatch.setattr(retriever, "ee", FakeEE(FakeImage("2025-01-15", 3)))
    args = dict(
        investigation_id="GS-INVALID", latitude=10, longitude=76,
        before_date="2025-01-01", after_date="2025-02-01",
        gee_project="test-project", artifact_root=tmp_path, downloader=write_download,
    )
    args.update(kwargs)
    result = retriever.retrieve_sentinel2_images(**args)
    assert result["success"] is False
    assert message in result["error"]
