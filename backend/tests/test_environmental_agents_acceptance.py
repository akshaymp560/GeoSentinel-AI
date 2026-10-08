import httpx

from backend.agents.fire_api import retrieve_fire_evidence
from backend.agents.landcover_api import retrieve_landcover_evidence
from backend.agents.news_api import retrieve_news_evidence
from backend.agents.terrain_api import retrieve_terrain_evidence
from backend.agents.weather_api import retrieve_weather_evidence


class FakeResponse:
    def __init__(self, payload=None, text="", error=None):
        self.payload = payload
        self.text = text
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeValue:
    def __init__(self, value):
        self.value = value

    def getInfo(self):
        return self.value


class FakeWorldCoverImage:
    def select(self, _band):
        return self

    def reduceRegion(self, **_kwargs):
        return FakeValue({"Map": {"50": 3, "40": 1}})


class FakeEE:
    class Geometry:
        @staticmethod
        def Rectangle(coordinates):
            return coordinates

    class Reducer:
        @staticmethod
        def frequencyHistogram():
            return "frequencyHistogram"

    def Initialize(self, project):
        assert project == "acceptance-project"

    def ImageCollection(self, _collection):
        image = FakeWorldCoverImage()
        return type("Collection", (), {"first": lambda _self: image})()


def test_all_environmental_agents_return_fusion_ready_evidence(capsys):
    latitude = 10.1234
    longitude = 76.1234
    before_date = "2025-01-01"
    after_date = "2025-01-03"
    acceptance_key = "acceptance-placeholder"

    weather_payload = {
        "daily": {
            "time": [before_date, "2025-01-02", after_date],
            "precipitation_sum": [0.0, 2.0, 1.0],
            "temperature_2m_mean": [20.0, 21.0, 22.0],
            "temperature_2m_max": [24.0, 25.0, 26.0],
            "wind_speed_10m_max": [8.0, 9.0, 10.0],
        }
    }
    news_payload = {
        "articles": [
            {
                "title": "Local environmental report",
                "domain": "example.org",
                "seendate": "20250102120000",
                "url": "https://example.org/report",
                "snippet": "Contextual public reporting.",
            }
        ]
    }
    fire_csv = (
        "latitude,longitude,acq_date,acq_time,confidence,frp,satellite,instrument,daynight\n"
        "10.1234,76.1234,2025-01-02,1200,nominal,10.0,Suomi-NPP,VIIRS,D\n"
    )

    def weather_get(*args, **kwargs):
        return FakeResponse(payload=weather_payload)

    def terrain_get(*args, **kwargs):
        return FakeResponse(payload={"elevation": [123.4]})

    def fire_get(*args, **kwargs):
        return FakeResponse(text=fire_csv)

    def news_get(*args, **kwargs):
        return FakeResponse(payload=news_payload)

    calls = {
        "weather": lambda: retrieve_weather_evidence(latitude, longitude, before_date, after_date, http_get=weather_get),
        "fire": lambda: retrieve_fire_evidence(latitude, longitude, before_date, after_date, firms_map_key=acceptance_key, http_get=fire_get),
        "terrain": lambda: retrieve_terrain_evidence(latitude, longitude, http_get=terrain_get),
        "landcover": lambda: retrieve_landcover_evidence(latitude, longitude, gee_project="acceptance-project", ee_module=FakeEE()),
        "news": lambda: retrieve_news_evidence(latitude, longitude, before_date, after_date, http_get=news_get),
    }

    results = {agent: call() for agent, call in calls.items()}
    combined = {
        "location": {"latitude": latitude, "longitude": longitude},
        "period": {"before_date": before_date, "after_date": after_date},
        "agents": results,
    }

    assert set(results) == {"weather", "fire", "terrain", "landcover", "news"}
    assert all(result["agent"] == agent for agent, result in results.items())
    assert all(result["status"] in {"success", "failed"} for result in results.values())
    assert all("evidence" in result for result in results.values())
    assert all(result["evidence"] is not None for result in results.values())
    assert results["weather"]["metrics"]["total_precipitation_mm"] == 3.0
    assert results["fire"]["metrics"]["detection_count"] == 1
    assert results["terrain"]["metrics"]["elevation_m"] == 123.4
    assert results["landcover"]["land_cover"]["class_code"] == 50
    assert results["news"]["metrics"]["article_count"] == 1
    assert combined["location"] == {"latitude": latitude, "longitude": longitude}
    assert combined["period"] == {"before_date": before_date, "after_date": after_date}
    assert set(combined["agents"]) == set(calls)
    assert acceptance_key not in capsys.readouterr().out
