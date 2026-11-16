"""API tests with a mock model (no weights needed)."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.serve.app import create_app, load_serve_config
from src.serve.inference import MockEngine
from tests.conftest import make_document_image


@pytest.fixture
def cfg():
    cfg = load_serve_config()
    cfg["mode"] = "mock"
    cfg["limits"] = {**cfg["limits"], "max_upload_mb": 1, "max_question_chars": 50, "max_batch_size": 3}
    return cfg


@pytest.fixture
def engine(cfg):
    return MockEngine(cfg, answer="$1,250.00")


@pytest.fixture
def client(cfg, engine):
    with TestClient(create_app(engine=engine, cfg=cfg)) as c:  # `with` runs the lifespan (engine.load)
        yield c


def _bytes(fmt="PNG", size=(400, 300)):
    buf = io.BytesIO()
    make_document_image(size=size).save(buf, format=fmt)
    return buf.getvalue()


def test_health(client, cfg):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"] == cfg["model_version"]


def test_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "<form" in r.text


@pytest.mark.parametrize("fmt,name", [("PNG", "doc.png"), ("JPEG", "doc.jpg"), ("PDF", "doc.pdf")])
def test_predict_accepts_png_jpg_pdf(client, fmt, name, engine):
    r = client.post("/predict", files={"file": (name, _bytes(fmt), "application/octet-stream")},
                    data={"question": "What is the total?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "$1,250.00"
    assert body["latency_ms"] >= 0 and body["model_version"]
    assert engine.calls[-1][1] == "What is the total?"


def test_large_image_is_resized_to_max_pixels(client, engine, cfg):
    r = client.post("/predict", files={"file": ("big.png", _bytes(size=(1200, 1100)), "image/png")},
                    data={"question": "Total?"})
    assert r.status_code == 200, r.text
    w, h = engine.calls[-1][0]
    assert w * h <= cfg["max_pixels"]


def test_unsupported_type_is_415(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello world", "text/plain")}, data={"question": "q?"})
    assert r.status_code == 415
    assert "Allowed" in r.json()["detail"]


def test_corrupt_png_is_400(client):
    data = _bytes()[:60]
    r = client.post("/predict", files={"file": ("bad.png", data, "image/png")}, data={"question": "q?"})
    assert r.status_code == 400


def test_too_large_upload_is_413(client):
    data = b"\x89PNG\r\n\x1a\n" + b"0" * (1024 * 1024 + 10)
    r = client.post("/predict", files={"file": ("huge.png", data, "image/png")}, data={"question": "q?"})
    assert r.status_code == 413


@pytest.mark.parametrize("question,fragment", [("   ", "empty"), ("x" * 51, "too long")])
def test_bad_question_is_400(client, question, fragment):
    r = client.post("/predict", files={"file": ("d.png", _bytes(), "image/png")}, data={"question": question})
    assert r.status_code == 400 and fragment in r.json()["detail"]


def test_missing_fields_are_422(client):
    assert client.post("/predict", data={"question": "q?"}).status_code == 422
    assert client.post("/predict", files={"file": ("d.png", _bytes(), "image/png")}).status_code == 422


def test_batch_predict(client):
    files = [("files", ("a.png", _bytes(), "image/png")), ("files", ("b.jpg", _bytes("JPEG"), "image/jpeg"))]
    r = client.post("/predict/batch", files=files, data={"questions": ["Total?", "Date?"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [x["question"] for x in body["results"]] == ["Total?", "Date?"]
    assert all(x["answer"] == "$1,250.00" for x in body["results"])


def test_batch_length_mismatch_is_400(client):
    files = [("files", ("a.png", _bytes(), "image/png"))]
    r = client.post("/predict/batch", files=files, data={"questions": ["Total?", "Date?"]})
    assert r.status_code == 400


def test_batch_too_large_is_400(client):
    files = [("files", (f"{i}.png", _bytes(), "image/png")) for i in range(4)]
    r = client.post("/predict/batch", files=files, data={"questions": ["q"] * 4})
    assert r.status_code == 400


def test_model_error_returns_500(cfg):
    class Broken(MockEngine):
        def predict(self, images, questions):
            raise RuntimeError("boom")

    with TestClient(create_app(engine=Broken(cfg), cfg=cfg)) as c:
        r = c.post("/predict", files={"file": ("d.png", _bytes(), "image/png")}, data={"question": "q?"})
    assert r.status_code == 500 and "boom" in r.json()["detail"]


def test_not_loaded_returns_503(cfg):
    class NeverLoads(MockEngine):
        def load(self):
            pass

    with TestClient(create_app(engine=NeverLoads(cfg), cfg=cfg)) as c:
        r = c.post("/predict", files={"file": ("d.png", _bytes(), "image/png")}, data={"question": "q?"})
        assert r.status_code == 503
        assert c.get("/health").json()["status"] == "loading"


def test_mock_mode_via_config(cfg):
    with TestClient(create_app(cfg=cfg)) as c:  # no engine passed -> built from cfg["mode"] == "mock"
        assert c.get("/health").json()["model_loaded"] is True
    assert isinstance(Image.new("RGB", (1, 1)), Image.Image)
