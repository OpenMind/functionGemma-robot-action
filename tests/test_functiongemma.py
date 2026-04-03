"""
Unit tests for FunctionGemma robot-action service.

Run: poetry run pytest tests/test_functiongemma.py -v
"""

import json

from fastapi.testclient import TestClient

from src.functiongemma.server import (
    ACTIONS,
    EMOTIONS,
    FUNCTIONS,
    PredictRequest,
    app,
    build_prompt,
)

client = TestClient(app)


class TestBuildPrompt:
    """Tests for prompt formatting — run without GPU."""

    def test_contains_user_input(self):
        assert "shake my hand" in build_prompt("shake my hand")

    def test_has_correct_structure(self):
        prompt = build_prompt("hello")
        assert "<bos><start_of_turn>developer" in prompt
        assert "<start_of_turn>user" in prompt
        assert "<start_of_turn>model" in prompt

    def test_contains_function_declarations(self):
        prompt = build_prompt("test")
        assert "robot_action" in prompt
        assert "show_emotion" in prompt

    def test_contains_all_actions(self):
        prompt = build_prompt("test")
        for action in ACTIONS:
            assert action in prompt

    def test_contains_all_emotions(self):
        prompt = build_prompt("test")
        for emotion in EMOTIONS:
            assert emotion in prompt


class TestConfig:
    """Tests for action/emotion definitions."""

    def test_expected_actions(self):
        assert set(ACTIONS) == {"shake_hand", "face_wave", "hands_up", "stand_still", "show_hand"}

    def test_expected_emotions(self):
        assert set(EMOTIONS) == {"happy", "sad", "excited", "confused", "curious", "think"}

    def test_function_definitions(self):
        assert len(FUNCTIONS) == 2
        assert {f["name"] for f in FUNCTIONS} == {"robot_action", "show_emotion"}

    def test_serializable(self):
        for f in FUNCTIONS:
            json.dumps(f)


class TestEndpoints:
    """Tests for API endpoints."""

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert "status" in r.json()

    def test_actions(self):
        r = client.get("/actions")
        assert r.status_code == 200
        assert set(r.json()["actions"]) == set(ACTIONS)
        assert set(r.json()["emotions"]) == set(EMOTIONS)

    def test_predict_empty_text(self):
        r = client.post("/predict", json={"text": ""})
        assert r.status_code in (400, 503)

    def test_predict_missing_text(self):
        r = client.post("/predict", json={})
        assert r.status_code == 422

    def test_predict_batch_missing_texts(self):
        r = client.post("/predict_batch", json={})
        assert r.status_code == 422
