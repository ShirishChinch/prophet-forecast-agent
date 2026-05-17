from __future__ import annotations

from fastapi.testclient import TestClient

from app import app


def test_predict_handles_bad_json_without_500() -> None:
    client = TestClient(app)

    response = client.post(
        "/predict",
        content="not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "probabilities": [
            {"market": "Yes", "probability": 0.5},
            {"market": "No", "probability": 0.5},
        ]
    }


def test_exclusive_multi_outcome_fallback_normalizes() -> None:
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "title": "Who will win the NBA championship?",
            "category": "Sports",
            "outcomes": ["A", "B", "C", "D"],
        },
    )

    probabilities = response.json()["probabilities"]
    assert response.status_code == 200
    assert [row["market"] for row in probabilities] == ["A", "B", "C", "D"]
    assert sum(row["probability"] for row in probabilities) == 1.0


def test_nonexclusive_multi_outcome_does_not_force_sum_to_one() -> None:
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "title": "Which teams will make the playoffs?",
            "category": "Sports",
            "outcomes": ["A", "B", "C"],
            "probabilities": [0.7, 0.6, 0.4],
        },
    )

    assert response.status_code == 200
    assert response.json()["probabilities"] == [
        {"market": "A", "probability": 0.7},
        {"market": "B", "probability": 0.6},
        {"market": "C", "probability": 0.4},
    ]
