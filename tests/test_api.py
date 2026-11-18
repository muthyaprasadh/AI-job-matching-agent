from fastapi.testclient import TestClient

from job_agent import api


def test_api_health_and_authenticated_search(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA", tmp_path)
    monkeypatch.setenv("JOB_AGENT_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/v1/jobs/search", json={"query": "Java"}).status_code == 401
        response = client.post(
            "/v1/jobs/search",
            headers={"X-API-Key": "test-api-key"},
            json={"query": "Java", "limit": 5},
        )
        chat_response = client.post(
            "/v1/chat",
            headers={"X-API-Key": "test-api-key"},
            json={"message": "Find suitable jobs"},
        )

    assert response.status_code == 200
    assert response.json() == {"jobs": []}
    assert chat_response.status_code == 503
