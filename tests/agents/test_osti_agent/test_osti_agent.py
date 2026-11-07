import json

from bs4 import BeautifulSoup
from requests import Response

from ursa.agents.acquisition_agents import OSTIAgent


async def test_osti_agent_ainvoke_returns_summary(
    chat_model, tmpdir, monkeypatch
):
    api_base = "https://osti.test/api/v1/records"
    query = "high-temperature superconductors"
    context = "Provide a quick overview of relevant OSTI research."

    landing_url = "https://osti.test/landing/osti-001"
    hit = {
        "osti_id": "osti-001",
        "title": "Advanced HTS Applications",
        "links": [{"rel": "citation", "href": landing_url}],
    }

    def fake_requests_get(url, *args, **kwargs):
        assert url == api_base
        params = kwargs.get("params") or {}
        assert params.get("q") == query
        response = Response()
        response.status_code = 200
        response.url = url
        response._content = json.dumps({"records": [hit]}).encode()
        return response

    def fake_resolve_pdf(record, *args, **kwargs):
        assert record["osti_id"] == hit["osti_id"]
        return None, landing_url, None

    def fake_get_soup(url, *args, **kwargs):
        assert url == landing_url
        return BeautifulSoup(
            "<html><body>OSTI document discussing superconductors and energy "
            "grids.</body></html>",
            "html.parser",
        )

    monkeypatch.setattr(
        "ursa.agents.acquisition_agents.requests.get", fake_requests_get
    )
    monkeypatch.setattr(
        "ursa.agents.acquisition_agents.resolve_pdf_from_osti_record",
        fake_resolve_pdf,
    )
    monkeypatch.setattr(
        "ursa.agents.acquisition_agents._get_soup",
        fake_get_soup,
    )

    agent = OSTIAgent(
        llm=chat_model,
        api_base=api_base,
        workspace=tmpdir,
        database_path="osti_db",
        summaries_path="osti_summaries",
        vectorstore_path="osti_vectors",
    )

    result = await agent.ainvoke({"query": query, "context": context})

    assert result["items"], "Expected at least one retrieved item."
    item = result["items"][0]
    assert item["id"] == hit["osti_id"]
    assert "superconductors" in item["full_text"]

    summaries = result.get("summaries")
    assert summaries and isinstance(summaries[0], str)

    final_summary = result.get("final_summary")
    assert isinstance(final_summary, str)
