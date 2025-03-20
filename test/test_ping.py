import pytest
from httpx import ASGITransport, AsyncClient

from app.api.http import app


@pytest.mark.asyncio
async def test_ping():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/ping")
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "pong"
