"""Verify that your LangSmith API key is valid and has write access."""

import os
import sys

import requests
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

API_KEY = os.getenv("LANGSMITH_API_KEY")
ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
PROJECT = os.getenv("LANGSMITH_PROJECT", "default")


def check_env():
    print("=== Environment ===")
    print(f"LANGSMITH_ENDPOINT: {ENDPOINT}")
    print(f"LANGSMITH_PROJECT:  {PROJECT}")
    print(f"LANGSMITH_API_KEY:  {'set' if API_KEY else 'MISSING'}")
    if API_KEY:
        print(f"  prefix: {API_KEY[:20]}...")
        print(f"  length: {len(API_KEY)}")
    print()


def check_raw_api():
    print("=== Direct API check ===")
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{ENDPOINT}/sessions?limit=1"
    try:
        resp = requests.get(url, headers=headers)
        print(f"GET {url}")
        print(f"Status: {resp.status_code}")
        print(f"Body:   {resp.text[:200]}")
        if resp.status_code == 200:
            print("✅ API key is valid for reading.")
        elif resp.status_code in (401, 403):
            print("❌ API key is invalid, revoked, expired, or lacks permission.")
        else:
            print("⚠️ Unexpected status.")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    print()


def check_client():
    print("=== LangSmith SDK Client check ===")
    try:
        client = Client(api_url=ENDPOINT, api_key=API_KEY)
        projects = list(client.list_projects())
        print(f"✅ SDK authenticated. Found {len(projects)} projects.")
        print(f"   Projects: {[p.name for p in projects][:5]}")
    except Exception as e:
        print(f"❌ SDK authentication failed: {type(e).__name__}: {str(e)[:300]}")
    print()


def check_write():
    print("=== Write access check ===")
    try:
        client = Client(api_url=ENDPOINT, api_key=API_KEY)
        # Create a minimal run to verify write access
        run_id = client.create_run(
            name="verify_write",
            run_type="chain",
            inputs={"test": "hello"},
            project_name=PROJECT,
        )
        print(f"✅ Write access confirmed. Created run: {run_id}")
    except Exception as e:
        print(f"❌ Write access failed: {type(e).__name__}: {str(e)[:300]}")
    print()


if __name__ == "__main__":
    check_env()
    check_raw_api()
    check_client()
    check_write()
