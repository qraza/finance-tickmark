"""Load .env from the repo root so ANTHROPIC_API_KEY is set for every entry point."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
