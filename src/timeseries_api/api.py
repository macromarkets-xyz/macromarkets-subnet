import argparse
import asyncio
import sys
from random import random

from fastapi import FastAPI, Depends
from supabase import create_client, Client
import uvicorn
from dynaconf import Dynaconf
import os
from typing import Annotated


def from_compressed_str(cls, cs: str) -> Type["ModelId"]:
    """Returns an instance of this class from a compressed string representation"""
    tokens = cs.split(":")
    return cls(
        namespace=tokens[0],
        name=tokens[1],
        chat_template=tokens[2] if tokens[2] != "None" else None,
        commit=tokens[3] if tokens[3] != "None" else None,
        hash=tokens[4] if tokens[4] != "None" else None,
        competition_id=(
            tokens[5] if len(tokens) >= 6 and tokens[5] != "None" else None
        ),
    )


class TimeseriesEvaluationService:
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()
        self.settings = Dynaconf(settings_files=["settings.toml"])
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
        )
        self.aggressive_score = True

    def setup_routes(self):
        @self.app.get("/score")
        async def get_score():
            # Implement your score logic here
            return {"score": 100}

        @self.app.get("/a")
        async def a():
            # Implement your score logic here
            return {"score": random()}

        @self.app.get("/b")
        async def b():
            # Implement your score logic here
            return self.check_existing_score()

    def check_existing_score(self):
        if self.aggressive_score:
            return {"ayy": "lmao"}

    def get_supabase(self) -> Client:
        return self.supabase

    def run(self):
        parser = argparse.ArgumentParser(description="Run the Score Service")
        parser.add_argument(
            "--port", type=int, default=9141, help="Port to run the service on"
        )
        args = parser.parse_args()

        uvicorn.run(self.app, host="0.0.0.0", port=args.port)

    async def shutdown(self):
        # Perform any cleanup tasks here
        print("Cleaning up resources...")


# Usage
if __name__ == "__main__":
    try:
        service = TimeseriesEvaluationService()
        service.run()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Initiating graceful shutdown...")
        # If we're using asyncio, we need to run the shutdown coroutine
        asyncio.run(service.shutdown())
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        print("Service has stopped.")


# Additional endpoint example
@service.app.get("/additional_endpoint")
async def additional_endpoint(
    supabase: Annotated[Client, Depends(service.get_supabase)]
):
    # Use supabase client here
    result = supabase.table("some_table").select("*").execute()
    return {"result": result}
