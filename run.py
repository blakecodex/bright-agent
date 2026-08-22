"""run.py — do not edit. Runs your agent against the mocks."""
import sys

from mock_client import MockClient, MalformedMockClient
from agent import run_agent


def main():
    broken = "--broken" in sys.argv
    client = MalformedMockClient() if broken else MockClient()
    label = "MALFORMED" if broken else "HAPPY PATH"
    print(f"=== {label} ===")
    answer = run_agent(client, "Is 123 Oak St listed at a fair price?")
    print("\nFINAL ANSWER:\n" + answer)


if __name__ == "__main__":
    main()
