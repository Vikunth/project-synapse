from __future__ import annotations

import uvicorn

from synapse.config import SynapseConfig


def main() -> None:
    """Main entrypoint for the application."""
    config = SynapseConfig()
    uvicorn.run(
        "synapse.app:create_app", factory=True, host=config.host, port=config.port, log_level="info"
    )


if __name__ == "__main__":
    main()
