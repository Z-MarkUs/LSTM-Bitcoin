"""Allow ``python -m lstm_bitcoin`` to invoke the CLI."""

from .cli import main

raise SystemExit(main())
