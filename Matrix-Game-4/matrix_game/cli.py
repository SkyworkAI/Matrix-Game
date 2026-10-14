"""CLI imports only the dependencies required by the selected workflow."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Matrix-Game 4.0")
    parser.add_argument("command", choices=("serve", "render", "download", "verify"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    rest = args.arguments
    if args.command == "serve":
        from .server.app import main as run
    elif args.command == "render":
        from .inference.offline import main as run
    else:
        from .resources import main as run
        rest = [args.command] + rest
    run(rest)


if __name__ == "__main__":
    main()
