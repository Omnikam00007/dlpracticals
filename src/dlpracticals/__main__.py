"""Command line:
    python -m dlpracticals <number> [--data-dir PATH]          run a practical
    python -m dlpracticals show <number> [--helpers] [--save FILE]   display its code
"""
import argparse
import sys

from . import run_practical, set_data_dir, show_code


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'show':
        parser = argparse.ArgumentParser(prog='python -m dlpracticals show',
                                         description='Display the source code of a practical.')
        parser.add_argument('number', type=int, choices=range(1, 9), help='practical number (1-8)')
        parser.add_argument('--helpers', action='store_true', help='also show data.py and plots.py')
        parser.add_argument('--save', metavar='FILE', help='also write the code to FILE')
        args = parser.parse_args(sys.argv[2:])
        show_code(args.number, helpers=args.helpers, save_to=args.save)
        return

    parser = argparse.ArgumentParser(prog='python -m dlpracticals',
                                     description='Run a practical (or use "show <number>" to display its code).')
    parser.add_argument('number', type=int, choices=range(1, 9), help='practical number (1-8)')
    parser.add_argument('--data-dir', help='folder holding Handwritten.zip, Fashion.zip, ... (default: ./Dataset)')
    args = parser.parse_args()
    if args.data_dir:
        set_data_dir(args.data_dir)
    run_practical(args.number)


if __name__ == '__main__':
    main()
