"""Command line:
    python -m dlpracticals prepare [--source Dataset] [--out data]
    python -m dlpracticals notebook <number|all> [--data-dir Dataset] [--out FOLDER]
    python -m dlpracticals show <number> [--save FILE]
    python -m dlpracticals run <number> [--data-dir Dataset]
"""
import argparse

from . import all_notebooks, notebook, prepare_data, run_practical, set_data_dir, show_code


def main():
    parser = argparse.ArgumentParser(prog='python -m dlpracticals')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('prepare', help='extract all dataset zips into one folder')
    p.add_argument('--source', default='Dataset', help='Dataset folder or the downloaded Dataset zip')
    p.add_argument('--out', default='data')

    p = sub.add_parser('notebook', help='create Practical_<n>.ipynb (or all 8)')
    p.add_argument('number', help='1-8 or "all"')
    p.add_argument('--data-dir', default='Dataset')
    p.add_argument('--out', default='.', help='folder for the notebooks')

    p = sub.add_parser('show', help='display the code of a practical')
    p.add_argument('number', type=int, choices=range(1, 9))
    p.add_argument('--save', metavar='FILE')

    p = sub.add_parser('run', help='run the full version of a practical')
    p.add_argument('number', type=int, choices=range(1, 9))
    p.add_argument('--data-dir', default='Dataset')

    args = parser.parse_args()
    if args.command == 'prepare':
        print('Data ready in', prepare_data(args.source, args.out))
    elif args.command == 'notebook':
        if args.number == 'all':
            all_notebooks(args.out, args.data_dir)
        else:
            import os
            os.makedirs(args.out, exist_ok=True)
            notebook(int(args.number), os.path.join(args.out, f'Practical_{args.number}.ipynb'), args.data_dir)
    elif args.command == 'show':
        show_code(args.number, save_to=args.save)
    else:
        set_data_dir(args.data_dir)
        run_practical(args.number)


if __name__ == '__main__':
    main()
