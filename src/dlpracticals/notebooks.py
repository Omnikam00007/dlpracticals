"""Simple, comment-free practical code: show it, put it into a notebook, or fill the next cell."""
import json
import os

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')


def setup_code(data_dir='Dataset'):
    return f"import dlpracticals as dp\nDATA = dp.prepare_data({data_dir!r})"


def cells(number, data_dir='Dataset'):
    """List of (cell_type, source) for practical `number`, with the data-setup cell added when needed."""
    if not 1 <= number <= 8:
        raise ValueError(f'Practical must be 1-8, got {number}')
    with open(os.path.join(TEMPLATE_DIR, f'p{number}.txt'), encoding='utf-8') as f:
        lines = f.read().splitlines()

    result, kind, body = [], None, []
    for line in lines + ['# %%']:
        if line.startswith('# %%'):
            if kind and '\n'.join(body).strip():
                result.append((kind, '\n'.join(body).strip('\n')))
            kind = 'markdown' if '[markdown]' in line else 'code'
            body = []
        elif kind == 'markdown':
            body.append(line[2:] if line.startswith('# ') else line.lstrip('#'))
        else:
            body.append(line)

    if number != 1:
        result.insert(1, ('code', setup_code(data_dir)))
    return result


def code(number, data_dir='Dataset'):
    """All code cells of practical `number` joined into one script."""
    return '\n\n'.join(src for kind, src in cells(number, data_dir) if kind == 'code')


def notebook(number, path=None, data_dir='Dataset'):
    """Write Practical_<number>.ipynb with one cell per step and return its path."""
    path = path or f'Practical_{number}.ipynb'
    nb_cells = []
    for kind, src in cells(number, data_dir):
        lines = src.split('\n')
        source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
        cell = {'cell_type': kind, 'metadata': {}, 'source': source}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    nb = {'cells': nb_cells,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f'Created {path} - open it in Jupyter and run all cells.')
    return path


def all_notebooks(folder='.', data_dir='Dataset'):
    os.makedirs(folder, exist_ok=True)
    return [notebook(n, os.path.join(folder, f'Practical_{n}.ipynb'), data_dir) for n in range(1, 9)]


def fill(number, data_dir='Dataset'):
    """In Jupyter, put the whole practical into the next cell. Elsewhere, print it."""
    script = code(number, data_dir)
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        ip = None
    if ip is None:
        print(script)
        return
    ip.set_next_input(script, replace=False)


def show_code(number, data_dir='Dataset', save_to=None):
    """Display practical `number`'s code (highlighted in Jupyter); optionally save it to a .py file."""
    script = code(number, data_dir)
    if save_to:
        with open(save_to, 'w', encoding='utf-8') as f:
            f.write(script + '\n')
        print(f'Saved to {save_to}')
    try:
        from IPython import get_ipython
        from IPython.display import Code, display
        if get_ipython() is not None:
            display(Code(script, language='python'))
            return
    except ImportError:
        pass
    print(script)
