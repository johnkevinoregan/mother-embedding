#!/usr/bin/env python3
"""pluto2md.py -- convert a Pluto .jl notebook to plain Markdown, without Pluto.

Markdown cells (md\"\"\"...\"\"\" or md"...") become raw markdown;
every other cell becomes a fenced ```julia block. Cell order from the
"# ╔═╡ Cell order:" footer is respected when present.

usage: python3 pluto2md.py notebook.jl [> out.md]
       python3 pluto2md.py notebook.jl --code=skip     # markdown only
"""
import re, sys

CELL = re.compile(r'^# ╔═╡ ([0-9a-f\-]{8,})\s*$')

def parse(path):
    cells, order, cur, uuid, in_order = {}, [], [], None, False
    for line in open(path, encoding='utf-8').read().splitlines():
        if line.startswith('# ╔═╡ Cell order:'):
            if uuid: cells[uuid] = '\n'.join(cur).strip('\n')
            uuid, in_order = None, True
            continue
        if in_order:
            # Pluto marks code cells '# ╠═<uuid>' and folded/markdown cells '# ╟─<uuid>'
            m = re.match(r'^# (?:╠═|╟─)(.+?)\s*$', line)
            if m: order.append(m.group(1).strip())
            continue
        m = CELL.match(line)
        if m:
            if uuid: cells[uuid] = '\n'.join(cur).strip('\n')
            uuid, cur = m.group(1), []
        elif uuid is not None:
            cur.append(line)
    if uuid: cells[uuid] = '\n'.join(cur).strip('\n')
    if not order: order = list(cells.keys())
    return [cells[u] for u in order if u in cells]

def as_markdown(src):
    """Return markdown text if the cell is a md-literal, else None."""
    s = src.strip()
    for open_, close_ in (('md"""', '"""'), ("md'''", "'''")):
        if s.startswith(open_) and s.endswith(close_) and len(s) > len(open_)+len(close_)-1:
            return s[len(open_):-len(close_)].strip('\n')
    if s.startswith('md"') and s.endswith('"') and '\n' not in s:
        return s[3:-1]
    return None

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    code_mode = 'fence'
    for a in sys.argv[1:]:
        if a.startswith('--code='): code_mode = a.split('=',1)[1]
    if not args:
        print(__doc__); sys.exit(1)
    out = []
    for src in parse(args[0]):
        if not src.strip(): continue
        md = as_markdown(src)
        if md is not None:
            out.append(md)
        elif code_mode != 'skip':
            out.append('```julia\n' + src + '\n```')
    print('\n\n'.join(out))

if __name__ == '__main__':
    main()
