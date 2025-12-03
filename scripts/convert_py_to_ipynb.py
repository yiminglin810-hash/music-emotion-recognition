"""
Convert .py file (with # %% markers) to .ipynb notebook
"""

import json
from pathlib import Path
import re

def parse_py_to_cells(py_content):
    """Parse Python file with # %% markers into notebook cells"""
    cells = []
    lines = py_content.split('\n')
    
    current_cell = []
    current_type = None
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for cell marker
        if line.strip().startswith('# %% [markdown]'):
            # Save previous cell if exists
            if current_cell and current_type:
                cells.append(create_cell(current_type, current_cell))
            
            # Start markdown cell
            current_type = 'markdown'
            current_cell = []
            i += 1
            continue
        
        elif line.strip().startswith('# %%'):
            # Save previous cell if exists
            if current_cell and current_type:
                cells.append(create_cell(current_type, current_cell))
            
            # Start code cell
            current_type = 'code'
            current_cell = []
            i += 1
            continue
        
        # Process cell content
        if current_type == 'markdown':
            # Remove leading '# ' from markdown lines
            if line.startswith('# '):
                current_cell.append(line[2:] + '\n')
            elif line.startswith('#'):
                current_cell.append(line[1:] + '\n')
            else:
                current_cell.append(line + '\n')
        
        elif current_type == 'code':
            current_cell.append(line + '\n')
        
        i += 1
    
    # Save last cell
    if current_cell and current_type:
        cells.append(create_cell(current_type, current_cell))
    
    return cells

def create_cell(cell_type, lines):
    """Create a notebook cell"""
    # Join lines and clean up
    if cell_type == 'markdown':
        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
    
    elif cell_type == 'code':
        # Remove trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()
        # Ensure last line has newline
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
    
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    
    return cell

def convert_py_to_ipynb(py_file, output_file):
    """Convert Python file to Jupyter notebook"""
    print(f"Reading: {py_file}")
    
    # Read Python file
    with open(py_file, 'r', encoding='utf-8') as f:
        py_content = f.read()
    
    # Parse cells
    print("Parsing cells...")
    cells = parse_py_to_cells(py_content)
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save notebook
    print(f"Saving: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 Output: {output_file}")
    print(f"📊 Total cells: {len(cells)}")
    print(f"   - Markdown: {len([c for c in cells if c['cell_type'] == 'markdown'])}")
    print(f"   - Code: {len([c for c in cells if c['cell_type'] == 'code'])}")

def main():
    """Main function"""
    # Set paths
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    
    # Check command line argument for file name
    import sys
    if len(sys.argv) > 1:
        source_name = sys.argv[1]
    else:
        source_name = "02_feature_extraction"
    
    py_file = notebooks_dir / f"{source_name}.py"
    output_file = notebooks_dir / f"{source_name}_EN.ipynb"
    
    # Check if input exists
    if not py_file.exists():
        print(f"❌ Error: {py_file} not found")
        return
    
    # Convert
    convert_py_to_ipynb(py_file, output_file)
    
    print(f"\n💡 100% English notebook ready for GitHub!")

if __name__ == "__main__":
    main()

