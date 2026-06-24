# Configuration file for the Sphinx documentation builder.
# Location: docs/source/conf.py

import os
import sys
import shutil
import re

# -- Path Setup --------------------------------------------------------------
# Point to the source code directory for autodoc (API reference)
sys.path.insert(0, os.path.abspath('../../src/tam'))

# -- Project Information -----------------------------------------------------
project = 'TAM'
copyright = '2026, Yann Allioux, Nathan Doumèche, Éloi Bedek'
author = 'Yann Allioux'
release = '1.2.5'

# -- General Configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Generate API docs from docstrings
    'sphinx.ext.viewcode',     # Link to source code
    'sphinx.ext.napoleon',     # Parse Google/NumPy-style docstrings
    'sphinx.ext.mathjax',      # Render math via LaTeX
    'sphinx.ext.todo',         # Support TODOs
    'myst_parser',             # Markdown support
    'sphinxcontrib.bibtex',    # Bibliography management
]

# Tell Sphinx to ignore missing cross-reference warnings for GitHub-only files
suppress_warnings = ["myst.xref_missing"]

# -- Extension Settings ------------------------------------------------------
# Docstrings (Napoleon)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Markdown (MyST) configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]
# Allow spaces inside inline math delimiters (fixes unrendered equations)
myst_dmath_allow_space = True 
# Auto-generate anchors for headings so cross-references work properly in PDF
myst_heading_anchors = 3

# Bibliography (BibTeX)
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'plain'
bibtex_reference_style = 'author_year'

templates_path = ['_templates']
exclude_patterns = []

# -- HTML Output -------------------------------------------------------------
html_theme = 'furo'
html_logo = '_static/logo.png'
html_static_path = ['_static']

html_theme_options = {
    "sidebar_hide_name": False,  # Set to True if logo includes the project name
}

# -- LaTeX (PDF) Output ------------------------------------------------------
latex_engine = 'pdflatex'

latex_elements = {
    'pointsize': '11pt',
    'sphinxsetup': ', '.join([
        'hmargin={2.5cm, 2.5cm}',
        'vmargin={2.5cm, 2.5cm}',
        'marginpar=1cm',
        'TitleColor={rgb}{0.1, 0.2, 0.5}',
        'HeaderFamily=\\rmfamily\\bfseries',
        'InnerLinkColor={rgb}{0.1, 0.2, 0.5}',
        'OuterLinkColor={rgb}{0.1, 0.2, 0.5}',
        'VerbatimColor={rgb}{0.96, 0.96, 0.96}',       # Light grey background
        'VerbatimBorderColor={rgb}{0.85, 0.85, 0.85}', # Subtle border
        'verbatimwithframe=true',                      # Draw the border
        'verbatimwrapslines=true',                     # Auto-wrap long lines (Fixes the issue)
    ]),
    'fontpkg': '\\usepackage{mathpazo}',  # Palatino font
    'preamble': r'''
        \setcounter{tocdepth}{0}
        
        \usepackage[utf8]{inputenc}
        \usepackage{charter}
        \usepackage{fancyhdr}
        \usepackage{hyperref}

        \AtBeginDocument{\hypertarget{toc_link}{}}

        \pagestyle{fancy}
        \fancyhf{} 
        
        % Use a box to protect the link from being overwritten by long titles
        \fancyfoot[L]{\parbox[b]{0.6\textwidth}{\hyperlink{toc_link}{\small \textit{Back to Summary}}}}
        \fancyfoot[R]{\thepage}
        \renewcommand{\footrulewidth}{0.4pt}

        % Shorten the header to prevent vertical bleed
        \fancyhead[L]{\small\nouppercase{\leftmark}}

        % Re-apply to plain pages (Chapter starts)
        \fancypagestyle{plain}{
            \fancyhf{}
            \fancyfoot[L]{\parbox[b]{0.6\textwidth}{\hyperlink{toc_link}{\small \textit{Back to Summary}}}}
            \fancyfoot[R]{\thepage}
            \renewcommand{\footrulewidth}{0.4pt}
        }

        % IMPORTANT: Increase footer separation to prevent title overlap
        \setlength{\footskip}{30pt} 
        \setlength{\headheight}{13.6pt}
        \addtolength{\topmargin}{-1.6pt}
    ''',
    'extraclassoptions': 'openany,oneside',
}

latex_documents = [
    ('index', 'TAM_documentation.tex', 'TAM: Complete Reference Manual', author, 'manual'),
]

# -- Build Hooks -------------------------------------------------------------
def clean_markdown_for_latex(app, docname, source):
    """
    Intercepts the Markdown parsing ONLY during PDF (latex) builds.
    - Deletes SVG badges (shields.io)
    - Deletes Emojis (which cause fatal errors in pdflatex)
    - Deletes Markdown navigation links
    - Converts HTML center blocks to MyST image directives
    """
    if app.builder.name == 'latex':
        text = source[0]
        
        # 1. Remove shields.io and zenodo.org badges
        text = re.sub(r'(?m)^.*(?:shields\.io|zenodo\.org).*$', '', text)
                
        # 2. Remove Emojis based on standard Unicode blocks
        emoji_pattern = re.compile(
            r'['
            r'\U0001F600-\U0001F64F'  # Emoticons
            r'\U0001F300-\U0001F5FF'  # Misc Symbols and Pictographs
            r'\U0001F680-\U0001F6FF'  # Transport and Map (🚀)
            r'\U0001F700-\U0001F77F'  # Alchemical Symbols
            r'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
            r'\U0001F800-\U0001F8FF'  # Supplemental Arrows
            r'\U0001F900-\U0001F9FF'  # Supplemental Symbols (🧪, 🧩)
            r'\U0001FA00-\U0001FA6F'  # Chess Symbols
            r'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended
            r'\u2600-\u26FF'          # Misc symbols (⚙, ⚡)
            r'\u2700-\u27BF'          # Dingbats (✨)
            r']+',
            re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # 3. Remove the invisible Variation Selector-16 (often attached to emojis like ⚙️)
        text = text.replace('\ufe0f', '')

        # 4. Remove internal Markdown navigation links (e.g., [`⬅️ README`](README.md))
        nav_pattern = re.compile(r'^(?:\[`[^`]+`\]\([^)]+\)(?:\s*\|\s*)?)+$', re.MULTILINE)
        text = nav_pattern.sub('', text)      

        # 5. Convert HTML image tags to MyST directives for LaTeX rendering
        html_img_pattern = re.compile(
            r'(?:<div[^>]*>\s*|<br>\s*)?<img\s+src="([^"]+)"[^>]*alt="([^"]*)"[^>]*>(?:\s*</div>)?',
            re.IGNORECASE
        )
        
        def img_replacer(match):
            # Strip 'docs/source/' so Sphinx can find the file relative to its root
            img_path = match.group(1).replace("docs/source/", "")
            alt_text = match.group(2)
            ticks = "```"
            return f"\n{ticks}{{image}} {img_path}\n:alt: {alt_text}\n:align: center\n:width: 100%\n{ticks}\n"
            
        text = html_img_pattern.sub(img_replacer, text)

        source[0] = text

def setup(app):
    """
    Auto-syncs Markdown files from the repository root to the Sphinx source directory
    before compilation. Enforces the 'Docs as Code' Mirror Architecture.
    """
    # Attach the new master cleaner for LaTeX
    app.connect('source-read', clean_markdown_for_latex)
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    print(f"\n[Auto-Sync] Building documentation silently...")

    # Sync root-level files
    files_to_copy = [
        'README.md', 'THEORY.md', 'AUTHORS.md', 'ACKNOWLEDGEMENTS.md', 
        'CHANGELOG.md', 'EXAMPLES.md', 
        'bibliography.md', 'references.bib', 'CONTRIBUTING.md',
        'README_DOC.md'
    ]
    for filename in files_to_copy:
        src, dst = os.path.join(root_dir, filename), os.path.join(app.srcdir, filename)
        if os.path.exists(src):
            if filename == 'README.md' or filename == 'THEORY.md':
                with open(src, 'r', encoding='utf-8') as f:
                    content = f.read()
                content = content.replace('docs/source/_static/', '_static/')
                with open(dst, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                shutil.copyfile(src, dst)

    # Sync mirrored directories
    for folder in ['math', 'architecture']:
        src_folder = os.path.join(root_dir, folder)
        dst_folder = os.path.join(app.srcdir, folder)
        if os.path.exists(src_folder):
            if os.path.exists(dst_folder):
                shutil.rmtree(dst_folder)
            shutil.copytree(src_folder, dst_folder)