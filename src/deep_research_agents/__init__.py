"""Deep research agents package.

Several submodules under this package import sibling packages
(``agents``, ``agent_tools``, ``prompts``) as *top-level* modules rather than
via the fully-qualified ``deep_research_agents.*`` path.  For those bare
imports to resolve, ``src/`` and ``src/deep_research_agents/`` must be on
``sys.path`` before any submodule is imported.  Registering them here — in the
package ``__init__`` — guarantees that ordering regardless of which submodule is
imported first.
"""

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent          # src/deep_research_agents
_SRC_DIR = _PKG_DIR.parent                           # src

for _p in (str(_SRC_DIR), str(_PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
