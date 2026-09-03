"""pytest 共享配置：把项目根目录加入 sys.path，保证 `import src` 可用。

在项目根目录执行 `python -m pytest tests -q` 即可。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
