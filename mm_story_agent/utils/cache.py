import json
import hashlib
from pathlib import Path
from typing import Any, Optional


class DiskCache:
    """
    极简磁盘缓存：以(命名空间 + key)计算哈希作为文件名，保存/读取JSON。
    - 无第三方依赖
    - 适用于LLM调用等可缓存结果
    """

    def __init__(self, cache_dir: str = ".cache") -> None:
        self.cache_root = Path(cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        ns_dir = self.cache_root / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return ns_dir / f"{h}.json"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        p = self._path(namespace, key)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
        except Exception:
            pass
