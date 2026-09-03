"""
document_parser 层的可插拔注册表实例。

各解析器实现类通过 ``@document_parser_registry.register("name")`` 注册到此表，
上层（API / 编排）只按配置里的名字调用 ``create()``，做到新增解析器不改上层代码。
"""
from src.common.registry import Registry
from src.document_parser.base import BaseDocumentParser

document_parser_registry = Registry(
    name="document_parser",
    base_type=BaseDocumentParser,
)
