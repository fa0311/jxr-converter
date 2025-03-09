import argparse
from typing import Callable, ClassVar

from pydantic import BaseModel


class ArgsParserConfig:
    parser: argparse.ArgumentParser

    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser

    def add_argument(self, fn: Callable[[argparse.ArgumentParser], argparse.Action]):
        args = fn(self.parser)
        return {"default": args.default, "description": args.help}


class ArgsParserBase(BaseModel):
    parser: ClassVar[ArgsParserConfig]

    @classmethod
    def parse_args(cls):
        args = cls.parser.parser.parse_args()
        return cls(**args.__dict__)
