from src.data.constants import Currency

import click


class CurrencyType(click.ParamType):
    def convert(self, value, param, ctx):
        if isinstance(value, str):
            try:
                return Currency(value.upper())
            except ValueError:
                self.fail(f"{value!r} is not a valid string", param, ctx)
