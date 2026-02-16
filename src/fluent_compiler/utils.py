import builtins
import keyword
import re


# From spec:
#    NamedArgument ::= Identifier blank? ":" blank? (StringLiteral | NumberLiteral)
#    Identifier ::= [a-zA-Z] [a-zA-Z0-9_-]*

NAMED_ARG_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def allowable_keyword_arg_name(name: str) -> re.Match | None:
    return NAMED_ARG_RE.match(name)


def allowable_name(ident: str, for_method: bool = False, allow_builtin: bool = False) -> bool:
    if keyword.iskeyword(ident):
        return False

    if not (for_method or allow_builtin):
        if ident in dir(builtins):
            return False

    if not ident.isidentifier():
        return False

    return True
