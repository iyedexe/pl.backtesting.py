from .base import Classifier
from .rules import RuleClassifier

__all__ = ['Classifier', 'RuleClassifier', 'ClaudeClassifier']


def __getattr__(name):  # lazy: the Claude classifier needs the optional `anthropic` package
    if name == 'ClaudeClassifier':
        from .claude import ClaudeClassifier
        return ClaudeClassifier
    raise AttributeError(name)
