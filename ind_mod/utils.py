"""
01/24, R. S. James
"""

def exporter():
    """Export utility modified from https://stackoverflow.com/a/41895194.
    """
    all_ = []

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter()
__all__.extend(['exporter'])
