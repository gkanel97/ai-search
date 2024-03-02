import sys

def get_deep_size(obj):
    """
    Recursively finds the size of objects.

    Parameters:
    obj (object): The object for which the size needs to be calculated.

    Returns:
    int: The size of the object in bytes.
    """
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum([get_deep_size(v) for v in obj.values()])
        size += sum([get_deep_size(k) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_deep_size(obj.__dict__)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_deep_size(i) for i in obj])
    return size