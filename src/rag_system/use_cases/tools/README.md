The format of all the tools should be like 

```python
from langchain_core.tools import tool
from typing import Annotated


@tool
def multiply_by_max(
    a: Annotated[int, "scale factor"],
    b: Annotated[List[int], "list of ints over which to take maximum"],
) -> int:
    """Multiply a by the maximum of b."""
    return a * max(b)
```

This is not just because I like to, the offitial docummentation says that 
https://python.langchain.com/docs/how_to/custom_tools/