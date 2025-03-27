from typing import Union, Optional, Dict, List, Any, Callable

class AsyncBaseProvider:


    async def execute(self):
        pass

    def usage(self):
        pass

    async def get_models(self):
        pass