import pandas as pd
from datetime import datetime, timedelta
from langchain_core.tools import tool
from typing import Annotated

@tool
def get_delivery_date(
    order_id: Annotated[str, "The customer's order ID."],
    ) -> datetime:
    """
    Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'
    """
    data = {
        'order_id': ['12345', '67890', '54321', '98765'],
        'delivery_date': [
            datetime.now() + timedelta(days=3),
            datetime.now() + timedelta(days=5),
            datetime.now() + timedelta(days=2),
            datetime.now() + timedelta(days=7),
        ]
    }
    df = pd.DataFrame(data)
    result = df[df['order_id'] == order_id]['delivery_date']
    return result.iloc[0] if not result.empty else None

def get_tools():
    return [get_delivery_date]