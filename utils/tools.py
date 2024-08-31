# https://github.com/cadifyai/tool2schema

import pandas as pd
from datetime import datetime, timedelta
from tool2schema import EnableTool

@EnableTool
def get_delivery_date(order_id: str) -> datetime:
    """
    Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'

    :param order_id: The customer's order ID.
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