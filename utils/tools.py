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


# https://community.openai.com/t/function-return-values-definition-in-function-calling-feature/317405
def format_returns(function_name, result):
    if function_name == 'get_delivery_date':
        return {"delivery_date": result.strftime('%Y-%m-%d %H:%M:%S') if result is not None else ""}
    raise ValueError(f"Invalid function name: {function_name}")

def format_direct_response(function_name, result, arguments):
    if function_name == 'get_delivery_date':
        if result is None:
            return f'Your order {arguments["order_id"]} is not in our records, please recheck the order ID.'
        else:
            return f'The delivery date of your order {arguments["order_id"]} is {result}'
    raise ValueError(f"Invalid function name: {function_name}")