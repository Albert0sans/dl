from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *

import threading
import time

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.accepted_orders = set()
        self.nextorderId = None

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextorderId = orderId
        print('The next valid order id is:', self.nextorderId)

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                    permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        print(f'orderStatus - orderId: {orderId}, status: {status}, filled: {filled}, remaining: {remaining}, lastFillPrice: {lastFillPrice}')
        if status in ['Submitted', 'PreSubmitted'] and orderId not in self.accepted_orders:
            print(f"Limit order {orderId} accepted!")
            self.accepted_orders.add(orderId)

    def openOrder(self, orderId, contract, order, orderState):
        print(f'openOrder - id: {orderId}, {contract.symbol}, {contract.secType}@{contract.exchange}, action: {order.action}, type: {order.orderType}, qty: {order.totalQuantity}, status: {orderState.status}')
        if orderState.status in ['Submitted', 'PreSubmitted'] and orderId not in self.accepted_orders:
            print(f"Limit order {orderId} accepted (via openOrder callback)!")
            self.accepted_orders.add(orderId)

    def execDetails(self, reqId, contract, execution):
        print(f'Order Executed - reqId: {reqId}, symbol: {contract.symbol}, execId: {execution.execId}, orderId: {execution.orderId}, shares: {execution.shares}, lastLiquidity: {execution.lastLiquidity}')
    def error(self, reqId, errorCode, errorString):
	    if errorCode == 202:
		    print('order canceled') 

def run_loop():
    app.run()

def NewContract(symbol: str) -> Contract:
    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'STK'  # Update as needed for crypto/FX
    contract.exchange = 'SMART'
    contract.currency = 'USD'
    return contract

def createOrders(price: float, addStopLoss: bool, addTakeProfit: bool):
    orders = []

    # Parent (main) order
    parent_order = Order()
    parent_order.eTradeOnly = False
    parent_order.firmQuoteOnly = False
    parent_order.action = 'BUY'
    parent_order.totalQuantity = 10
    parent_order.orderType = 'LMT'
    parent_order.lmtPrice = price
    parent_order.orderId = app.nextorderId
    parent_order.transmit = not (addStopLoss or addTakeProfit)
    orders.append(parent_order)
    app.nextorderId += 1

    if addStopLoss:
        stop_order = Order()
        stop_order.eTradeOnly = False
        stop_order.firmQuoteOnly = False
        stop_order.action = 'SELL'
        stop_order.totalQuantity = 10
        stop_order.orderType = 'STP'
        stop_order.auxPrice = price * 0.97
        stop_order.orderId = app.nextorderId
        stop_order.parentId = parent_order.orderId
        stop_order.transmit = not addTakeProfit
        orders.append(stop_order)
        app.nextorderId += 1

    if addTakeProfit:
        tp_order = Order()
        tp_order.eTradeOnly = False
        tp_order.firmQuoteOnly = False
        tp_order.action = 'SELL'
        tp_order.totalQuantity = 10
        tp_order.orderType = 'LMT'
        tp_order.lmtPrice = price * 1.04
        tp_order.orderId = app.nextorderId
        tp_order.parentId = parent_order.orderId
        tp_order.transmit = True  # Last in bracket transmits the group
        orders.append(tp_order)
        app.nextorderId += 1

    return orders


# Initialize app
app = IBapi()
app.connect('127.0.0.1', 4002, 123)

# Start the socket in a thread
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

# Wait for connection
while isinstance(app.nextorderId, type(None)):
    print('waiting for connection...')
    time.sleep(1)

print('Connected\n')

# Create and place bracket orders
symbol = 'AAPL'  # Change to valid symbol (e.g., 'EUR' or 'BTC' if supported)
orders = createOrders(price=200, addStopLoss=True, addTakeProfit=True)
contract = NewContract(symbol)

for order in orders:
    app.placeOrder(order.orderId, contract, order)
