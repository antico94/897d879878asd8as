import MetaTrader5 as mt5
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import time


class MT5Connector:
    """Connector for trading operations with MetaTrader 5."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.mt5_config = config.get('MetaTrader5', {})
        self.connected = False

    def connect(self) -> bool:
        """Connect to MetaTrader 5 terminal.

        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            if self.connected:
                self.logger.info("Already connected to MT5")
                return True

            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False

            # Connect with the provided credentials
            login = self.mt5_config.get('Login')
            password = self.mt5_config.get('Password')
            server = self.mt5_config.get('Server')
            timeout = self.mt5_config.get('Timeout', 60000)

            # Check API documentation requirements
            try:
                if not mt5.login(login, password=password, server=server):
                    error = mt5.last_error()
                    self.logger.error(f"MT5 login failed: {error}")
                    mt5.shutdown()
                    return False
            except Exception as e:
                self.logger.error(f"MT5 login exception: {e}")
                mt5.shutdown()
                return False

            self.connected = True
            self.logger.info(f"Connected to MT5 server: {server}")
            return True

        except Exception as e:
            self.logger.error(f"Error connecting to MT5: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from MetaTrader 5 terminal.

        Returns:
            bool: True if disconnected successfully, False otherwise
        """
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")
            return True
        return False

    def is_connected(self) -> bool:
        """Check if connected to MT5.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information.

        Returns:
            Dict[str, Any]: Account information
        """
        try:
            if not self.connected and not self.connect():
                return {}

            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return {}

            # Convert MT5 account info tuple to dictionary
            result = {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'profit': account_info.profit
            }

            return result
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}

    def get_market_data(self, symbol: str, timeframe: str, bars: int = 100) -> Dict[str, Any]:
        """Get market data for a symbol.

        Args:
            symbol: Symbol name (e.g., "XAUUSD")
            timeframe: Timeframe (e.g., "M15", "H1")
            bars: Number of bars to fetch

        Returns:
            Dict[str, Any]: Market data
        """
        try:
            if not self.connected and not self.connect():
                return {}

            # Convert timeframe string to MT5 timeframe
            timeframe_map = {
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }

            mt5_timeframe = timeframe_map.get(timeframe.upper())
            if not mt5_timeframe:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return {}

            # Fetch rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                self.logger.error(f"No data received for {symbol} with timeframe {timeframe}")
                return {}

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Get the latest price from ticker info
            ticker_info = mt5.symbol_info_tick(symbol)
            if ticker_info is None:
                self.logger.error(f"Failed to get ticker info for {symbol}")
                return {}

            current_price = (ticker_info.bid + ticker_info.ask) / 2

            # Create ATR (Average True Range) calculation for volatility measure
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # Calculate True Range
            tr1 = high[1:] - low[1:]
            tr2 = abs(high[1:] - close[:-1])
            tr3 = abs(low[1:] - close[:-1])

            # True Range is the max of the three
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1).values

            # Calculate ATR (14-period simple moving average of TR)
            atr = sum(tr[-14:]) / 14 if len(tr) >= 14 else sum(tr) / len(tr) if len(tr) > 0 else 0

            # Create market data dictionary
            market_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'time': df['time'].iloc[-1],
                'open': df['open'].iloc[-1],
                'high': df['high'].iloc[-1],
                'low': df['low'].iloc[-1],
                'close': df['close'].iloc[-1],
                'volume': df['tick_volume'].iloc[-1],
                'price': current_price,
                'spread': ticker_info.ask - ticker_info.bid,
                'atr': atr
            }

            return market_data

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}

    def send_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a trading order to MT5.

        Args:
            order_params: Order parameters

        Returns:
            Dict[str, Any]: Order result
        """
        try:
            if not self.connected and not self.connect():
                return {}

            # Create MT5 request structure
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order_params.get("symbol", ""),
                "volume": float(order_params.get("volume", 0.01)),
                "type": mt5.ORDER_TYPE_BUY if order_params.get("action") == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": float(order_params.get("price", 0)),
                "sl": float(order_params.get("sl", 0)),
                "tp": float(order_params.get("tp", 0)),
                "deviation": int(order_params.get("deviation", 10)),
                "magic": int(order_params.get("magic", 123456)),
                "comment": order_params.get("comment", "MT5 Connector Order"),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK
            }

            # Send order
            result = mt5.order_send(request)
            if result is None:
                error = mt5.last_error()
                self.logger.error(f"Order send failed: {error}")
                return {"retcode": -1, "message": str(error)}

            # Format response
            response = {
                "retcode": result.retcode,
                "message": self._get_retcode_message(result.retcode),
                "deal": result.deal,
                "order": result.order,
                "volume": result.volume,
                "price": result.price,
                "bid": result.bid,
                "ask": result.ask,
                "request": request
            }

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {response['message']}")
            else:
                self.logger.info(
                    f"Order executed successfully: {order_params.get('action')} {order_params.get('symbol')} {order_params.get('volume')} lots")

            return response

        except Exception as e:
            self.logger.error(f"Error sending order: {e}")
            return {"retcode": -1, "message": str(e)}

    def get_open_positions(self, magic: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all open positions.

        Args:
            magic: Optional magic number to filter positions

        Returns:
            List[Dict[str, Any]]: List of open positions
        """
        try:
            if not self.connected and not self.connect():
                return []

            # Get positions
            if magic is not None:
                positions = mt5.positions_get(magic=magic)
            else:
                positions = mt5.positions_get()

            if positions is None or len(positions) == 0:
                return []

            # Convert to list of dictionaries
            result = []
            for position in positions:
                position_dict = {
                    'ticket': position.ticket,
                    'symbol': position.symbol,
                    'type': 'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': position.volume,
                    'open_price': position.price_open,
                    'current_price': position.price_current,
                    'sl': position.sl,
                    'tp': position.tp,
                    'profit': position.profit,
                    'magic': position.magic,
                    'comment': position.comment,
                    'open_time': pd.to_datetime(position.time, unit='s')
                }
                result.append(position_dict)

            return result

        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []

    def close_position(self, close_params: Dict[str, Any]) -> Dict[str, Any]:
        """Close an open position.

        Args:
            close_params: Close parameters (must include ticket)

        Returns:
            Dict[str, Any]: Close result
        """
        try:
            if not self.connected and not self.connect():
                return {}

            # Get position details
            position = mt5.positions_get(ticket=close_params.get('ticket'))
            if position is None or len(position) == 0:
                self.logger.error(f"Position not found: {close_params.get('ticket')}")
                return {"retcode": -1, "message": "Position not found"}

            position = position[0]

            # Create close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": float(close_params.get("volume", position.volume)),
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": mt5.symbol_info_tick(
                    position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(
                    position.symbol).ask,
                "magic": position.magic,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK
            }

            # Send close order
            result = mt5.order_send(request)
            if result is None:
                error = mt5.last_error()
                self.logger.error(f"Close position failed: {error}")
                return {"retcode": -1, "message": str(error)}

            # Format response
            response = {
                "retcode": result.retcode,
                "message": self._get_retcode_message(result.retcode),
                "deal": result.deal,
                "order": result.order,
                "volume": result.volume,
                "price": result.price,
                "request": request
            }

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Close position failed: {response['message']}")
            else:
                self.logger.info(f"Position {position.ticket} closed successfully")

            return response

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {"retcode": -1, "message": str(e)}

    def modify_position(self, modify_params: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an open position (change SL/TP).

        Args:
            modify_params: Modification parameters (must include ticket)

        Returns:
            Dict[str, Any]: Modification result
        """
        try:
            if not self.connected and not self.connect():
                return {}

            # Get position details
            position = mt5.positions_get(ticket=modify_params.get('ticket'))
            if position is None or len(position) == 0:
                self.logger.error(f"Position not found: {modify_params.get('ticket')}")
                return {"retcode": -1, "message": "Position not found"}

            position = position[0]

            # Create modify request
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": float(modify_params.get("sl", position.sl)),
                "tp": float(modify_params.get("tp", position.tp))
            }

            # Send modify order
            result = mt5.order_send(request)
            if result is None:
                error = mt5.last_error()
                self.logger.error(f"Modify position failed: {error}")
                return {"retcode": -1, "message": str(error)}

            # Format response
            response = {
                "retcode": result.retcode,
                "message": self._get_retcode_message(result.retcode)
            }

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Modify position failed: {response['message']}")
            else:
                self.logger.info(
                    f"Position {position.ticket} modified successfully: SL={request['sl']}, TP={request['tp']}")

            return response

        except Exception as e:
            self.logger.error(f"Error modifying position: {e}")
            return {"retcode": -1, "message": str(e)}

    def _get_retcode_message(self, retcode: int) -> str:
        """Get human-readable message for MT5 return code.

        Args:
            retcode: MT5 return code

        Returns:
            str: Human-readable message
        """
        return {
            mt5.TRADE_RETCODE_DONE: "Done",
            mt5.TRADE_RETCODE_REJECT: "Request rejected",
            mt5.TRADE_RETCODE_CANCEL: "Request canceled by trader",
            mt5.TRADE_RETCODE_TIMEOUT: "Request canceled by timeout",
            mt5.TRADE_RETCODE_ERROR: "Request processing error",
            mt5.TRADE_RETCODE_INVALID: "Invalid request",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume in request",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price in request",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops in request",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trade is disabled",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market is closed",
            mt5.TRADE_RETCODE_NO_MONEY: "Not enough money",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
            mt5.TRADE_RETCODE_PRICE_OFF: "No quotes to process request",
            mt5.TRADE_RETCODE_REQUOTE: "Requote",
            mt5.TRADE_RETCODE_ORDER_LOCKED: "Order is locked",
            mt5.TRADE_RETCODE_LIMIT_ORDERS: "Too many orders",
            mt5.TRADE_RETCODE_LIMIT_VOLUME: "Volume limit exceeded"
        }.get(retcode, f"Unknown error code: {retcode}")