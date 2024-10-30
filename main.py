import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Order:
    order_id: str
    timestamp: pd.Timestamp
    side: str  # 'bid' or 'ask'
    price: float
    quantity: float
    status: str = 'active'


class OrderBook:
    def __init__(self):
        self.bids = defaultdict(float)  # price -> quantity
        self.asks = defaultdict(float)
        self.orders = {}  # order_id -> Order

    def add_order(self, order: Order):
        self.orders[order.order_id] = order
        if order.side == 'bid':
            self.bids[order.price] += order.quantity
        else:
            self.asks[order.price] += order.quantity

    def cancel_order(self, order_id: str):
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == 'active':
                if order.side == 'bid':
                    self.bids[order.price] -= order.quantity
                    if self.bids[order.price] == 0:
                        del self.bids[order.price]
                else:
                    self.asks[order.price] -= order.quantity
                    if self.asks[order.price] == 0:
                        del self.asks[order.price]
                order.status = 'cancelled'

    def get_spread(self) -> float:
        if not self.bids or not self.asks:
            return float('inf')
        return min(self.asks.keys()) - max(self.bids.keys())


class MarketAnalyzer:
    def __init__(self, trades_df: pd.DataFrame, orders_df: pd.DataFrame):
        self.trades_df = trades_df
        self.orders_df = orders_df
        self.order_book = OrderBook()

    def reconstruct_order_book(self, timestamp: pd.Timestamp) -> OrderBook:
        """Reconstruct order book state at given timestamp"""
        relevant_orders = self.orders_df[self.orders_df['timestamp'] <= timestamp]
        order_book = OrderBook()

        for _, row in relevant_orders.iterrows():
            order = Order(
                order_id=row['order_id'],
                timestamp=row['timestamp'],
                side=row['side'],
                price=row['price'],
                quantity=row['quantity']
            )
            order_book.add_order(order)

        return order_book

    def calculate_vwap(self, window: str = '1h') -> pd.Series:  # Use lowercase 'h'
        """Calculate Volume Weighted Average Price"""
        self.trades_df['volume'] = self.trades_df['price'] * self.trades_df['quantity']
        vwap = (
                self.trades_df.groupby(pd.Grouper(key='timestamp', freq=window))['volume'].sum() /
                self.trades_df.groupby(pd.Grouper(key='timestamp', freq=window))['quantity'].sum()
        )
        return vwap

    def analyze_market_impact(self, trade_size: float) -> str:
        """Analyze market impact for given trade size and return a plain language summary."""
        spreads = []
        depths = []

        for timestamp in self.trades_df['timestamp'].unique():
            book = self.reconstruct_order_book(timestamp)
            spread = book.get_spread()
            if spread != float('inf'):
                spreads.append(spread)

            # Calculate market depth at best bid/ask
            best_bid = max(book.bids.keys()) if book.bids else 0
            best_ask = min(book.asks.keys()) if book.asks else float('inf')
            bid_depth = book.bids[best_bid]
            ask_depth = book.asks[best_ask]
            depths.append((bid_depth + ask_depth) / 2)

        avg_spread = np.mean(spreads) if spreads else float('inf')
        avg_depth = np.mean(depths) if depths else 0
        estimated_impact = (avg_spread * trade_size) / avg_depth if avg_depth > 0 else float('inf')

        # Format the result in plain English
        analysis_result = (
            f"Market Impact Analysis:\n"
            f"1. **Average Price Difference (Spread)**: On average, the difference between "
            f"what buyers are willing to pay and what sellers are willing to accept is about "
            f"${avg_spread:.2f}.\n"
            f"2. **Market Depth**: There are typically around {avg_depth:.2f} units available "
            f"for trade at the best prices, which gives a sense of how many orders are available "
            f"at the most favorable prices.\n"
            f"3. **Estimated Price Impact**: A trade of size {trade_size} is expected to change the "
            f"price by approximately ${estimated_impact:.2f} due to the impact on available orders."
        )

        return analysis_result

    def visualize_order_flow(self, start_time: pd.Timestamp, end_time: pd.Timestamp):
        """Create order flow visualization"""
        mask = (self.trades_df['timestamp'] >= start_time) & (self.trades_df['timestamp'] <= end_time)
        trades_subset = self.trades_df[mask]

        fig = go.Figure()

        # Plot trades
        fig.add_trace(go.Scatter(
            x=trades_subset['timestamp'],
            y=trades_subset['price'],
            mode='markers',
            name='Trades',
            marker=dict(
                size=trades_subset['quantity'] / trades_subset['quantity'].mean() * 5,
                color=trades_subset['side'].map({'buy': 'green', 'sell': 'red'})
            )
        ))

        # Plot VWAP
        vwap = self.calculate_vwap('5min')
        fig.add_trace(go.Scatter(
            x=vwap.index,
            y=vwap.values,
            mode='lines',
            name='VWAP',
            line=dict(color='blue', dash='dash')
        ))

        fig.update_layout(
            title='Order Flow Analysis',
            xaxis_title='Time',
            yaxis_title='Price',
            showlegend=True
        )

        return fig


def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample data for testing"""
    # Create sample trades data
    trades = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
        'price': np.random.normal(100, 1, 100),
        'quantity': np.random.randint(1, 100, 100),
        'side': np.random.choice(['buy', 'sell'], 100)
    })

    # Create sample orders data
    orders = pd.DataFrame({
        'order_id': [f'ord_{i}' for i in range(1000)],
        'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='6s'),
        'side': np.random.choice(['bid', 'ask'], 1000),
        'price': np.random.normal(100, 1, 1000),
        'quantity': np.random.randint(1, 50, 1000)
    })

    return trades, orders


# Example usage
if __name__ == "__main__":
    # Load sample data
    trades_df, orders_df = load_sample_data()

    # Initialize analyzer
    analyzer = MarketAnalyzer(trades_df, orders_df)

    # Calculate VWAP
    vwap = analyzer.calculate_vwap()
    print("VWAP Analysis Complete")

    # Analyze market impact
    impact = analyzer.analyze_market_impact(100.0)
    print(f"Market Impact Analysis: {impact}")

    # Create visualization
    start_time = trades_df['timestamp'].min()
    end_time = trades_df['timestamp'].max()
    fig = analyzer.visualize_order_flow(start_time, end_time)
    fig.show()
    print("Visualization Created")
