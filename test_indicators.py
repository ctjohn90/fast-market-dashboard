"""Quick test of indicator calculations."""

from fast_market_dashboard.indicators import IndicatorCalculator


def main() -> None:
    calc = IndicatorCalculator()
    result = calc.calculate()

    if result is None:
        print("No data available. Run the FRED fetcher first.")
        return

    print(f"\nMarket Stress Dashboard - As of {result.as_of_date}")
    print("=" * 60)
    print(f"\nCOMPOSITE SCORE: {result.composite_score:.1f} / 100")
    print(f"STATUS: {result.stress_level}")
    print("\n" + "-" * 60)
    print("Component Breakdown:\n")

    for key, indicator in result.indicators.items():
        print(f"  {indicator.name:20} | {indicator.percentile:5.1f} | {indicator.current_value:>10.2f}")

    print("\n" + "-" * 60)

    # Get some history
    history = calc.get_history(days=30)
    if not history.empty:
        print(f"\n30-Day Composite Range: {history['composite'].min():.1f} - {history['composite'].max():.1f}")
        print(f"Current vs 30-day avg: {result.composite_score:.1f} vs {history['composite'].mean():.1f}")


if __name__ == "__main__":
    main()
