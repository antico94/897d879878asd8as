import questionary
from questionary import Choice
from UI.Constants import AppMode
from Utilities.ConfigurationUtils import Config
from Configuration.Constants import TimeFrames, CurrencyPairs


class TradingBotCLI:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.main_choices = [
            Choice("Fetch Data", AppMode.FETCH_DATA.value),
            Choice("Exit", "exit")
        ]

    def main_menu(self) -> str:
        return questionary.select(
            'Select an action:',
            choices=self.main_choices
        ).ask() or 'exit'

    def display_fetch_config(self):
        fetch_config = self.config.get('FetchingSettings')
        pair_code = fetch_config.get('DefaultPair', 'XAUUSD')

        # Create a formatted display of current configuration
        config_display = [
            "Current Fetching Configuration:",
            f"• Currency Pair: {CurrencyPairs.display_name(pair_code)}",
            f"• Time Period: {fetch_config.get('DefaultTimeperiod', 2001)} days",
            f"• Timeframe: {fetch_config.get('DefaultTimeframe', 'H1')}",
            f"• Splitting Ratio: {fetch_config.get('SplittingRatio', {}).get('Training', 70)}% training, "
            f"{fetch_config.get('SplittingRatio', {}).get('Validation', 15)}% validation, "
            f"{fetch_config.get('SplittingRatio', {}).get('Testing', 15)}% testing"
        ]

        for line in config_display:
            print(line)

    def fetch_data_menu(self) -> str:
        self.display_fetch_config()
        print()  # Add an empty line for better readability

        choices = [
            Choice("Fetch data with current configuration", "fetch_current"),
            Choice("Change configuration", "change_config"),
            Choice("Go back", "back")
        ]

        return questionary.select(
            'Select an option:',
            choices=choices
        ).ask() or 'back'

    def change_config_menu(self):
        fetch_config = self.config.get('FetchingSettings', {})

        # 1. Select currency pair
        pairs = [CurrencyPairs.XAUUSD, CurrencyPairs.USDJPY, CurrencyPairs.EURUSD, CurrencyPairs.GBPUSD]
        pair_choices = [Choice(CurrencyPairs.display_name(p), p) for p in pairs]

        selected_pair = questionary.select(
            'Select currency pair:',
            choices=pair_choices
        ).ask()

        if not selected_pair:
            return None

        # 2. Input days for time period
        default_days = fetch_config.get('DefaultTimeperiod', 2001)
        days = questionary.text(
            f'Enter number of days (default: {default_days}):',
            default=str(default_days),
            validate=lambda text: text.isdigit() and int(text) > 0
        ).ask()

        if not days:
            return None

        # 3. Select timeframe
        timeframes = [tf.value for tf in TimeFrames]
        timeframe = questionary.select(
            'Select timeframe:',
            choices=timeframes
        ).ask()

        if not timeframe:
            return None

        # Return the selected configuration
        return {
            'pair': selected_pair,
            'days': int(days),
            'timeframe': timeframe
        }
