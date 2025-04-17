import questionary
from questionary import Choice, Separator
from UI.Constants import AppMode


class TradingBotCLI:
    def __init__(self):
        self.main_choices = [
            Choice("Fetch Data", AppMode.FETCH_DATA.value),
            Choice("Exit", "exit")
        ]

    def main_menu(self) -> str:
        return questionary.select(
            'Select an action:',
            choices=self.main_choices
        ).ask() or 'exit'
