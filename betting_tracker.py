
class BettingTracker:
    def __init__(self):
        self.hand_number = None
        self.phase = None
        self.previous_bet = 0
        self.minimum_raise = 20
        self.can_raise = True
        self.acted_since_full_raise = False
        self.raise_info_reliable = True


    def is_raise_action(self, state):
        last_action = state.get("last_action")

        if not isinstance(last_action, dict):
            return False

        return last_action.get("type") == "raise"

    def update(self, state):
        hand_number = state.get("hand_number")
        phase = state.get("phase")
        big_blind = state.get("big_blind", 20)
        current_bet = max(
            (p.get("bet", 0) for p in state.get("players", [])),
            default=0
        )

        new_round = hand_number != self.hand_number or phase != self.phase
        # Reset when a new hand or betting round begins
        if new_round:
            self.hand_number = hand_number
            self.phase = phase
            self.previous_bet = 0
            self.minimum_raise = big_blind
            self.can_raise = True
            self.acted_since_full_raise = False
            self.raise_info_reliable = True

        # Detect a new bet or raise
        
        if current_bet > self.previous_bet:
            raise_size = current_bet - self.previous_bet

            # Check for evidence of missed intermediate betting
            intermediate_bet = any(
                self.previous_bet < player.get("bet", 0) < current_bet
                for player in state.get("players", [])
            )

            
            # Ignore the initial blinds when checking for missed raises
            if intermediate_bet and not new_round:
                self.raise_info_reliable = False


            # If the bet increased but we cannot identify the action
            # that caused it, don't trust our raise estimate.
            last_action = state.get("last_action")

            
            # Initial blinds are expected at the start of a hand
            initial_blinds = (
                new_round
                and phase == "PREFLOP"
                and current_bet <= big_blind
            )

            # Only flag unexplained increases after the initial blinds
            if not initial_blinds:
                if (
                    not isinstance(last_action, dict)
                    or last_action.get("type") not in ("raise", "deal")
                ):
                    self.raise_info_reliable = False




            # Only update the minimum raise when we have
            # evidence that an actual raise occurred.
            if self.raise_info_reliable and self.is_raise_action(state):

                if raise_size >= self.minimum_raise:
                    # A full raise reopens betting
                    self.minimum_raise = raise_size
                    self.acted_since_full_raise = False
                    self.can_raise = True

                elif self.acted_since_full_raise:
                    # A short raise does not reopen betting
                    self.can_raise = False



        self.previous_bet = current_bet

        return {
            "current_bet": current_bet,
            "minimum_raise": self.minimum_raise,
            "can_raise": self.can_raise,
            "raise_info_reliable": self.raise_info_reliable,
            "raise_verified": False
        }

    

    def record_our_action(self, action):
        # Track actions that count as acting in a betting round
        if action in ("check", "call", "raise", "fold"):
            self.acted_since_full_raise = True


