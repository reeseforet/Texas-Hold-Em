
import asyncio
import json
import websockets
from strategy import choose_strategy

#Our bots name and ID 
MY_ID = "groupM-poker-bot"
MY_NAME = "Group M Poker Bot"



# Choose an action using our Monte Carlo poker strategy
def choose_action(state):
    # Find our bot's information
    me = next(p for p in state["players"] if p["id"] == MY_ID)

    # Don't make decisions when a hand isn't being played
    if state["phase"] in ["WAITING", "SHOWDOWN"]:
        return None

    # Don't act if we're folded or already all-in
    if me["folded"] or me["all_in"]:
        return None

    # Make sure we have our two cards before making a decision
    if len(me["hole_cards"]) != 2:
        return None


    # Get our cards and the community cards
    my_cards = me["hole_cards"]
    community_cards = state["community_cards"]

    # Find the highest bet and how much we need to call
    highest_bet = max(p["bet"] for p in state["players"])
    to_call = max(0, highest_bet - me["bet"])

    # Count opponents who are still playing this hand
    active_opponents = [
        p for p in state["players"]
        if p["id"] != MY_ID
        and not p["folded"]
    ]

    num_opponents = len(active_opponents)
        # Avoid running a simulation when no opponents remain
    if num_opponents == 0:
        return "check", 0

    # Use our AI strategy to choose an action
    action = choose_strategy(
        my_cards,
        community_cards,
        pot=state["pot"],
        to_call=to_call,
        num_opponents=num_opponents
    )

    return action, 0



# Connect to the poker server using WebSockets
async def main():
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        print("Connected to the poker server!")

        # Create a message with our bot's information and status 
        join_message = {
            "type": "join",
            "name": MY_NAME,
            "player_id": MY_ID
        }

        await websocket.send(json.dumps(join_message))
        print("Join request sent!")

        # Remember our last action in case the server rejects it
        last_sent = None

        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["type"] == "state":
                state = data["payload"]
                print("Game phase:", state["phase"])
                print("Pot:", state["pot"])
                print("Current turn:", state["current_turn"])

                me = next(p for p in state["players"] if p["id"] == MY_ID)

                print("My chips:", me["stack"])
                print("My cards:", [card["label"] for card in me["hole_cards"]])

                # Check whether it is our bot's turn to act
                if state["current_turn"] == MY_ID:
                    print("It's my turn!")

                   
                    # Ask our strategy to choose an action
                    decision = choose_action(state)

                    # Only send an action if we received a valid decision
                    if decision is not None:
                        action, amount = decision

                        action_message = {
                            "type": "action",
                            "action": action,
                            "amount": amount
                        }

                        last_sent = action
                        await websocket.send(json.dumps(action_message))
                        print("Action sent:", action)



                else:
                    print("Waiting for my turn...")
     
            # Check whether the server accepted our action
            elif data["type"] == "action_result":
                result = data["payload"]

                # If our action was rejected, print the error
                if not result["ok"]:
                    print("Action rejected:", result["error"])

                    if last_sent != "call":
                        last_sent = "call"

                        await websocket.send(json.dumps({
                            "type": "action",
                            "action": "call"
                        }))




# Run the main function when this file is executed
if __name__ == "__main__":
    asyncio.run(main())
