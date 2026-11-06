
import asyncio
import json
import websockets

#Our bots name and ID 
MY_ID = "groupM-poker-bot"
MY_NAME = "Group M Poker Bot"


# Basic function to decide whether to check or call
def choose_action(state):
    me = next(p for p in state["players"] if p["id"] == MY_ID)

    highest_bet = max(p["bet"] for p in state["players"])
    to_call = highest_bet - me["bet"]

    if to_call == 0:
        return "check", 0
    else:
        return "call", 0


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

                    action, amount = choose_action(state)
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
