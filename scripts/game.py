import cv2
from utils import load_trained_model, detect_rps_sign, get_computer_choice, determine_winner, retrain_model_with_frame

def play_rps_game():
    model = load_trained_model()
    cap = cv2.VideoCapture(0)
    rounds = 5
    player_score = 0
    computer_score = 0

    for round_num in range(1, rounds + 1):
        print(f"Round {round_num}")
        print("Get ready...")
        for i in range(3, 0, -1):
            print(i)
            cv2.waitKey(1000)

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue

        player_choice = detect_rps_sign(model, frame)
        computer_choice = get_computer_choice()

        print(f"Player shows: {player_choice}")
        print(f"Computer shows: {computer_choice}")

        winner = determine_winner(player_choice, computer_choice)
        if winner == "Player":
            player_score += 1
        elif winner == "Computer":
            computer_score += 1

        print(f"Round {round_num} Winner: {winner}")
        print(f"Player: {player_score} - Computer: {computer_score}")

        # Retrain with the frame
        retrain_model_with_frame(model, frame, player_choice)

    cap.release()
    cv2.destroyAllWindows()

    if player_score > computer_score:
        print("You win the game!")
    elif player_score < computer_score:
        print("Computer wins the game!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    play_rps_game()
