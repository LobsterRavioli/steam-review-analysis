from pathlib import Path


def game_selection():
    """
    Prompts the user to select between a mini version (quick test) or a full version
    (complete dataset). If the full version is selected, make the user decide the game e returns
    the corrisponding folder.
    """
    print("Please choose the version to run:")
    print("1. Run Mini Version (for a quick test with a small dataset)")
    print("2. Run Full Version")

    foldername = ""

    while True:
        # Get user input and strip any surrounding whitespace
        choice = input("Enter '1' for Mini Version or '2' for Full Version: ").strip()
        # Validate the input and return the corresponding version
        if choice == '1':
            print(f"You have selected the \"mini\" version.")
            foldername = "mini"
            break
        elif choice == '2':
            print(f"You have selected the \"full\" version.")
            gameconfirmed = False
            while not gameconfirmed:
                # Specify the directory path
                directory_path = Path('../data')
                # List all subdirectories
                subfolders = [f.name for f in directory_path.iterdir() if f.is_dir()]

                # Display the list with indices
                for i, subfolder in enumerate(subfolders):
                    print(f"{i}: {subfolder}")

                # Ask the user to select an option by index
                while True:
                    try:
                        user_choice = int(input("Select the number corresponding "
                                                "to the game for which you want to analyze: "))
                        if 0 <= user_choice < len(subfolders):
                            confirmation = input(f"The game you selected is \"{subfolders[user_choice]}\". "
                                                 f"Is this correct? (Y/N): ").strip().upper()

                            if confirmation == "Y":
                                print("Game confirmed. Proceeding with the analysis...")
                                foldername = subfolders[user_choice]
                                gameconfirmed = True
                                break
                            elif confirmation == "N":
                                print("Let's try again. Please re-enter the correct number.")
                            else:
                                print("Invalid input. Please type 'Y' for Yes or 'N' for No.")
                        else:
                            print("Invalid choice. Please select a valid index.")
                    except ValueError:
                        print("Please enter a valid integer.")
                break
            break
        else:
            # Provide feedback on invalid input
            print("Invalid input. Please choose '1' for Mini Version or '2' for Full Version.")

    return foldername
