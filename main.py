from framework.framework import AIFramework

def main():

    framework = AIFramework()

    print("=" * 60)
    print("Welcome to AI Framework")
    print("=" * 60)

    while True:

        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        response = framework.process_request(question)

        print(f"\nAssistant:\n{response}")


if __name__ == "__main__":
    main()